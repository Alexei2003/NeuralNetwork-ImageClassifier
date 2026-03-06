import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from glob import glob
from PIL import Image
import onnxruntime as ort
from sklearn.metrics import precision_score, recall_score, f1_score
import time
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.backends.cudnn
from IPython.display import Audio, display
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import math

# ====================== КОНФИГУРАЦИЯ ======================
class Config:
    # выбор системы
    system = "google"
    if (system == "google"):
        dir_data = "/content/NeuralNetwork-ImageClassifier/DataSet/ARTS/"
        dir_save = "/content/drive/MyDrive/Colab Notebooks/NeuralNetwork-ImageClassifier/Model/"
    else:
        dir_data = "/kaggle/working/NeuralNetwork-ImageClassifier/DataSet/ARTS/"
        dir_save = "/kaggle/working/"

    # Пути к данным и моделям
    source_dir = dir_data + "Original"                # Папка с исходными изображениями
    checkpoint_path = dir_save + "best_model.pth"     # Путь для сохранения/загрузки модели
    labels_path = dir_save + "labels.txt"             # Файл с метками классов
    onnx_path = dir_save + "model.onnx"               # Путь для экспортированной модели в ONNX формате

    # Флаги управления обучением
    resume_training = False             # Продолжать обучение с сохраненного чекпоинта, если True

    # Параметры входных данных
    input_size = (224, 224)             # Размер входного изображения (ширина, высота)

    # Архитектура модели и гиперпараметры
    depth = 2                           # Количество ResidualBlock
    dropout = 0.1                       # Вероятность отключения нейронов (dropout)

    # Параметры обучения
    val_split = 0.2                     # Доля данных, выделяемая под валидацию
    gradient_clip = 1.0                 # Максимальная норма градиента
    batch_size = 256                    # Размер батча (число примеров, обрабатываемых за один проход)
    epochs = 100                        # Количество эпох обучения (полных проходов по всему датасету)
    focal_gamma = 1                     # Параметр гамма для Focal Loss, регулирует степень фокусировки на сложных примерах
    smoothing = 0.1                     # Параметр label smoothing, задаёт уровень сглаживания меток для улучшения обобщения
    mixed_precision = True              # Использовать смешанную точность (fp16) для ускорения обучения

    # Параметры LR
    max_lr = 0.0001                     # Максимальная скорость обучения
    min_lr = 0.0000000001               # Минимальная скорость обучения
    period_lr = 10                      # Период скорости обучения
    gamma_lr = 0.9                      # Скорость уменьшения max_lr

config = Config()

# ====================== КОМПОНЕНТЫ МОДЕЛИ ======================
class ECABlock(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 1D свёртка по каналам
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)  # [B, C, 1, 1]
        # Преобразуем в форму [B, 1, C] для 1D conv
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        return x * y.expand_as(x)

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.norm1 = nn.GroupNorm(32, in_channels)
        self.act1 = nn.GELU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)

        self.eca = ECABlock(out_channels)

        self.drop_path = DropPath(config.dropout)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act2(x)
        x = self.conv2(x)

        x = self.eca(x)

        return residual + self.drop_path(x)

class AnimeClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        def make_stage(in_channels, out_channels, num_blocks, first_stride=2):
            """Создает стадию с несколькими ResidualBlock"""
            blocks = []
            for i in range(num_blocks):
                stride = first_stride if i == 0 else 1
                blocks.append(ResidualBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    stride=stride
                ))
            return nn.Sequential(*blocks)

        # Базовые каналы (фиксированные или можно масштабировать отдельно)
        base_channels = [64, 128, 256, 512, 1024]
        self.backbone = nn.Sequential(
            # Stem
            nn.Conv2d(3, base_channels[0], 7, stride=2, padding=3, bias=False),
            nn.GroupNorm(32, base_channels[0]),
            nn.GELU(),

            # Стадии
            make_stage(base_channels[0], base_channels[1], config.depth, first_stride=2),
            make_stage(base_channels[1], base_channels[2], config.depth, first_stride=2),
            make_stage(base_channels[2], base_channels[3], config.depth, first_stride=2),
            make_stage(base_channels[3], base_channels[4], config.depth, first_stride=2),

            nn.AdaptiveAvgPool2d(1)
        )
        # Embedding с residual блоками
        hidden = base_channels[-1]
        self.embedding_block = nn.Sequential(
            # 1
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.LayerNorm(hidden * 2),
            nn.Dropout(p=config.dropout),

            # 2
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(p=config.dropout),

            # 3
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.LayerNorm(hidden // 2),
            nn.Dropout(p=config.dropout),

            # 4
            nn.Linear(hidden // 2, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(p=config.dropout),

            # 5
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.LayerNorm(hidden // 2),
            nn.Dropout(p=config.dropout),
        )

        # Классификатор теперь принимает backbone + embedding
        self.classifier = nn.Linear(hidden + hidden // 2, num_classes)

    def forward(self, x):
        x_backbone = self.backbone(x).flatten(1)  # [B, hidden]
        x_embed = self.embedding_block(x_backbone)  # [B, hidden//2]

        # соединяем backbone + embedding
        x = torch.cat([x_backbone, x_embed], dim=1)  # [B, hidden + hidden//2]

        return self.classifier(x)

# ====================== ОБРАБОТКА ДАННЫХ ======================
class ImageDataset(Dataset):
    def __init__(self, root, transform=None, mode='train'):
        # Всегда читаем классы из labels.txt
        if os.path.exists(config.labels_path):
            with open(config.labels_path, 'r') as f:
                self.classes = [line.strip() for line in f]
        else:
            # Если файла нет, берем из папки и создаем файл
            self.classes = sorted(os.listdir(root))
            with open(config.labels_path, 'w') as f:
                f.write('\n'.join(self.classes))

        self.samples = []
        for label, cls in enumerate(self.classes):
            cls_path = os.path.join(root, cls)
            # Проверяем, существует ли папка
            if os.path.exists(cls_path):
                self.samples.extend([(f, label) for f in glob(os.path.join(cls_path, '*'))])
            else:
                print(f"⚠️  Папка класса '{cls}' не найдена, пропускаем")

        self.transform = transform or self._get_transforms(mode)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = np.array(Image.open(img_path).convert('RGB'))

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']

        return img, label

    @staticmethod
    def _get_transforms(mode):
        if mode == 'train':
            return A.Compose([
                # ===================== ВСЕГДА ПРИМЕНЯЕТСЯ (p=1.0) =====================
                A.RandomResizedCrop(             # 📏 СЛУЧАЙНОЕ КАДРИРОВАНИЕ С ИЗМЕНЕНИЕМ РАЗМЕРА
                    size=config.input_size,      # Фиксированный выходной размер (224x224)
                    scale=(0.5, 1.0),            # Масштаб: 80-100% от исходного изображения
                                                # (меньше значение → больше обрезка)
                    interpolation=1,             # Метод интерполяции (1=билинейная)
                    p=1.0                        # ✅ ВСЕГДА применяется - основная аугментация
                ),

                # ===================== ЧАСТО ПРИМЕНЯЕТСЯ (p=0.5) =====================
                A.Rotate(limit=90, p=0.5),      # 🔄 ПОВОРОТ
                                                # limit=30: случайный поворот ±30 градусов
                                                # Учит модель распознавать объекты под разными углами

                A.HorizontalFlip(p=0.5),         # ↔️ ГОРИЗОНТАЛЬНОЕ ОТРАЖЕНИЕ
                                                # Лево-правая симметрия, часто встречается в изображениях

                A.ColorJitter(                   # 🎨 ЦВЕТОВЫЕ ИСКАЖЕНИЯ
                    brightness=0.5,              # Яркость: ±20% (0.8-1.2 от исходной)
                    contrast=0.5,                # Контраст: ±20%
                    saturation=0.5,              # Насыщенность: ±20%
                    hue=0.5,                     # Оттенок: ±0.2 (в нормированных единицах HSV)
                                                # Меняет цвета, имитирует разные условия освещения
                    p=0.5                        # 50% вероятность применения
                ),

                A.Affine(                           # 📐 АФФИННЫЕ ПРЕОБРАЗОВАНИЯ
                    translate_percent=(-0.2, 0.2),  # Сдвиг: ±10% от размеров изображения
                    keep_ratio=True,                # Сохранять соотношение сторон при сдвиге
                    p=0.5                           # 50% вероятность
                ),

                # ===================== РЕДКО ПРИМЕНЯЕТСЯ (p=0.2) =====================
                A.GaussianBlur(                  # 🌫 ГАУССОВО РАЗМЫТИЕ
                    blur_limit=(5, 5),           # Размер ядра размытия: 3×3 пикселя
                                                # (нечетное число, больше → сильнее размытие)
                    p=0.2                        # 20% вероятность - имитирует расфокус
                ),

                # ===================== ОЧЕНЬ РЕДКО (p=0.1) =====================
                A.Sharpen(                       # 🔪 ПОВЫШЕНИЕ РЕЗКОСТИ
                    alpha=(0.2, 0.5),            # Сила эффекта: 20-50%
                                                # (0=нет эффекта, 1=максимальная резкость)
                    lightness=(0.5, 1.0),        # Яркость: 50-100%
                    p=0.1                        # 10% вероятность - имитирует "острый" стиль
                ),

                # ===================== ВСЕГДА В КОНЦЕ (без p) =====================
                A.ToFloat(max_value=255.0),      # 🔢 КОНВЕРТАЦИЯ В FLOAT [0, 1]
                                                # max_value=255: делит на 255 для нормализации

                ToTensorV2(),                    # ⚡ КОНВЕРТАЦИЯ В ТЕНЗОР PYTORCH
                                                # Меняет порядок осей HWC → CHW
            ])
        return A.Compose([
            A.Resize(config.input_size[0], config.input_size[1]),  # Добавили Resize для валидации
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ])

# ====================== ОБУЧЕНИЕ ======================
class CosineDecayRestarts():
    def __init__(
        self,
        optimizer,
        first_cycle_steps,
        max_lr,
        min_lr,
        gamma
    ):
        self.optimizer = optimizer

        self.first_cycle_steps = first_cycle_steps
        self.cycle_steps = first_cycle_steps
        self.step_in_cycle = 0
        self.cycle = 0

        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.gamma = gamma

    def step(self):
        self.step_in_cycle += 1
        if self.step_in_cycle >= self.cycle_steps:

            self.cycle += 1
            self.step_in_cycle = 0

            # уменьшаем max_lr
            self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)

        cos = (1 + math.cos(math.pi * self.step_in_cycle / self.cycle_steps)) / 2
        lr = self.min_lr + (self.max_lr - self.min_lr) * cos

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_last_lr(self):
        return [self.optimizer.param_groups[0]['lr']]

    def state_dict(self):
        return {
            k: v for k, v in self.__dict__.items()
            if k != "optimizer"
        }

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    """
    Смешивает левую половину одного изображения с правой половиной другого
    """
    batch_size, _, h, w = x.size()
    index = torch.randperm(batch_size).to(x.device)

    # Выбираем точку разделения (может быть случайной)
    split_point = np.random.randint(int(0.4 * w), int(0.6 * w))

    # Создаём смешанное изображение
    mixed_x = x.clone()

    # ЛЕВАЯ часть от исходного изображения
    mixed_x[:, :, :, :split_point] = x[:, :, :, :split_point]

    # ПРАВАЯ часть от другого изображения
    mixed_x[:, :, :, split_point:] = x[index, :, :, split_point:]

    y_a, y_b = y, y[index]
    lam = split_point / w

    return mixed_x, y_a, y_b, lam

def get_class_weights_from_dirs(root_dir, class_names):
    class_counts = []
    for class_name in class_names:
        path = os.path.join(root_dir, class_name)
        count = len(glob(os.path.join(path, "*")))
        class_counts.append(count)

    total = sum(class_counts)
    weights = [total / (count + 1e-6) for count in class_counts]  # защита от деления на 0
    weights = torch.tensor(weights)
    weights = weights / weights.max()  # нормализация
    return weights

def focal_loss_with_smoothing(outputs, targets, class_weights=None):
    num_classes = outputs.size(1)
    confidence = 1.0 - config.smoothing

    log_probs = torch.nn.functional.log_softmax(outputs, dim=-1)
    probs = torch.exp(log_probs)

    true_dist = torch.full_like(log_probs, config.smoothing / (num_classes - 1))
    true_dist.scatter_(1, targets.unsqueeze(1), confidence)

    pt = torch.sum(true_dist * probs, dim=-1)
    focal_factor = (1 - pt).pow(config.focal_gamma)
    loss = -torch.sum(true_dist * log_probs, dim=-1)

    if class_weights is not None:
        weights = class_weights[targets]
        loss = loss * weights

    return torch.mean(focal_factor * loss)

def forward_with_mixup_cutmix(model, inputs, labels, class_weights, device):
    inputs, labels = inputs.to(device), labels.to(device)

    use_mix = np.random.rand() < 0.50
    use_cutmix = np.random.rand() < 0.5

    if use_mix:
        if use_cutmix:
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, alpha=1.0)
        else:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=1.0)

        with torch.amp.autocast('cuda', enabled=config.mixed_precision):
            outputs = model(inputs)
            loss = lam * focal_loss_with_smoothing(outputs, targets_a, class_weights)\
                 + (1 - lam) * focal_loss_with_smoothing(outputs, targets_b, class_weights)
    else:
        with torch.amp.autocast('cuda', enabled=config.mixed_precision):
            outputs = model(inputs)
            loss = focal_loss_with_smoothing(outputs, labels, class_weights)

    return outputs, loss

def save_checkpoint(model, epoch, optimizer, scheduler, path):
    """
    Сохранение чекпоинта без сжатия
    """
    # 1. Создаём стандартный чекпоинт
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }

    # 2. Сохраняем без сжатия
    torch.save(checkpoint, path, pickle_protocol=5)  # Протокол 5 для эффективности

    # 3. Показываем размер файла
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"[System]  Чекпоинт сохранен: {size_mb:.1f} MB")

    return size_mb

def load_checkpoint(model, optimizer, scheduler, path, device):
    """
    Загрузка чекпоинта
    """
    try:
        # 1. Загружаем напрямую (без gzip)
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        # 2. Загружаем веса модели напрямую (типы сохраняются автоматически)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        # 3. Восстанавливаем optimizer и scheduler
        if optimizer and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            if hasattr(scheduler, 'load_state_dict'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                scheduler.optimizer = optimizer
            else:
                print("⚠️  Scheduler не поддерживает load_state_dict")

        print(f"✓ Чекпоинт загружен")
        print(f"  Эпоха: {checkpoint['epoch']}")

        return {
            'model': model,
            'epoch': checkpoint['epoch'],
            'optimizer': optimizer,
            'scheduler': scheduler,
        }

    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        import traceback
        traceback.print_exc()
        return None

def make_sound():
    # Генерируем короткий "бип" (440 Гц, 0.2 секунды)
    sample_rate = 44100
    duration = 0.2  # секунды
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_signal = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440 Гц - нота "Ля"

    # Воспроизводим автоматически
    display(Audio(audio_signal, rate=sample_rate, autoplay=True))

def create_learning(model):
    optimizer = optim.AdamW(model.parameters(), lr=config.max_lr)

    scheduler = CosineDecayRestarts(
        optimizer=optimizer,
        first_cycle_steps=config.period_lr,
        max_lr=config.max_lr,
        min_lr=config.min_lr,
        gamma=config.gamma_lr
    )

    return optimizer, scheduler

def run_training():
    # Оптимизация матричных операций
    torch.set_float32_matmul_precision('medium')

    # Включение оптимизации cuDNN
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)

    full_classes = sorted(os.listdir(config.source_dir))

    # Сохраняем/перезаписываем список классов
    with open(config.labels_path, 'w') as f:
        f.write('\n'.join(full_classes))

    print(f"📊 Количество классов: {len(full_classes)}")
    print(f"🚀 Конфигурация обучения:")
    print(f"  • Max LR: {config.max_lr:.10f}")
    print(f"  • Min LR: {config.min_lr:.10f}")
    print(f"  • Период: {config.period_lr}")
    print(f"  • Всего эпох: {config.epochs}")

    # Создаем датасет с фиксированным порядком классов
    full_dataset = ImageDataset(config.source_dir)
    full_dataset.classes = full_classes

    train_size = int((1 - config.val_split) * len(full_dataset))
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, len(full_dataset) - train_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        prefetch_factor=4,
        pin_memory=True)
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        num_workers=2,
        persistent_workers=True,
        prefetch_factor=4,
        pin_memory=True)

    model = AnimeClassifier(len(full_classes)).to(device)

    class_weights = get_class_weights_from_dirs(config.source_dir, full_classes).to(device)

    optimizer, scheduler = create_learning(model)

    scaler = torch.amp.GradScaler('cuda', enabled=config.mixed_precision and torch.cuda.is_available())
    start_epoch = 1

    if config.resume_training:
        loaded = load_checkpoint(model, optimizer, scheduler, config.checkpoint_path, device)

        if loaded is not None:
            model = loaded['model']
            optimizer = loaded['optimizer']
            scheduler = loaded['scheduler']
            start_epoch = loaded['epoch'] + 1
            print(f"🔄 Продолжение обучения с эпохи {start_epoch}")
            print(f"  Текущий LR: {optimizer.param_groups[0]['lr']:.10f}")
        else:
            print("❌ Не удалось загрузить чекпоинт, начинаем обучение с нуля")
            save_checkpoint(
                model=model,
                epoch=-1,
                optimizer=optimizer,
                scheduler=scheduler,
                path=config.checkpoint_path
            )
    else:
        save_checkpoint(
            model=model,
            epoch=-1,
            optimizer=optimizer,
            scheduler=scheduler,
            path=config.checkpoint_path
        )

    start_time = time.time()
    train_loader_len = len(train_loader)
    val_loader_len = len(val_loader)

    for epoch in range(start_epoch, config.epochs+1):
        make_sound()

        model.train()
        train_loss = 0.0
        train_correct, train_total = 0, 0
        optimizer.zero_grad(set_to_none=True)
        epoch_start_time = time.time()

        current_lr = scheduler.get_last_lr()[0]
        print(f"[LR] Current: {current_lr:.10f}")

        batch_start_time = time.time()
        optimizer.zero_grad()

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs, loss = forward_with_mixup_cutmix(model, inputs, labels, class_weights, device)

            # Сохраняем значение loss для логирования
            raw_loss = loss.item()

            scaler.scale(loss).backward()
            train_loss += raw_loss   # добавляем исходный loss (до деления)

            _, predicted = torch.max(outputs, 1)
            current_batch_size = labels.size(0)
            train_total += current_batch_size
            train_correct += (predicted == labels).sum().item()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # Расчёт времени и вывод (без изменений)
            batch_duration = (time.time() - batch_start_time) / (batch_idx + 1)
            remaining_batches = train_loader_len - (batch_idx + 1)
            estimated_remaining_time = remaining_batches * batch_duration
            remaining_time_str = time.strftime('%H:%M:%S', time.gmtime(estimated_remaining_time))

            print(
                f"\r[Train] Epoch {epoch}/{config.epochs} | Batch {batch_idx+1}/{train_loader_len} | "
                f"Loss: {(raw_loss):.4f} | Remaining time: {remaining_time_str}",
                end='', flush=True)

        train_accuracy = 100 * train_correct / train_total
        print()

        # Валидация
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []

        batch_start_time = time.time()
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=config.mixed_precision):
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(device, non_blocking=True)  # Асинхронная передача
                labels = labels.to(device, non_blocking=True)

                outputs = model(inputs)

                loss = focal_loss_with_smoothing(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                batch_duration = (time.time() - batch_start_time) / (batch_idx + 1)
                remaining_batches = val_loader_len - (batch_idx + 1)
                estimated_remaining_time = remaining_batches * batch_duration
                remaining_time_str = time.strftime('%H:%M:%S', time.gmtime(estimated_remaining_time))

                print(
                    f"\r[Val]   Epoch {epoch}/{config.epochs} | Batch {batch_idx+1}/{val_loader_len} | "
                    f"Loss: {loss.item():.4f} | Remaining: {remaining_time_str}",
                    end='', flush=True)

        print()

        # Расчет метрик
        val_accuracy = 100 * val_correct / val_total
        val_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        val_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        total_elapsed_time = epoch_end_time - start_time
        epoch_duration_str = time.strftime("%H:%M:%S", time.gmtime(epoch_duration))
        total_elapsed_str = time.strftime("%H:%M:%S", time.gmtime(total_elapsed_time))

        scheduler.step()
        next_lr = scheduler.get_last_lr()[0]

        save_checkpoint(
            model=model,
            epoch=epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            path=config.checkpoint_path
        )

        # Логирование
        print(f"[Summary]   Train Loss: {train_loss/train_loader_len:.4f} | Acc: {train_accuracy:.2f}%")
        print(f"[Summary]   Val   Loss: {val_loss/val_loader_len:.4f} | Acc: {val_accuracy:.2f}%")
        print(f"[Summary]   Val Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}")
        print(f"[Time]      Epoch: {epoch_duration_str} | Total: {total_elapsed_str}")
        print(f"[LR]        Current: {current_lr/config.max_lr:.2f} %")
        print(f"[LR]        Next:    {next_lr/config.max_lr:.2f} %")

        print()

# ====================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ======================
def convert_to_onnx():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(config.labels_path) as f:
        classes = [line.strip() for line in f]

    model = AnimeClassifier(len(classes)).to(device)
    optimizer, scheduler = create_learning(model)

    loaded = load_checkpoint(model, optimizer, scheduler, config.checkpoint_path, device)

    if loaded is None:
        print("❌ Не удалось загрузить чекпоинт для конвертации в ONNX!")
        return

    model = loaded['model']
    model.eval()

    print(f"✅ Модель загружена для ONNX экспорта")
    print(f"   Классов: {len(classes)}")

    dummy_input = torch.randn(1, 3, *config.input_size).to(device)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            config.onnx_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            do_constant_folding=True,
            opset_version=14,
            training=torch.onnx.TrainingMode.EVAL,
            verbose=False
        )
        print(f"✅ ONNX модель сохранена: {config.onnx_path}")

        if os.path.exists(config.onnx_path):
            size_mb = os.path.getsize(config.onnx_path) / 1024 / 1024
            print(f"   Размер ONNX файла: {size_mb:.2f} MB")

    except Exception as e:
        print(f"❌ Ошибка при экспорте в ONNX: {e}")
        import traceback
        traceback.print_exc()

def test_onnx():
    if not os.path.exists(config.onnx_path):
        print("❌ ONNX модель не найдена!")
        print(f"   Путь: {config.onnx_path}")
        return

    try:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']

        session = ort.InferenceSession(config.onnx_path, options, providers=providers)
        print("✅ ONNX Runtime сессия создана")
    except Exception as e:
        print(f"❌ Ошибка загрузки ONNX модели: {e}")
        return

    test_image_path = os.path.join("/content/NeuralNetwork-ImageClassifier/", "test.jpg")
    if not os.path.exists(test_image_path):
        print(f"❌ Тестовое изображение не найдено: {test_image_path}")
        print("   Создайте файл test.jpg в папке проекта")
        return

    try:
        img = Image.open(test_image_path).convert('RGB')
        img_np = np.array(img)

        transform = ImageDataset._get_transforms('val')
        augmented = transform(image=img_np)
        img_tensor = augmented['image'].unsqueeze(0)

        print(f"✅ Изображение загружено: {img.size[0]}x{img.size[1]}")
    except Exception as e:
        print(f"❌ Ошибка загрузки изображения: {e}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(config.labels_path) as f:
        classes = [line.strip() for line in f]

    model = AnimeClassifier(len(classes)).to(device)

    optimizer, scheduler = create_learning(model)

    loaded = load_checkpoint(model, optimizer, scheduler, config.checkpoint_path, device)

    if loaded is None:
        print("❌ Не удалось загрузить PyTorch модель для сравнения!")
        return

    model = loaded['model']
    model.eval()

    with torch.no_grad():
        pytorch_output = model(img_tensor.to(device))
        pytorch_probs = torch.softmax(pytorch_output, dim=1).cpu()

    try:
        onnx_input = img_tensor.numpy().astype(np.float32)
        onnx_outputs = session.run(None, {'input': onnx_input})
        onnx_probs = torch.softmax(torch.tensor(onnx_outputs[0]), dim=1)

        print("✅ ONNX inference выполнен успешно")
    except Exception as e:
        print(f"❌ Ошибка ONNX inference: {e}")
        return

    print("\n" + "="*50)
    print("[PyTorch] Топ-5 предсказаний:")
    pytorch_top_probs, pytorch_top_indices = torch.topk(pytorch_probs, 5)
    for i, (prob, idx) in enumerate(zip(pytorch_top_probs[0], pytorch_top_indices[0])):
        print(f"{i+1}. {classes[idx]}: {prob.item()*100:.2f}%")

    print("\n[ONNX] Топ-5 предсказаний:")
    onnx_top_probs, onnx_top_indices = torch.topk(onnx_probs, 5)
    for i, (prob, idx) in enumerate(zip(onnx_top_probs[0], onnx_top_indices[0])):
        print(f"{i+1}. {classes[idx]}: {prob.item()*100:.2f}%")

    diff = torch.max(torch.abs(pytorch_probs - onnx_probs)).item()
    print(f"\n[Сравнение] Расхождение между выходами: {diff:.6f}")

    if diff < 0.001:
        print("✅ Конвертация успешна! Расхождение < 0.001")
    elif diff < 0.01:
        print("⚠️  Небольшое расхождение (0.001-0.01), возможно из-за численной точности")
    else:
        print("❌ Большое расхождение (> 0.01)! Возможная ошибка конвертации")

    print("\n" + "="*50)
    print(f"PyTorch device: {device}")
    print(f"PyTorch dtype: {pytorch_probs.dtype}")
    print(f"ONNX dtype: {onnx_probs.dtype}")
    print(f"Количество классов: {len(classes)}")

def count_parameters_by_module():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if os.path.exists(config.labels_path):
        with open(config.labels_path, 'r') as f:
            classes = [line.strip() for line in f if line.strip()]
    else:
        classes = [f"class_{i}" for i in range(1000)]

    model = AnimeClassifier(len(classes)).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Всего параметров: {total_params:,}")

    print("Параметры по группам:")
    backbone_params = 0
    embedding_params = 0
    classifier_params = 0
    for name, param in model.named_parameters():
        num = param.numel()
        if name.startswith('backbone'):
            backbone_params += num
        if name.startswith('embedding'):
            embedding_params += num
        elif name.startswith('classifier'):
            classifier_params += num
    print(f"Backbone: {backbone_params:,}")
    print(f"Embedding: {embedding_params:,}")
    print(f"Classifier: {classifier_params:,}")
    print(f"Total: {(backbone_params + embedding_params + classifier_params):,}")

    # Процентное соотношение
    print(f"\nПроцентное распределение:")
    print(f"Backbone: {backbone_params/total_params*100:.2f}%")
    print(f"Embedding: {embedding_params/total_params*100:.2f}%")
    print(f"Classifier: {classifier_params/total_params*100:.2f}%\n")

# ====================== ИНТЕРФЕЙС ======================
def main_menu():
    make_sound()
    while True:
        print("\n" + "="*50)
        print("🚀 Anime Classifier")
        print("="*50)
        print("1. Обучить модель (с нуля)")
        print("2. Продолжить обучение")
        print("3. Конвертировать в ONNX")
        print("4. Протестировать ONNX")
        print("5. Размер модели")
        choice = input("Выбор: ").strip()

        if choice == '1':
            config.resume_training = False
            count_parameters_by_module()
            run_training()
        elif choice == '2':
            if not os.path.exists(config.checkpoint_path):
                print("❌ Чекпоинт для продолжения не найден!")
                continue
            config.resume_training = True
            run_training()
        elif choice == '3':
            convert_to_onnx()
        elif choice == '4':
            test_onnx()
        elif choice == '5':
            count_parameters_by_module()
        elif choice == '0':
            break
        else:
            print("Неверный ввод!")

if __name__ == "__main__":
    main_menu()