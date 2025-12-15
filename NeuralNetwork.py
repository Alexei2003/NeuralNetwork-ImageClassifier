import torch
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
from torchinfo import summary
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.backends.cudnn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gzip

# ====================== КОНФИГУРАЦИЯ ======================
class Config:
    # выбор системы
    system = "colab"

    match system:
        case "my":
            notebook = False
            dir = "/media/alex/Games/WPS/NeuralNetwork-ImageClassifier/"
        case "colab":
            notebook = True
            dir = "/content/NeuralNetwork-ImageClassifier/"

    # Пути к данным и моделям
    source_dir = dir + "DataSet/ARTS/Original"         # Папка с исходными изображениями
    checkpoint_path = dir + "Model/best_model.pth"     # Путь для сохранения/загрузки модели
    labels_path = dir + "Model/labels.txt"             # Файл с метками классов
    onnx_path = dir + "Model/model.onnx"               # Путь для экспортированной модели в ONNX формате

    # Флаги управления обучением
    resume_training = False         # Продолжать обучение с сохраненного чекпоинта, если True

    # Параметры входных данных
    input_size = (224, 224)         # Размер входного изображения (ширина, высота)

    # Архитектура модели и гиперпараметры
    num_experts = 32                # Количество экспертов в MoE (Mixture of Experts)
    expert_units = 1024             # Количество нейронов в каждом эксперте
    k_top_expert = 8                # Количество активных экспертов на один пример
    se_reduction = 16               # Коэффициент редукции для SE (Squeeze-and-Excitation) блока
    dropout = 0.5                   # Вероятность отключения нейронов (dropout)

    # Параметры обучения
    lr = 0.002                      # Начальная скорость обучения (learning rate)
    batch_size = 512                # Размер батча (число примеров, обрабатываемых за один проход)
    epochs = 100                    # Количество эпох обучения (полных проходов по всему датасету)
    focal_gamma = 5                 # Параметр гамма для Focal Loss, регулирует степень фокусировки на сложных примерах
    smoothing = 0.1                 # Параметр label smoothing, задаёт уровень сглаживания меток для улучшения обобщения

    # Настройки оптимизации и контроля обучения
    mixed_precision = True          # Использовать смешанную точность (fp16) для ускорения обучения
    early_stopping_patience = 5     # Количество эпох без улучшения для ранней остановки
    val_split = 0.2                 # Доля данных, выделяемая под валидацию
    factor_lr = 0.5                 # Коэффициент уменьшения learning rate при plateau

config = Config()

# ====================== КОМПОНЕНТЫ МОДЕЛИ ======================
class MoE(nn.Module):
    def __init__(self, input_dim, num_experts, base_expert_units, k_top):
        super().__init__()
        self.num_experts = num_experts
        self.k_top = k_top
        self.experts = nn.ModuleList()

        # Создаем список размеров экспертов от 0.5 до 1.5 от базового
        expert_sizes = []
        for i in range(num_experts):
            # Линейная интерполяция от 0.5 до 1.5
            scale = 0.5 + (i / (num_experts - 1)) if num_experts > 1 else 1.0
            size = int(base_expert_units * scale)
            expert_sizes.append(size)

        print(f"MoE expert sizes: {expert_sizes}")

        for size in expert_sizes:
            self.experts.append(nn.Sequential(
                nn.Linear(input_dim, size),
                nn.BatchNorm1d(size),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout),
                nn.Linear(size, input_dim),
                nn.BatchNorm1d(input_dim)
            ))

        self.router = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        logits = self.router(x)
        top_k_weights, top_k_indices = logits.topk(self.k_top, dim=1)
        top_k_weights = torch.softmax(top_k_weights, dim=1)

        # Собираем выходы всех экспертов
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, num_experts, D]

        # Создаем маску для выбранных экспертов
        mask = torch.zeros_like(expert_outputs)
        mask = torch.scatter(
            mask,
            1,
            top_k_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.size(-1)),
            1.0
        )

        # Объединяем градиенты только для выбранных экспертов
        expert_outputs = expert_outputs * mask + (expert_outputs * (1 - mask)).detach()

        # Выбираем топ-k экспертов
        selected_outputs = expert_outputs.gather(
            1,
            top_k_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.size(-1))
        )

        # Взвешенное суммирование
        output = (selected_outputs * top_k_weights.unsqueeze(-1)).sum(dim=1)
        return output + x

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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.eca = ECABlock(out_channels)  # 🔹 заменили SE на ECA
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(config.dropout)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x = self.eca(x)  # 🔹 используем ECA
        return self.act(x + residual)

class AnimeClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 512, stride=2),
            nn.AdaptiveAvgPool2d(1)
        )
        self.moe = MoE(512, config.num_experts, config.expert_units, config.k_top_expert)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x).flatten(1)
        x = self.moe(x)
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
        img = np.array(Image.open(img_path).convert('RGB'))  # Конвертация в numpy array

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']

        return img, label

    @staticmethod
    def _get_transforms(mode):
        if mode == 'train':
            return A.Compose([
                A.Rotate(limit=30, p=0.5),
                A.RandomResizedCrop(
                    size=config.input_size,
                    scale=(0.8, 1.0),
                    ratio=(0.75, 1.33),          # Опционально (по умолчанию (0.75, 1.33))
                    interpolation=1,             # BILINEAR
                    p=1.0
                ),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.0,                     # Обязательный параметр
                    p=0.5
                ),
                A.GaussianBlur(blur_limit=(3, 3), p=0.2),
                A.Affine(
                    translate_percent=(-0.1, 0.1),
                    keep_ratio=True,
                    p=0.5
                ),
                A.ToFloat(max_value=255.0),
                ToTensorV2(),
            ])
        return A.Compose([
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ])

# ====================== ОБУЧЕНИЕ ======================
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
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size, _, h, w = x.size()
    index = torch.randperm(batch_size).to(x.device)

    cx = np.random.randint(w)
    cy = np.random.randint(h)
    cut_w = int(w * np.sqrt(1 - lam))
    cut_h = int(h * np.sqrt(1 - lam))

    x1 = np.clip(cx - cut_w // 2, 0, w)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    y2 = np.clip(cy + cut_h // 2, 0, h)

    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    y_a, y_b = y, y[index]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
    return x, y_a, y_b, lam

def get_class_weights_from_dirs(root_dir, class_names):
    class_counts = []
    for class_name in class_names:
        path = os.path.join(root_dir, class_name)
        count = len(glob(os.path.join(path, "*")))
        class_counts.append(count)

    total = sum(class_counts)
    weights = [total / (count + 1e-6) for count in class_counts]  # защита от деления на 0
    weights = torch.tensor(weights)
    weights = weights / weights.mean()  # нормализация
    return weights

def focal_loss_with_smoothing(outputs, targets, gamma=5.0, smoothing=0.1, class_weights=None):
    num_classes = outputs.size(1)
    confidence = 1.0 - smoothing

    log_probs = torch.nn.functional.log_softmax(outputs, dim=-1)
    probs = torch.exp(log_probs)

    true_dist = torch.full_like(log_probs, smoothing / (num_classes - 1))
    true_dist.scatter_(1, targets.unsqueeze(1), confidence)

    pt = torch.sum(true_dist * probs, dim=-1)
    focal_factor = (1 - pt).pow(gamma)
    loss = -torch.sum(true_dist * log_probs, dim=-1)

    if class_weights is not None:
        weights = class_weights[targets]
        loss = loss * weights

    return torch.mean(focal_factor * loss)

def imshow(img_tensor, title=None):
    # img_tensor: Tensor с форматом (C, H, W)
    # Преобразуем тензор в numpy для matplotlib и нормализуем к [0,1]
    img = img_tensor.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # C,H,W -> H,W,C
    img = np.clip(img, 0, 1)  # Чтобы избежать проблем с цветами

    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.savefig('output_image.png')
    plt.close()

def forward_with_mixup_cutmix(model, inputs, labels, config, class_weights, device):
    inputs, labels = inputs.to(device), labels.to(device)

    use_mix = np.random.rand() < 0.50
    use_cutmix = np.random.rand() < 0.5

    if use_mix:
        if use_cutmix:
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, alpha=1.0)
        else:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=1.0)

        #imshow(inputs[1], title="Original image")

        with torch.amp.autocast('cuda', enabled=config.mixed_precision):
            outputs = model(inputs)
            loss = lam * focal_loss_with_smoothing(outputs, targets_a, config.focal_gamma, config.smoothing, class_weights)\
                 + (1 - lam) * focal_loss_with_smoothing(outputs, targets_b, config.focal_gamma, config.smoothing, class_weights)
    else:
        with torch.amp.autocast('cuda', enabled=config.mixed_precision):
            outputs = model(inputs)
            loss = focal_loss_with_smoothing(outputs, labels, config.focal_gamma, config.smoothing, class_weights)

    return outputs, loss

def compile_model(model):
    torch.compile(model,
        mode="max-autotune",
        dynamic=False,
        fullgraph=False)
    torch.cuda.empty_cache()

def save_compressed_checkpoint(model, epoch, best_loss, lr, path):
    """
    Умное сжатие с учетом mixed precision
    """
    # 1. Определяем, какие веса можно сжимать
    checkpoint = {
        'epoch': epoch,
        'best_loss': best_loss,
        'learning_rate': lr,
    }

    # 2. Сжимаем ВСЕ веса в float16 (даже если они float32)
    compressed_weights = {}
    for name, param in model.state_dict().items():
        if param.is_floating_point():
            # Принудительно в float16 для сжатия
            compressed_weights[name] = param.half().clone()  # Важно: .clone()
        else:
            compressed_weights[name] = param

    checkpoint['model_state_dict'] = compressed_weights

    # 3. Сохраняем с максимальным сжатием
    with gzip.open(path, 'wb', compresslevel=9) as f:
        torch.save(checkpoint, f, pickle_protocol=4)

    # 4. Показываем результат сжатия
    size_mb = os.path.getsize(path) / 1024 / 1024

    # Сравниваем с размером без сжатия
    temp_path = path + '.tmp'
    torch.save(checkpoint, temp_path)  # Без сжатия
    uncompressed_size = os.path.getsize(temp_path) / 1024 / 1024
    os.remove(temp_path)

    compression_ratio = (1 - size_mb / uncompressed_size) * 100

    print(f"[System]  Чекпоинт сохранен: {size_mb:.1f} MB")
    print(f"[System]  Сжатие: {compression_ratio:.0f}% от {uncompressed_size:.1f} MB")

    return size_mb

def load_compressed_checkpoint(model, path, device):
    """
    Загрузка с автоматическим восстановлением типов данных
    """
    try:
        # 1. Загружаем
        with gzip.open(path, 'rb') as f:
            checkpoint = torch.load(f, map_location='cpu', weights_only=False)

        # 2. Получаем сохраненные веса
        saved_weights = checkpoint['model_state_dict']
        current_weights = model.state_dict()
        loaded_weights = {}

        # 3. Восстанавливаем с правильными типами данных
        for name in current_weights.keys():
            if name in saved_weights:
                saved_param = saved_weights[name]
                current_param = current_weights[name]

                if saved_param.is_floating_point() and current_param.is_floating_point():
                    # Восстанавливаем в оригинальный dtype модели
                    loaded_weights[name] = saved_param.to(current_param.dtype)
                else:
                    loaded_weights[name] = saved_param
            else:
                # Если веса не найдены, оставляем как есть
                loaded_weights[name] = current_weights[name]
                print(f"⚠️  Пропущен параметр: {name}")

        # 4. Загружаем в модель
        model.load_state_dict(loaded_weights)
        model.to(device)

        print(f"✓ Чекпоинт загружен")
        print(f"  Эпоха: {checkpoint['epoch']}")
        print(f"  Best loss: {checkpoint['best_loss']:.4f}")
        print(f"  LR: {checkpoint['learning_rate']:.6f}")

        return {
            'model': model,
            'epoch': checkpoint['epoch'],
            'best_loss': checkpoint['best_loss'],
            'learning_rate': checkpoint['learning_rate'],
        }

    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_training():
    # Оптимизация матричных операций (НОВОЕ)
    torch.set_float32_matmul_precision('medium')

    # Включение оптимизации cuDNN
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)

    full_classes = sorted(os.listdir(config.source_dir))

    # Сохраняем/перезаписываем список классов
    with open(config.labels_path, 'w') as f:
        f.write('\n'.join(full_classes))

    print(f"📊 Количество классов: {len(full_classes)}")

    # Создаем датасет с фиксированным порядком классов
    full_dataset = ImageDataset(config.source_dir)
    full_dataset.classes = full_classes  # Переопределяем порядок

    train_size = int((1 - config.val_split) * len(full_dataset))
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, len(full_dataset) - train_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=True)
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        num_workers=os.cpu_count(),
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=True)

    model = AnimeClassifier(len(full_classes)).to(device)

    class_weights = get_class_weights_from_dirs(config.source_dir, full_classes).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    scaler = torch.amp.GradScaler('cuda', enabled=config.mixed_precision and torch.cuda.is_available())
    start_epoch = 0
    best_loss = float('inf')
    early_stop_counter = 0

    if config.resume_training:
        # Просто загружаем сжатый чекпоинт
        loaded = load_compressed_checkpoint(model, config.checkpoint_path, device)

        if loaded is not None:
            # Получаем модель из чекпоинта
            model = loaded['model']

            # Создаем оптимизатор с сохраненным LR
            optimizer = optim.AdamW(model.parameters(), lr=loaded['learning_rate'])

            # Восстанавливаем состояние обучения
            start_epoch = loaded['epoch'] + 1
            best_loss = loaded['best_loss']

            scaler = torch.amp.GradScaler('cuda', enabled=config.mixed_precision and torch.cuda.is_available())

            # Компиляция модели
            compile_model(model)

            print(f"🔄 Продолжение обучения с эпохи {start_epoch}, LR={loaded['learning_rate']:.6f}")
        else:
            # Если не удалось загрузить, начинаем с нуля
            print("❌ Не удалось загрузить чекпоинт, начинаем обучение с нуля")

            # Оптимизация модели при первом запуске
            compile_model(model)

            # Сохраняем начальный сжатый чекпоинт
            save_compressed_checkpoint(
                model=model,
                epoch=-1,
                best_loss=float('inf'),
                lr=config.lr,
                path=config.checkpoint_path
            )
            print("[System]  Initial compressed checkpoint saved")
    else:
        # Оптимизация модели при первом запуске
        compile_model(model)

        # Сохраняем начальный сжатый чекпоинт
        save_compressed_checkpoint(
            model=model,
            epoch=-1,
            best_loss=float('inf'),
            lr=config.lr,
            path=config.checkpoint_path
        )
        print("[System]  Initial compressed checkpoint saved")

    summary(model, input_size=(1, 3, 224, 224))

    start_time = time.time()  # Засекаем время начала
    optimizer.zero_grad(set_to_none=True)  # Инициализация градиентов
    train_loader_len = len(train_loader)
    val_loader_len = len(val_loader)
    for epoch in range(start_epoch, config.epochs):
        model.train()
        train_loss = 0.0
        train_correct, train_total = 0, 0
        optimizer.zero_grad()

        epoch_start_time = time.time()  # Время начала эпохи
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            batch_start_time = time.time()  # Время начала обработки батча
            inputs, labels = inputs.to(device), labels.to(device)

            outputs, loss = forward_with_mixup_cutmix(model, inputs, labels, config, class_weights, device)
            # Накопление градиентов (основное изменение)
            scaler.scale(loss).backward()
            train_loss += loss.item()

            # Градиентный клиппинг
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Шаг оптимизатора
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # Расчет метрик только при обновлении
            _, predicted = torch.max(outputs, 1)
            current_batch_size = labels.size(0)
            train_total += current_batch_size
            train_correct += (predicted == labels).sum().item()

            # Расчет времени для логирования
            batch_duration = time.time() - batch_start_time
            remaining_batches = train_loader_len - (batch_idx + 1)
            estimated_remaining_time = remaining_batches * batch_duration * 3

            remaining_time_str = time.strftime('%H:%M:%S', time.gmtime(estimated_remaining_time))
            print(
                f"\r[Train] Epoch {epoch+1}/{config.epochs} | Batch {batch_idx+1}/{train_loader_len} | "
                f"Loss: {(loss.item()):.4f} | Remaining time: {remaining_time_str}",
                end='', flush=True)

        train_accuracy = 100 * train_correct / train_total
        print()

        # Валидация
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=config.mixed_precision):
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                batch_start_time = time.time()

                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                loss = focal_loss_with_smoothing(outputs, labels, config.focal_gamma, config.smoothing)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Расчет оставшегося времени
                batch_duration = time.time() - batch_start_time
                remaining_batches = val_loader_len - (batch_idx + 1)
                estimated_remaining_time = remaining_batches * batch_duration * 3
                remaining_time_str = time.strftime('%H:%M:%S', time.gmtime(estimated_remaining_time))

                print(
                    f"\r[Val]   Epoch {epoch+1}/{config.epochs} | Batch {batch_idx+1}/{val_loader_len} | "
                    f"Loss: {loss.item():.4f} | Remaining: {remaining_time_str}",
                    end='', flush=True)

        print()

        # Уменьшение скорости обучения
        current_lr = optimizer.param_groups[0]['lr']
        if val_loss > best_loss:
            next_lr = current_lr * config.factor_lr
            optimizer = optim.AdamW(model.parameters(), lr=next_lr)
        else:
            next_lr = current_lr
        
        # Расчет метрик
        val_accuracy = 100 * val_correct / val_total
        val_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        val_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        epoch_end_time = time.time()  # Время окончания эпохи
        epoch_duration = epoch_end_time - epoch_start_time
        total_elapsed_time = epoch_end_time - start_time
        epoch_duration_str = time.strftime("%H:%M:%S", time.gmtime(epoch_duration))
        total_elapsed_str = time.strftime("%H:%M:%S", time.gmtime(total_elapsed_time))

        # Логирование
        print(f"[Summary] Train Loss: {train_loss/len(train_loader):.4f} | Acc: {train_accuracy:.2f}%")
        print(f"[Summary] Val   Loss: {val_loss/len(val_loader):.4f} | Acc: {val_accuracy:.2f}%")
        print(f"[Summary] Val Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}")
        print(f"[Time]    Epoch: {epoch_duration_str} | Total: {total_elapsed_str}")
        print(f"[Summary] LR: {current_lr:.10f}")
        print(f"[Summary] Next LR: {next_lr:.10f}")

        # Ранняя остановка
        if val_loss < best_loss:
            best_loss = val_loss
            early_stop_counter = 0
            current_lr = optimizer.param_groups[0]['lr']
            save_compressed_checkpoint(
                model=model,
                epoch=epoch,
                best_loss=best_loss,
                lr=current_lr,
                path=config.checkpoint_path
            )
            print("[System]  Checkpoint saved (compressed)")
        else:
            early_stop_counter += 1
            if early_stop_counter >= config.early_stopping_patience:
                print("[System]  Early Stop")
                break
        print()

# ====================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ======================
def convert_to_onnx():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Загружаем классы
    with open(config.labels_path) as f:
        classes = [line.strip() for line in f]

    # Создаем модель
    model = AnimeClassifier(len(classes)).to(device)

    # Загружаем сжатый чекпоинт
    loaded = load_compressed_checkpoint(model, config.checkpoint_path, device)

    if loaded is None:
        print("❌ Не удалось загрузить чекпоинт для конвертации в ONNX!")
        return

    model = loaded['model']
    model.eval()

    # Удаляем префикс _orig_mod. если модель была скомпилирована
    model_state_dict = model.state_dict()
    if any('_orig_mod.' in key for key in model_state_dict.keys()):
        # Если модель была скомпилирована, нужно ее декомпилировать
        model = torch._dynamo.run(model)

    print(f"✅ Модель загружена для ONNX экспорта")
    print(f"   Классов: {len(classes)}")
    print(f"   Устройство: {device}")

    # Экспорт в ONNX
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
            opset_version=14,  # Используем более новую версию
            training=torch.onnx.TrainingMode.EVAL,
            verbose=False
        )
        print(f"✅ ONNX модель сохранена: {config.onnx_path}")

        # Проверяем размер файла
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

    # Загрузка ONNX-модели
    try:
        # Настраиваем сессию ONNX Runtime
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Используем доступные провайдеры
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']

        session = ort.InferenceSession(config.onnx_path, options, providers=providers)
        print("✅ ONNX Runtime сессия создана")
    except Exception as e:
        print(f"❌ Ошибка загрузки ONNX модели: {e}")
        return

    # Загружаем тестовое изображение
    test_image_path = os.path.join(config.dir, "test.jpg")
    if not os.path.exists(test_image_path):
        print(f"❌ Тестовое изображение не найдено: {test_image_path}")
        print("   Создайте файл test.jpg в папке проекта")
        return

    try:
        img = Image.open(test_image_path).convert('RGB')
        img_np = np.array(img)

        # Применение преобразований
        transform = ImageDataset._get_transforms('val')
        augmented = transform(image=img_np)
        img_tensor = augmented['image'].unsqueeze(0)

        print(f"✅ Изображение загружено: {img.size[0]}x{img.size[1]}")
    except Exception as e:
        print(f"❌ Ошибка загрузки изображения: {e}")
        return

    # ====================== PyTorch предсказание ======================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Загружаем классы
    with open(config.labels_path) as f:
        classes = [line.strip() for line in f]

    # Инициализация и загрузка PyTorch-модели
    model = AnimeClassifier(len(classes)).to(device)
    loaded = load_compressed_checkpoint(model, config.checkpoint_path, device)

    if loaded is None:
        print("❌ Не удалось загрузить PyTorch модель для сравнения!")
        return

    model = loaded['model']
    model.eval()

    # Предсказание PyTorch
    with torch.no_grad():
        pytorch_output = model(img_tensor.to(device))
        pytorch_probs = torch.softmax(pytorch_output, dim=1).cpu()

    # ====================== ONNX предсказание ======================
    try:
        # Подготавливаем входные данные для ONNX
        onnx_input = img_tensor.numpy().astype(np.float32)

        # Запускаем inference
        onnx_outputs = session.run(None, {'input': onnx_input})
        onnx_probs = torch.softmax(torch.tensor(onnx_outputs[0]), dim=1)

        print("✅ ONNX inference выполнен успешно")
    except Exception as e:
        print(f"❌ Ошибка ONNX inference: {e}")
        return

    # ====================== Вывод результатов ======================

    # Результаты PyTorch
    print("\n" + "="*50)
    print("[PyTorch] Топ-5 предсказаний:")
    pytorch_top_probs, pytorch_top_indices = torch.topk(pytorch_probs, 5)
    for i, (prob, idx) in enumerate(zip(pytorch_top_probs[0], pytorch_top_indices[0])):
        print(f"{i+1}. {classes[idx]}: {prob.item()*100:.2f}%")

    # Результаты ONNX
    print("\n[ONNX] Топ-5 предсказаний:")
    onnx_top_probs, onnx_top_indices = torch.topk(onnx_probs, 5)
    for i, (prob, idx) in enumerate(zip(onnx_top_probs[0], onnx_top_indices[0])):
        print(f"{i+1}. {classes[idx]}: {prob.item()*100:.2f}%")

    # Проверка совпадения результатов
    diff = torch.max(torch.abs(pytorch_probs - onnx_probs)).item()
    print(f"\n[Сравнение] Расхождение между выходами: {diff:.6f}")

    if diff < 0.001:
        print("✅ Конвертация успешна! Расхождение < 0.001")
    elif diff < 0.01:
        print("⚠️  Небольшое расхождение (0.001-0.01), возможно из-за численной точности")
    else:
        print("❌ Большое расхождение (> 0.01)! Возможная ошибка конвертации")

    # Дополнительная информация
    print("\n" + "="*50)
    print(f"PyTorch device: {device}")
    print(f"PyTorch dtype: {pytorch_probs.dtype}")
    print(f"ONNX dtype: {onnx_probs.dtype}")
    print(f"Количество классов: {len(classes)}")

def get_classes():
  with open(config.labels_path) as f:
      return [line.strip() for line in f]

# ====================== ИНТЕРФЕЙС ======================
def main_menu():
    while True:
        print("\nМеню:")
        print("1. Обучить модель")
        print("2. Продолжить обучение")  # Новая опция
        print("3. Конвертировать в ONNX")
        print("4. Протестировать ONNX")
        print("0. Выход")
        choice = input("Выбор: ").strip()

        if choice == '1':
            config.resume_training = False
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
        elif choice == '0':
            break
        else:
            print("Неверный ввод!")

if __name__ == "__main__":
    main_menu()