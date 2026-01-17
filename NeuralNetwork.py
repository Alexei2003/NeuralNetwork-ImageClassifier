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
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.backends.cudnn
import gzip

# ====================== КОНФИГУРАЦИЯ ======================
class Config:
    # выбор системы
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
    num_experts = 64                # Количество экспертов в MoE (Mixture of Experts)
    expert_units = 1024             # Количество нейронов в каждом эксперте
    k_top_expert = 4                # Количество активных экспертов на один пример
    se_reduction = 16               # Коэффициент редукции для SE (Squeeze-and-Excitation) блока
    dropout = 0.5                   # Вероятность отключения нейронов (dropout)

    # Параметры обучения
    val_split = 0.2                 # Доля данных, выделяемая под валидацию
    gradient_clip = 1.0             # Максимальная норма градиента
    batch_size = 256                # Размер батча (число примеров, обрабатываемых за один проход)
    epochs = 100                    # Количество эпох обучения (полных проходов по всему датасету)
    focal_gamma = 5                 # Параметр гамма для Focal Loss, регулирует степень фокусировки на сложных примерах
    smoothing = 0.1                 # Параметр label smoothing, задаёт уровень сглаживания меток для улучшения обобщения
    mixed_precision = True          # Использовать смешанную точность (fp16) для ускорения обучения

    # Параметры LR
    max_lr = 0.005                  # Максимальная скорость обучения (learning rate)
    ini_lr = 0.001                  # Начальная скорость обучения
    plateau_factor = 0.9            # Уменьшать lr
    plateau_threshold = 0.01        # Порог улучшения (относительный)
    early_stopping_patience = 5     # Количество эпох без улучшения для ранней остановки

config = Config()

# ====================== КОСИНУСНЫЙ ШЕДУЛЕР С WARMUP ======================
class WarmupReduceLROnPlateau():
    """Warmup + ReduceLROnPlateau логика"""

    def __init__(self, optimizer, ini_lr, max_lr, factor, threshold):
        self.optimizer = optimizer

        self.max_lr = max_lr
        self.ini_lr = ini_lr
        self.current_epoch = 0

        # ReduceLROnPlateau параметры
        self.factor = factor
        self.threshold = threshold
        self.num_reduced = 0

        # Трекинг лучшего loss
        self.best_loss = float('inf')

    def step(self, epoch=None, validation_loss=None):
        """Вызывается в конце каждой эпохи с validation_loss"""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        if self.current_epoch == 1:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.ini_lr
            self.best_loss = validation_loss
            return self.optimizer.param_groups[0]['lr']

        # Warmup фаза
        if self.current_epoch == 2:
            factor = self.max_lr / self.ini_lr
            self._change_lr(factor)
            self.best_loss = validation_loss
            return self.optimizer.param_groups[0]['lr']

        if self._is_better(validation_loss, self.best_loss):
            self.best_loss = validation_loss
            print(f"✓ Улучшение!")
        else:
            self._reduce_lr()
            print(f"📉 Уменьшение LR! Новый LR: {self.optimizer.param_groups[0]['lr']:.6f}")

        return self.optimizer.param_groups[0]['lr']

    def _is_better(self, current, best):
        """Проверка, лучше ли текущий loss"""
        return current < best - best * self.threshold

    def _reduce_lr(self):
        """Уменьшение LR для всех групп параметров"""
        self.num_reduced += 1
        factor = self.factor**self.num_reduced
        print(f"[LR]    Factor:    {factor:.8f}")
        self._change_lr(factor)

    def _change_lr(self, factor):
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = old_lr * factor
            param_group['lr'] = new_lr

    def get_last_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self):
        return {
            'current_epoch': self.current_epoch,
            'ini_lr': self.ini_lr,
            'max_lr': self.max_lr,
            'best_loss': self.best_loss,
            'factor': self.factor,
            'threshold': self.threshold,
        }

    def load_state_dict(self, state_dict):
        for key, value in state_dict.items():
            setattr(self, key, value)

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
                    ratio=(0.75, 1.33),
                    interpolation=1,
                    p=1.0
                ),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,  # Добавили hue для лучшей аугментации
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
            A.Resize(config.input_size[0], config.input_size[1]),  # Добавили Resize для валидации
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
    weights = weights / weights.max()  # нормализация
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

def forward_with_mixup_cutmix(model, inputs, labels, config, class_weights, device):
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
        fullgraph=True)
    torch.cuda.empty_cache()

def save_compressed_checkpoint(model, epoch, optimizer, scheduler, path):
    """
    Умное сжатие с учетом mixed precision
    """
    # 1. Определяем, какие веса можно сжимать
    checkpoint = {
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }

    # 2. Сжимаем ВСЕ веса в float16
    compressed_weights = {}
    for name, param in model.state_dict().items():
        if param.is_floating_point():
            compressed_weights[name] = param.half().clone()
        else:
            compressed_weights[name] = param

    checkpoint['model_state_dict'] = compressed_weights

    # 3. Сохраняем с максимальным сжатием
    with gzip.open(path, 'wb', compresslevel=9) as f:
        torch.save(checkpoint, f, pickle_protocol=4)

    # 4. Показываем результат сжатия
    size_mb = os.path.getsize(path) / 1024 / 1024
    temp_path = path + '.tmp'
    torch.save(checkpoint, temp_path)
    uncompressed_size = os.path.getsize(temp_path) / 1024 / 1024
    os.remove(temp_path)

    compression_ratio = (1 - size_mb / uncompressed_size) * 100

    print(f"[System]  Чекпоинт сохранен: {size_mb:.1f} MB")
    print(f"[System]  Сжатие: {compression_ratio:.0f}% от {uncompressed_size:.1f} MB")

    return size_mb

def load_compressed_checkpoint(model, optimizer, scheduler, path, device):
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
                    loaded_weights[name] = saved_param.to(current_param.dtype)
                else:
                    loaded_weights[name] = saved_param
            else:
                loaded_weights[name] = current_weights[name]
                print(f"⚠️  Пропущен параметр: {name}")

        # 4. Загружаем в модель
        model.load_state_dict(loaded_weights)
        model.to(device)

        # 5. Восстанавливаем optimizer и scheduler
        if optimizer and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            if hasattr(scheduler, 'load_state_dict'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
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
    print(f"  • Ini LR: {config.ini_lr:.4f}")
    print(f"  • Max LR: {config.max_lr:.4f}")
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
        num_workers=os.cpu_count()-1,  
        persistent_workers=True,
        prefetch_factor=1,
        pin_memory=True)
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        num_workers=os.cpu_count()-1,
        persistent_workers=True,
        prefetch_factor=1,
        pin_memory=True)

    model = AnimeClassifier(len(full_classes)).to(device)

    class_weights = get_class_weights_from_dirs(config.source_dir, full_classes).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config.ini_lr)

    scheduler = WarmupReduceLROnPlateau(
        optimizer=optimizer,
        ini_lr=config.ini_lr,
        max_lr=config.max_lr,
        factor=config.plateau_factor,
        threshold=config.plateau_threshold,
    )

    scaler = torch.amp.GradScaler('cuda', enabled=config.mixed_precision and torch.cuda.is_available())
    start_epoch = 1
    early_stop_counter = 0

    if config.resume_training:
        loaded = load_compressed_checkpoint(model, optimizer, scheduler, config.checkpoint_path, device)

        if loaded is not None:
            model = loaded['model']
            optimizer = loaded['optimizer']
            scheduler = loaded['scheduler']
            start_epoch = loaded['epoch'] + 1

            # Компиляция модели
            compile_model(model)

            print(f"🔄 Продолжение обучения с эпохи {start_epoch}")
            print(f"  Текущий LR: {optimizer.param_groups[0]['lr']:.6f}")
        else:
            print("❌ Не удалось загрузить чекпоинт, начинаем обучение с нуля")
            compile_model(model)
            save_compressed_checkpoint(
                model=model,
                epoch=-1,
                optimizer=optimizer,
                scheduler=scheduler,
                path=config.checkpoint_path
            )
            print("[System]  Initial compressed checkpoint saved")
    else:
        compile_model(model)
        save_compressed_checkpoint(
            model=model,
            epoch=-1,
            optimizer=optimizer,
            scheduler=scheduler,
            path=config.checkpoint_path
        )
        print("[System]  Initial compressed checkpoint saved")

    start_time = time.time()
    train_loader_len = len(train_loader)
    val_loader_len = len(val_loader)

    for epoch in range(start_epoch, config.epochs+1):
        model.train()
        train_loss = 0.0
        train_correct, train_total = 0, 0
        optimizer.zero_grad(set_to_none=True)
        epoch_start_time = time.time()

        current_lr = scheduler.get_last_lr()
        print(f"[LR] Current: {current_lr:.8f}")

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            batch_start_time = time.time()
            inputs, labels = inputs.to(device), labels.to(device)

            outputs, loss = forward_with_mixup_cutmix(model, inputs, labels, config, class_weights, device)

            scaler.scale(loss).backward()
            train_loss += loss.item()

            # Градиентный клиппинг
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            _, predicted = torch.max(outputs, 1)
            current_batch_size = labels.size(0)
            train_total += current_batch_size
            train_correct += (predicted == labels).sum().item()

            batch_duration = time.time() - batch_start_time
            remaining_batches = train_loader_len - (batch_idx + 1)
            estimated_remaining_time = remaining_batches * batch_duration * 3

            remaining_time_str = time.strftime('%H:%M:%S', time.gmtime(estimated_remaining_time))
            print(
                f"\r[Train] Epoch {epoch}/{config.epochs} | Batch {batch_idx+1}/{train_loader_len} | "
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

                batch_duration = time.time() - batch_start_time
                remaining_batches = val_loader_len - (batch_idx + 1)
                estimated_remaining_time = remaining_batches * batch_duration * 3
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

        # Ранняя остановка и сохранение чекпоинта
        if val_loss < scheduler.best_loss:
            early_stop_counter = 0
            save_compressed_checkpoint(
                model=model,
                epoch=epoch,
                optimizer=optimizer,
                scheduler=scheduler,
                path=config.checkpoint_path
            )
            print("[System]  Checkpoint saved (compressed)")
        else:
            early_stop_counter += 1
            if early_stop_counter >= config.early_stopping_patience:
                print(f"[System]  Early Stop (no improvement for {early_stop_counter} epochs)")
                break

        next_lr = scheduler.step(epoch+1, val_loss)

        # Логирование
        print(f"[Summary] Train Loss: {train_loss/len(train_loader):.4f} | Acc: {train_accuracy:.2f}%")
        print(f"[Summary] Val   Loss: {val_loss/len(val_loader):.4f} | Acc: {val_accuracy:.2f}%")
        print(f"[Summary] Val Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}")
        print(f"[Time]    Epoch: {epoch_duration_str} | Total: {total_elapsed_str}")
        print(f"[LR]      Current: {current_lr:.8f}")
        print(f"[LR]      Next:    {next_lr:.8f}")

        print()

# ====================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ======================
def convert_to_onnx():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(config.labels_path) as f:
        classes = [line.strip() for line in f]

    model = AnimeClassifier(len(classes)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.ini_lr)
    scheduler = WarmupReduceLROnPlateau(
        optimizer=optimizer,
        ini_lr=config.ini_lr,
        max_lr=config.max_lr,
        factor=config.plateau_factor,
        threshold=config.plateau_threshold,
    )

    loaded = load_compressed_checkpoint(model, optimizer, scheduler, config.checkpoint_path, device)

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

    test_image_path = os.path.join(config.dir, "test.jpg")
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
    optimizer = optim.AdamW(model.parameters(), lr=config.ini_lr)
    scheduler = WarmupReduceLROnPlateau(
        optimizer=optimizer,
        ini_lr=config.ini_lr,
        max_lr=config.max_lr,
        factor=config.plateau_factor,
        threshold=config.plateau_threshold,
    )

    loaded = load_compressed_checkpoint(model, optimizer, scheduler, config.checkpoint_path, device)

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

# ====================== ИНТЕРФЕЙС ======================
def main_menu():
    while True:
        print("\n" + "="*50)
        print("🚀 Anime Classifier")
        print("="*50)
        print("1. Обучить модель (с нуля)")
        print("2. Продолжить обучение")
        print("3. Конвертировать в ONNX")
        print("4. Протестировать ONNX")
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