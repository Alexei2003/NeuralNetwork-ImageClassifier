import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import os
from glob import glob
from PIL import Image
import onnxruntime as ort
from sklearn.metrics import precision_score, recall_score, f1_score
import time
from torchsummary import summary

# ====================== КОНФИГУРАЦИЯ ======================
class Config:
    source_dir = "/media/alex/Programs/NeuralNetwork/DataSet/ARTS/Original"
    checkpoint_path = "/media/alex/Programs/NeuralNetwork/Model/best_model.pth"
    labels_path = "/media/alex/Programs/NeuralNetwork/Model/labels.txt"
    onnx_path = "/media/alex/Programs/NeuralNetwork/Model/model.onnx"
    input_size = (224, 224)
    num_experts = 16
    expert_units = 1024
    k_top_expert = 2
    se_reduction = 16
    lr = 1e-3
    factor_lr = 0.5
    patience_lr =2
    batch_size = 64
    epochs = 30
    momentum = 0.95
    focal_gamma = 5
    dropout = 0.5
    mixed_precision = True
    early_stopping_patience = 10
    val_split = 0.2

config = Config()

# ====================== КОМПОНЕНТЫ МОДЕЛИ ======================
class MoE(nn.Module):
    def __init__(self, input_dim, num_experts, expert_units, k_top):
        super().__init__()
        self.num_experts = num_experts
        self.k_top = k_top
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_units),
                nn.SiLU(inplace=True) ,
                nn.Dropout(config.dropout),
                nn.Linear(expert_units, input_dim))
            for _ in range(num_experts)
        ])
        self.router = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        logits = self.router(x)
        top_k_weights, top_k_indices = logits.topk(self.k_top, dim=1)
        top_k_weights = torch.softmax(top_k_weights, dim=1)
        
        output = torch.zeros_like(x)
        for i in range(self.k_top):
            expert_idx = top_k_indices[:, i]
            expert_outputs = torch.stack([
                self.experts[idx](x[batch_idx]) 
                for batch_idx, idx in enumerate(expert_idx)
            ])
            output += expert_outputs * top_k_weights[:, i].unsqueeze(1)
        return output + x

class SEBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        reduced = max(1, channels // config.se_reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, 1),
            nn.SiLU(inplace=True) ,
            nn.Conv2d(reduced, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.act = nn.SiLU(inplace=True) 
        self.dropout = nn.Dropout2d(config.dropout)  # Добавлен Dropout

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):      
        residual = self.shortcut(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.dropout(x)  # Добавлен Dropout
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        return self.act(x + residual)

class AnimeClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True) ,
            nn.MaxPool2d(3, stride=2, padding=1),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 512, stride=2),
            nn.AdaptiveAvgPool2d(1)
        )
        self.moe = MoE(512, config.num_experts, config.expert_units, config.k_top_expert)
        self.head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x).flatten(1)
        x = self.moe(x)
        return self.head(x)

# ====================== ОБРАБОТКА ДАННЫХ ======================
class ImageDataset(Dataset):
    def __init__(self, root, transform=None, mode='train'):
        self.classes = sorted(os.listdir(root))
        self.samples = []
        for label, cls in enumerate(self.classes):
            self.samples.extend([(f, label) for f in glob(os.path.join(root, cls, '*'))])
        self.transform = transform or self._get_transforms(mode)

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        return self.transform(img), label

    @staticmethod
    def _get_transforms(mode):
        if mode == 'train':
            return transforms.Compose([
                transforms.RandomRotation(30),  # Поворот
                transforms.RandomResizedCrop(config.input_size, scale=(0.8, 1.0)),  # Случайный зум
                transforms.RandomHorizontalFlip(),  # Отражение

                transforms.ColorJitter(  # Яркость, контраст, насыщенность
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2
                ),

                transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
                transforms.RandomSolarize(threshold=0.5, p=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),

                transforms.ToTensor(),
            ])
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(config.input_size),
            transforms.ToTensor(),
        ])

# ====================== ОБУЧЕНИЕ ======================
def focal_loss(outputs, targets, gamma=5):
    ce_loss = nn.CrossEntropyLoss(reduction='none')(outputs, targets)
    pt = torch.exp(-ce_loss)
    return ((1 - pt)**gamma * ce_loss).mean()

def run_training():
    # Включение оптимизации cuDNN
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)

    full_dataset = ImageDataset(config.source_dir)
    train_size = int((1 - config.val_split) * len(full_dataset))
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, len(full_dataset) - train_size])

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=os.cpu_count(), persistent_workers=True, 
                              prefetch_factor=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, num_workers=os.cpu_count(), persistent_workers=True, 
                            prefetch_factor=2, pin_memory=True)

    with open(config.labels_path, 'w') as f:
        f.write('\n'.join(full_dataset.classes))

    model = AnimeClassifier(len(full_dataset.classes)).to(device)
    # Оптимизация модели
    model = torch.compile(
        model,
        mode="default",       # Режим оптимизации
        dynamic=False,        # Динамические формы тензоров (PyTorch 2.1+)
        fullgraph=False       # Требовать полной компиляции всего графа (если возможно)
    )
    torch.save(model.state_dict(), config.checkpoint_path)
    summary(model, (3, 224, 224)) 
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.factor_lr, patience=config.patience_lr)
    scaler = torch.amp.GradScaler('cuda', enabled=config.mixed_precision and torch.cuda.is_available())

    best_loss = float('inf')
    early_stop_counter = 0
    best_epoch = 0

    start_time = time.time()  # Засекаем время начала

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        train_correct, train_total = 0, 0

        epoch_start_time = time.time()  # Время начала эпохи
        print(f"\n--- Epoch {epoch + 1}/{config.epochs} ---")
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            batch_start_time = time.time()  # Время начала обработки батча
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=config.mixed_precision):
                outputs = model(inputs)
                loss = focal_loss(outputs, labels, config.focal_gamma)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient Clipping
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            batch_end_time = time.time()  # Время окончания обработки батча
            batch_duration = batch_end_time - batch_start_time
            remaining_batches = len(train_loader) - (batch_idx + 1)
            estimated_remaining_time = remaining_batches * batch_duration
            remaining_time_str = time.strftime('%H:%M:%S', time.gmtime(estimated_remaining_time))

            print(
                f"\r[Train] Epoch {epoch+1}/{config.epochs} | Batch {batch_idx+1}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f} | Remaining time: {remaining_time_str}",
                end='', flush=True)

        epoch_end_time = time.time()  # Время окончания эпохи
        epoch_duration = epoch_end_time - epoch_start_time
        total_elapsed_time = epoch_end_time - start_time
        epoch_duration_str = time.strftime("%H:%M:%S", time.gmtime(epoch_duration))
        total_elapsed_str = time.strftime("%H:%M:%S", time.gmtime(total_elapsed_time))

        print()
        train_accuracy = 100 * train_correct / train_total

        # Валидация
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad(), torch.amp.autocast('cuda', enabled=config.mixed_precision):
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                val_loss += focal_loss(outputs, labels, config.focal_gamma).item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Уменьшение скорости обучения      
        cosine_scheduler.step()
        plateau_scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Расчет метрик
        val_accuracy = 100 * val_correct / val_total
        val_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        val_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        # Логирование
        print(f"[Summary] Train Loss: {train_loss/len(train_loader):.4f} | Acc: {train_accuracy:.2f}%")
        print(f"[Summary] Val   Loss: {val_loss/len(val_loader):.4f} | Acc: {val_accuracy:.2f}%")
        print(f"[Summary] Val Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}")
        print(f"[Time] Epoch: {epoch_duration_str} | Total: {total_elapsed_str}")
        print(f"[Summary] LR: {current_lr:.6f}")
        print()

        # Ранняя остановка
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            early_stop_counter = 0
            torch.save(model.state_dict(), config.checkpoint_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= config.early_stopping_patience:
                print("Ранняя остановка!")
                break

    print(f"\n🏆 Лучшая эпоха: {best_epoch} с валидационным лоссом: {best_loss:.4f}")

# ====================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ======================
def convert_to_onnx():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AnimeClassifier(len(get_classes())).to(device)
    checkpoint = torch.load(config.checkpoint_path)
    # Исправление ключей
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    dummy_input = torch.randn(1, 3, *config.input_size).to(device)
    torch.onnx.export(
        model, 
        dummy_input, 
        config.onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=13,
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL, 
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK 
    )
    print("ONNX модель сохранена:", config.onnx_path)

def test_onnx():
    if not os.path.exists(config.onnx_path):
        print("ONNX модель не найдена!")
        return
    
    session = ort.InferenceSession(config.onnx_path)
    transform = ImageDataset._get_transforms('val')
    
    try:
        img = Image.open("test.jpg").convert('RGB')
        img_tensor = transform(img).unsqueeze(0).numpy()
    except FileNotFoundError:
        print("Файл test.jpg не найден!")
        return
    
    outputs = session.run(None, {'input': img_tensor.astype(np.float32)})
    probs = torch.softmax(torch.tensor(outputs[0]), dim=1)
    
    with open(config.labels_path) as f:
        classes = [line.strip() for line in f]
    
    print("\nТоп-5 предсказаний:")
    top_probs, top_indices = torch.topk(probs, 5)
    for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0])):
        print(f"{i+1}. {classes[idx]}: {prob*100:.2f}%")

def get_classes():
    with open(config.labels_path) as f:
        return [line.strip() for line in f]

# ====================== ИНТЕРФЕЙС ======================
def main_menu():
    while True:
        print("\nМеню:")
        print("1. Обучить модель")
        print("2. Конвертировать в ONNX")
        print("3. Протестировать ONNX")
        print("0. Выход")
        choice = input("Выбор: ").strip()
        
        if choice == '1':
            run_training()
        elif choice == '2':
            convert_to_onnx()
        elif choice == '3':
            test_onnx()
        elif choice == '0':
            break
        else:
            print("Неверный ввод!")

if __name__ == "__main__":
    main_menu()