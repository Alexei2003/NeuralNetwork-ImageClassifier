"""
Нейросетевая модель классификации изображений с MoE (Mixture of Experts)
и автоматическим определением количества классов
"""

# ====================== ИМПОРТ БИБЛИОТЕК ======================
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dense, Dropout, 
                                   BatchNormalization, Activation, GlobalAveragePooling2D,
                                   Add, Reshape, Multiply, Layer, LayerNormalization,
                                   RandomRotation, RandomZoom, RandomContrast, RandomBrightness)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras import backend as K
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# ========================== КОНФИГУРАЦИЯ ==========================
class Config:
    # -------------------- Архитектура модели --------------------
    input_shape = (224, 224, 3)    # Размер входных изображений (H, W, C)
    l1_value = 1e-5                # Коэффициент L1-регуляризации
    l2_value = 1e-4                # Коэффициент L2-регуляризации
    dropout_rate = 0.5             # Процент дропаута
    num_experts = 8                # Количество экспертов в слое MoE
    expert_units = 1024            # Нейронов в каждом эксперте
    se_reduction = 16              # Коэффициент уменьшения в SE-блоке

    # --------------------- Параметры обучения ---------------------
    initial_learning_rate = 1e-1   # Начальная скорость обучения
    batch_size = 64                # Размер батча
    epochs = 1000                  # Максимальное число эпох
    min_learning_rate = 1e-10      # Минимальная скорость обучения
    reduce_lr_factor = 0.25        # Фактор уменьшения LR
    reduce_lr_patience = 2         # Терпение для уменьшения LR
    early_stopping_patience = 10   # Терпение для ранней остановки
    focal_alpha = 0.25             # Параметр Focal Loss (баланс классов)
    focal_gamma = 2.0              # Параметр Focal Loss (фокусировка)

    # --------------------- Аугментация данных ---------------------
    rotation_range = 0.2           # Максимальный угол поворота (доля от 180°)
    zoom_range = 0.3               # Максимальное увеличение/уменьшение
    contrast_range = 0.2           # Диапазон изменения контраста
    brightness_range = 0.3         # Диапазон изменения яркости
    horizontal_flip = True         # Горизонтальное отражение
    vertical_flip = False          # Вертикальное отражение
    validation_split = 0.2         # Доля данных для валидации
    augment_seed = 123             # Сид для воспроизводимости аугментаций

    # --------------------- Пути сохранения ---------------------
    source_dir = "/media/alex/Programs/NeuralNetwork/DataSet/ARTS/Original"
    checkpoint_path = "/media/alex/Programs/NeuralNetwork/Model/best_model.keras"
    labels_path = "/media/alex/Programs/NeuralNetwork/Model/labels.txt"
    onnx_path = "/media/alex/Programs/NeuralNetwork/Model/model.onnx"

# Инициализация конфигурации
config = Config()
set_global_policy('mixed_float16')  # Включение mixed precision

# ====================== КАСТОМНЫЕ КОМПОНЕНТЫ ======================
class MoE(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = config.num_experts
        self.expert_units = config.expert_units

    def build(self, input_shape):
        self.experts = [self._build_expert(input_shape[-1]) for _ in range(self.num_experts)]
        self.router = Dense(
            self.num_experts,
            activation='softmax',
            kernel_regularizer=l1_l2(config.l1_value, config.l2_value)
        )
        super().build(input_shape)

    def _build_expert(self, input_dim):
        return tf.keras.Sequential([
            Dense(self.expert_units, 
                  activation='swish',
                  kernel_regularizer=l1_l2(config.l1_value, config.l2_value)),
            Dropout(config.dropout_rate),
            Dense(input_dim,
                  kernel_regularizer=l1_l2(config.l1_value, config.l2_value))
        ])

    def call(self, inputs):
        weights = self.router(inputs)
        expert_outputs = tf.stack([expert(inputs) for expert in self.experts], axis=1)
        weighted_outputs = tf.einsum('be,beu->bu', weights, expert_outputs)
        return weighted_outputs + inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_experts': self.num_experts,
            'expert_units': self.expert_units
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def focal_loss(y_true, y_pred):
    """Focal Loss для работы с несбалансированными классами"""
    y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
    cross_entropy = -y_true * K.log(y_pred)
    loss = config.focal_alpha * K.pow(1. - y_pred, config.focal_gamma) * cross_entropy
    return K.sum(loss, axis=1)

def se_block(input_tensor):
    """Squeeze-and-Excitation блок для перевзвешивания каналов"""
    channels = input_tensor.shape[-1]
    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape((1, 1, channels))(se)
    se = Dense(channels//config.se_reduction, activation='swish',
               kernel_regularizer=l1_l2(config.l1_value, config.l2_value))(se)
    se = Dense(channels, activation='sigmoid',
               kernel_regularizer=l1_l2(config.l1_value, config.l2_value))(se)
    return Multiply()([input_tensor, se])

def residual_block(x, filters, stride=1):
    """Остаточный блок с SE-модулем"""
    shortcut = x
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1,1), strides=stride,
                          kernel_regularizer=l1_l2(config.l1_value, config.l2_value))(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    x = Conv2D(filters, (3,3), strides=stride, padding='same',
               kernel_regularizer=l1_l2(config.l1_value, config.l2_value))(x)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)
    x = Conv2D(filters, (3,3), padding='same',
               kernel_regularizer=l1_l2(config.l1_value, config.l2_value))(x)
    x = BatchNormalization()(x)
    x = se_block(x)
    return Activation('swish')(Add()([x, shortcut]))

# ====================== ПОСТРОЕНИЕ МОДЕЛИ ======================
def build_model(num_classes):
    """Сборка полной архитектуры модели"""
    inputs = Input(shape=config.input_shape)
    
    # Бэкбон CNN
    x = Conv2D(64, (7,7), strides=2, padding='same',
               kernel_regularizer=l1_l2(config.l1_value, config.l2_value))(inputs)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)
    x = MaxPooling2D((3,3), strides=2, padding='same')(x)
    
    # Residual Blocks
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 512, stride=2)
    
    # Головная часть
    x = GlobalAveragePooling2D()(x)
    x = LayerNormalization()(x)
    x = Dense(1024, activation='swish',
              kernel_regularizer=l1_l2(config.l1_value, config.l2_value))(x)
    x = Dropout(config.dropout_rate)(x)
    x = MoE()(x)
    
    outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    # Компиляция модели
    model = Model(inputs, outputs, name='AnimeClassifier')
    optimizer = SGD(learning_rate=config.initial_learning_rate,
                    momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer,
                  loss=focal_loss,
                  metrics=['accuracy', 'top_k_categorical_accuracy'])
    return model

# ====================== ОБРАБОТКА ДАННЫХ ======================
def create_dataset(subset):
    """Создает tf.data.Dataset из директории с изображениями"""
    return tf.keras.utils.image_dataset_from_directory(
        config.source_dir,
        labels='inferred',
        label_mode='categorical',
        color_mode='rgb',
        batch_size=config.batch_size,
        image_size=config.input_shape[:2],
        validation_split=config.validation_split,
        subset=subset,
        seed=config.augment_seed,
        shuffle=(subset == 'training')
    )

class EpochSpacingCallback(Callback):
    """Визуальное разделение логов обучения"""
    def on_epoch_end(self, epoch, logs=None):
        print('\n' + '=' * 100 + '\n')

# ====================== ОБУЧЕНИЕ МОДЕЛИ ======================
def run_training():
    """Запуск процесса обучения с аугментацией"""
    os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
    
    # Загрузка данных
    train_ds = create_dataset('training')
    val_ds = create_dataset('validation')
    num_classes = len(train_ds.class_names)

    # Аугментация
    augmentations = tf.keras.Sequential([
        RandomRotation(config.rotation_range),
        RandomZoom(config.zoom_range),
        RandomContrast(config.contrast_range),
        RandomBrightness(config.brightness_range),
        tf.keras.layers.RandomFlip(
            mode='horizontal_and_vertical' if config.horizontal_flip and config.vertical_flip else
            'horizontal' if config.horizontal_flip else
            'vertical' if config.vertical_flip else None
        )
    ])
    train_ds = train_ds.map(
        lambda x, y: (augmentations(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    # Веса классов
    labels = np.concatenate([y.numpy().argmax(axis=1) for x, y in train_ds], axis=0)
    class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    class_weights_dict = {i: w for i, w in enumerate(class_weights)}

    # Инициализация модели
    if os.path.exists(config.checkpoint_path):
        print("Загрузка модели...")
        model = load_model(
            config.checkpoint_path,
            custom_objects={
                'MoE': MoE,
                'focal_loss': focal_loss,
                'LayerNormalization': LayerNormalization
            }
        )
        if model.output_shape[-1] != num_classes:
            raise ValueError(f"Модель обучена на {model.output_shape[-1]} классов, данные содержат {num_classes}")
    else:
        print("Создание новой модели...")
        model = build_model(num_classes)
        model.summary()

    # Колбэки
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=config.reduce_lr_factor,
                         patience=config.reduce_lr_patience, min_lr=config.min_learning_rate),
        ModelCheckpoint(config.checkpoint_path, save_best_only=True, monitor='val_loss'),
        EarlyStopping(monitor='val_loss', patience=config.early_stopping_patience,
                     restore_best_weights=True),
        EpochSpacingCallback()
    ]

    # Обучение
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=callbacks,
        class_weight=class_weights_dict
    )
    return model

# ====================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ======================
def save_labels():
    """Сохранение меток классов"""
    class_names = sorted(os.listdir(config.source_dir))
    with open(config.labels_path, "w") as f:
        for label in class_names: f.write(label + "\n")
    print(f"Метки сохранены в {config.labels_path}")

def convert_to_onnx():
    """Конвертация в ONNX формат"""
    import tf2onnx
    import onnxruntime as ort
    
    model = load_model(
        config.checkpoint_path,
        custom_objects={
            'MoE': MoE,
            'focal_loss': focal_loss,
            'LayerNormalization': LayerNormalization
        }
    )
    input_signature = [tf.TensorSpec(shape=[None, *config.input_shape], dtype=tf.float32)]
    tf2onnx.convert.from_keras(model, input_signature=input_signature, output_path=config.onnx_path)
    
    # Проверка
    session = ort.InferenceSession(config.onnx_path)
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, *config.input_shape).astype(np.float32)
    session.run(None, {input_name: dummy_input})
    print("ONNX конвертация успешна!")
    save_labels()

# ====================== ИНТЕРФЕЙС ПОЛЬЗОВАТЕЛЯ ======================
def main():
    """Главное меню программы"""
    while True:
        print("\nМеню:")
        print("1. Обучить модель")
        print("2. Конвертировать в ONNX")
        print("3. Выход")
        choice = input("Выберите действие: ").strip()
        
        if choice == '1':
            if not os.path.exists(config.source_dir):
                print("Ошибка: Директория с данными не найдена!")
                continue
            run_training()
            print("Обучение завершено!")
        elif choice == '2':
            if not os.path.exists(config.checkpoint_path):
                print("Ошибка: Модель не найдена!")
                continue
            convert_to_onnx()
        elif choice == '3':
            print("Выход...")
            break
        else:
            print("Неверный ввод!")

if __name__ == "__main__":
    main()