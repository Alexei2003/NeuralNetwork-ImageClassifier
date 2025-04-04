"""
Нейросетевая модель классификации изображений с MoE (Mixture of Experts)
и автоматическим определением количества классов (оптимизированная версия)
"""

# ====================== ИМПОРТ БИБЛИОТЕК ======================
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dense, Dropout, 
                                   BatchNormalization, Activation, GlobalAveragePooling2D,
                                   Add, Reshape, Multiply, Layer, LayerNormalization,
                                   RandomRotation, RandomZoom, RandomContrast, RandomBrightness,
                                   RandomFlip, RandomCrop, RandomSaturation)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras import backend as K
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import tf2onnx
import onnxruntime as ort
import math

# ========================== КОНФИГУРАЦИЯ ==========================
class Config:
    # -------------------- Архитектура модели --------------------
    input_shape = (224, 224, 3)    # Размер входных изображений (H, W, C)
    l1_value = 1e-6                # Коэффициент L1-регуляризации
    l2_value = 1e-5                # Коэффициент L2-регуляризации
    dropout_rate = 0.5             # Процент дропаута
    num_experts = 8                # Количество экспертов в слое MoE
    expert_units = 1024            # Нейронов в каждом эксперте
    se_reduction = 16              # Коэффициент уменьшения в SE-блоке

    # --------------------- Параметры обучения ---------------------
    initial_learning_rate = 1e-2   # Начальная скорость обучения
    batch_size = 32                # Размер батча
    epochs = 1500                  # Максимальное число эпох
    min_learning_rate = 1e-10      # Минимальная скорость обучения
    reduce_lr_factor = 0.1         # Фактор уменьшения LR
    reduce_lr_patience = 2         # Терпение для уменьшения LR
    early_stopping_patience = 10   # Терпение для ранней остановки
    focal_gamma = 4                # Параметр Focal Loss (фокусировка)
    class_weight_gamma = 2         # Усиление влияние весов класса

    # --------------------- Аугментация данных ---------------------
    rotation_range = 0.4           # Максимальный угол поворота (доля от 180°)
    zoom_range = 0.4               # Максимальное увеличение/уменьшение
    contrast_range = 0.4           # Диапазон изменения контраста
    saturation_range = 0.4         # Диапазон изменения насыщености
    brightness_range = 0.4         # Диапазон изменения яркости
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
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
tf.config.optimizer.set_experimental_options({
    "layout_optimizer": True,
    "constant_folding": True,
    "shape_optimization": True,
    "remapping": True,
    "arithmetic_optimization": True,
    "dependency_optimization": True,
    "loop_optimization": True,
    "function_optimization": True,
    "debug_stripper": True,
    "disable_meta_optimizer": False,
    "scoped_allocator_optimization": True,
    "pin_to_host_optimization": True,
    "auto_parallel" : True
})
set_global_policy('mixed_float16')  # Активация mixed precision
tf.config.optimizer.set_jit(True)

# ====================== КАСТОМНЫЕ КОМПОНЕНТЫ ======================
class MoE(Layer):
    def __init__(self, num_experts=8, expert_units=4096, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.expert_units = expert_units

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

    def call(self, inputs, training=None):
        # Получаем логиты маршрутизации
        logits = self.router(inputs)
        
        # Веса экспертов через softmax (тип: float16)
        weights = K.softmax(logits)
        
        # Маска для экспертов (ВАЖНО: приводим к типу weights!)
        expert_mask = K.cast(
            weights > 0.1,        # Булев тензор
            dtype=weights.dtype   # Явно указываем тип как у weights (float16)
        )
        
        # Выходы экспертов (предполагаем, что эксперты возвращают float16)
        expert_outputs = tf.stack([expert(inputs) for expert in self.experts], axis=1)
        
        # Взвешенная сумма (типы weights и expert_mask теперь совпадают)
        weighted_outputs = tf.einsum('be,beu->bu', weights * expert_mask, expert_outputs)
        
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
    y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
    
    # 1. Кросс-энтропия для истинного класса
    ce = -y_true * K.log(y_pred)  # [batch, num_classes]
    ce = K.sum(ce, axis=-1)       # [batch,] (сумма только по активному классу)
    
    # 2. Вероятность истинного класса
    p_t = K.sum(y_true * y_pred, axis=-1)  # [batch,]
    
    # 3. Модулятор gamma
    modulator = K.pow(1. - p_t, config.focal_gamma)  # [batch,]
    
    # 4. Итоговый loss (уже имеет размерность [batch,])
    loss = modulator * ce
    
    # Возвращаем среднее по батчу (корректно для Keras)
    return K.mean(loss)

def se_block(input_tensor):
    channels = input_tensor.shape[-1]
    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape((1, 1, channels))(se)
    se = Dense(channels//config.se_reduction, activation='swish',
               kernel_regularizer=l1_l2(config.l1_value, config.l2_value))(se)
    se = Dense(channels, activation='sigmoid',
               kernel_regularizer=l1_l2(config.l1_value, config.l2_value))(se)
    return Multiply()([input_tensor, se])

def residual_block(x, filters, stride=1):
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
    inputs = Input(shape=config.input_shape)
    
    # Бэкбон CNN
    x = Conv2D(64, (7,7), strides=2, padding='same',
               kernel_regularizer=l1_l2(config.l1_value, config.l2_value))(inputs)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)
    x = MaxPooling2D((3,3), strides=2, padding='same')(x)
    
    # Residual Blocks
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 512, stride=2)
    x = residual_block(x, 512)
    
    # Головная часть
    x = GlobalAveragePooling2D()(x)
    x = LayerNormalization()(x)
    x = Dense(2048, activation='swish',
              kernel_regularizer=l1_l2(config.l1_value, config.l2_value))(x)
    x = Dropout(config.dropout_rate)(x)
    x = MoE(config.num_experts, config.expert_units)(x)
    
    outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    model = Model(inputs, outputs, name='AnimeClassifier')
    optimizer = SGD(learning_rate=config.initial_learning_rate,
                    momentum=0.95, 
                    nesterov=True)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    model.compile(optimizer=optimizer,
                  loss=focal_loss,
                  metrics=['accuracy', 'precision', 'recall', 'auc', 'top_k_categorical_accuracy'])
    return model

# ====================== ОБРАБОТКА ДАННЫХ ======================
def create_dataset(subset):
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
    def on_epoch_end(self, epoch, logs=None):
        print('\n' + '=' * 100 + '\n')

# ====================== ОБУЧЕНИЕ МОДЕЛИ ======================
def run_training():
    os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
    train_ds_raw = create_dataset('training')
    val_ds_raw = create_dataset('validation')
    num_classes = len(train_ds_raw.class_names)
    save_labels(train_ds_raw.class_names)

    # Предобработка данных
    def static_preprocessing(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    augmentations = tf.keras.Sequential([
        RandomRotation(config.rotation_range),
        RandomZoom(config.zoom_range),
        RandomContrast(config.contrast_range),
        RandomBrightness(config.brightness_range),
        RandomFlip('horizontal'),
        RandomCrop(config.input_shape[0], config.input_shape[1]),  # Случайная обрезка
        RandomSaturation(config.saturation_range),  # Изменение насыщенности
    ])

    train_ds = (
        train_ds_raw
        .map(lambda x, y: (augmentations(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)  # Аугментация
        .map(static_preprocessing, num_parallel_calls=tf.data.AUTOTUNE)  # Нормализация
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        val_ds_raw
        .map(static_preprocessing, num_parallel_calls=tf.data.AUTOTUNE)  # Нормализация
        .prefetch(tf.data.AUTOTUNE)
    )

    # Веса классов
    labels = np.concatenate([y.numpy().argmax(axis=1) for x, y in train_ds_raw], axis=0)
    total_samples = len(labels)
    class_counts = np.bincount(labels)
    class_weights = (total_samples / (len(np.unique(labels)) * class_counts)) ** config.class_weight_gamma
    class_weights = class_weights.astype(np.float32)
    class_weights_dict = {i: w for i, w in enumerate(class_weights)}

    # Инициализация модели
    if os.path.exists(config.checkpoint_path):
        model = load_model(
            config.checkpoint_path,
            custom_objects={
                'MoE': MoE,
                'focal_loss': focal_loss,
                'LayerNormalization': LayerNormalization
            }
        )
        if model.output_shape[-1] != num_classes:
            raise ValueError("Несоответствие количества классов!")
    else:
        model = build_model(num_classes)
        model.summary()

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=config.reduce_lr_factor,
                         patience=config.reduce_lr_patience, min_lr=config.min_learning_rate),
        ModelCheckpoint(config.checkpoint_path, save_best_only=True, monitor='val_loss'),
        EarlyStopping(monitor='val_loss', patience=config.early_stopping_patience),
        EpochSpacingCallback()
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=callbacks,
        class_weight=class_weights_dict
    )
    return model

# ====================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ======================
def save_labels(class_names):
    with open(config.labels_path, "w") as f:
        for label in class_names: f.write(label + "\n")

def convert_to_onnx():
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
    check_onnx_work()

def check_onnx_work():
    # Получение меток классов
    train_ds = tf.keras.utils.image_dataset_from_directory(
        config.source_dir,
        image_size=config.input_shape[:2],
        batch_size=config.batch_size,
        shuffle=False
    )
    class_names = train_ds.class_names
    with open(config.labels_path, 'w') as f:
        f.write('\n'.join(class_names))

    # Загрузка ONNX модели
    session = ort.InferenceSession(config.onnx_path)
    input_name = session.get_inputs()[0].name

    # Обработка изображения в папке
    img_path = "test.jpg"

    # Загрузка изображения без изменения размера
    img = tf.keras.preprocessing.image.load_img(img_path)
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    # Добавляем размерность батча и нормализуем
    img_array = tf.expand_dims(img_array, 0) / 255.0 # Нормализация [0,1]
    
    # Вывод информации о изображении
    print(f"\n🔍 Анализ изображения: {img_path}")
    print(f"Форма входных данных: {img_array.shape}")

    # Выполнение предсказания
    results = session.run(None, {input_name: img_array.numpy().astype(np.float32)})
    probabilities = results[0][0]
    
    # Получение топ-5 предсказаний
    top5_indices = np.argsort(probabilities)[::-1][:5]
    top5_classes = [class_names[i] for i in top5_indices]
    top5_probs = [probabilities[i] for i in top5_indices]

    # Вывод результатов
    print("\n🔮 Результаты классификации onnx:")
    for cls, prob in zip(top5_classes, top5_probs):
        print(f"  {cls}: {prob*100:.2f}%")

    model = load_model(
        config.checkpoint_path,
        custom_objects={
            'MoE': MoE,
            'focal_loss': focal_loss,
            'LayerNormalization': LayerNormalization
        }
    )

    results = model.predict(img_array)

    probabilities = results[0]
    
    # Получение топ-5 предсказаний
    top5_indices = np.argsort(probabilities)[::-1][:5]
    top5_classes = [class_names[i] for i in top5_indices]
    top5_probs = [probabilities[i] for i in top5_indices]

    print("\n🔮 Результаты классификации keras:")
    for cls, prob in zip(top5_classes, top5_probs):
        print(f"  {cls}: {prob*100:.2f}%")

# ====================== ИНТЕРФЕЙС ПОЛЬЗОВАТЕЛЯ ======================
def main():
    while True:
        print("\nМеню:\n1. Обучить\n2. Конвертировать\n3. Тест ONNX\nexit. Выход")
        choice = input("Выбор: ").strip()
        if choice == '1':
            run_training()
        elif choice == '2':
            convert_to_onnx()
        elif choice == '3':
            check_onnx_work()
        elif choice == 'exit':
            break

if __name__ == "__main__":
    main()