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
import tf2onnx
import onnxruntime as ort
import matplotlib.pyplot as plt
import math

# ========================== КОНФИГУРАЦИЯ ==========================
class Config:
    # -------------------- Архитектура модели --------------------
    input_shape = (224, 224, 3)    # Размер входных изображений (H, W, C)
    l1_value = 1e-5                # Коэффициент L1-регуляризации
    l2_value = 1e-4                # Коэффициент L2-регуляризации
    dropout_rate = 0.5             # Процент дропаута
    num_experts = 4                # Количество экспертов в слое MoE
    expert_units = 2048            # Нейронов в каждом эксперте
    se_reduction = 16              # Коэффициент уменьшения в SE-блоке

    # --------------------- Параметры обучения ---------------------
    initial_learning_rate = 1e-1   # Начальная скорость обучения
    batch_size = 32                # Размер батча
    epochs = 1000                  # Максимальное число эпох
    min_learning_rate = 1e-10      # Минимальная скорость обучения
    reduce_lr_factor = 0.25        # Фактор уменьшения LR
    reduce_lr_patience = 1         # Терпение для уменьшения LR
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

# ====================== КАСТОМНЫЕ КОМПОНЕНТЫ ======================
class MoE(Layer):
    def __init__(self, num_experts=8, expert_units=1024, **kwargs):  # Добавляем параметры в конструктор
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
    x = MoE(config.num_experts, config.expert_units)(x)
    
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

def plot_images(ds, train_ds_raw, num_images=30, filename='samples.png', cols=5):
    rows = math.ceil(num_images / cols)  # Вычисляем количество строк
    plt.figure(figsize=(cols * 5, rows * 5))  # Автоматическое масштабирование

    for i, (image, label) in enumerate(ds.take(num_images)):
        image = image[0]  # Берем первое изображение из батча
        plt.subplot(rows, cols, i + 1)  # Размещаем по строкам и колонкам
        plt.imshow((image * 255.0).numpy().astype('uint8'))  # Денормализация для uint8
        plt.title(train_ds_raw.class_names[label.numpy().argmax()], fontsize=18)
        plt.axis('off')

    plt.tight_layout()  # Убираем наложение
    plt.savefig(filename)
    plt.close()

# ====================== ОБУЧЕНИЕ МОДЕЛИ ======================
def run_training():
    """Запуск процесса обучения с аугментацией"""
    os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)

    # Загрузка данных
    train_ds_raw = create_dataset('training')
    val_ds_raw = create_dataset('validation')
    num_classes = len(train_ds_raw.class_names)

    # Статическая предобработка (общая для train/val)
    def static_preprocessing(image, label):
        #Нормализация
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    # Аугментации (только для тренировочных данных)
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

    # Пайплайн для тренировочных данных
    train_ds = (
        train_ds_raw
        .map(lambda x, y: (augmentations(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        .map(static_preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Пайплайн для валидационных данных
    val_ds = (
        val_ds_raw
        .map(static_preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # вывод изображений
    plot_images(train_ds, train_ds_raw, filename='train_samples.png', cols=5)
    plot_images(val_ds, train_ds_raw, filename='val_samples.png', cols=5)

    # Веса классов (считаем на исходных данных)
    labels = np.concatenate([y.numpy().argmax(axis=1) for x, y in train_ds_raw], axis=0)
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
    """Конвертация модели в ONNX с тестированием на реальных изображениях"""
    # Загрузка модели
    model = load_model(
        config.checkpoint_path,
        custom_objects={
            'MoE': MoE,
            'focal_loss': focal_loss,
            'LayerNormalization': LayerNormalization
        }
    )

    # Конвертация в ONNX
    input_signature = [tf.TensorSpec(shape=[None, *config.input_shape], dtype=tf.float32)]
    tf2onnx.convert.from_keras(model, input_signature=input_signature, output_path=config.onnx_path)
    save_labels()

    print("модель сохранена")

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
    """Главное меню программы"""
    while True:
        print("\nМеню:")
        print("1. Обучить модель")
        print("2. Конвертировать в ONNX")
        print("3. Тест ONNX")
        print("exit. Выход")
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
            check_onnx_work()
        elif choice == 'exit':
            print("Выход...")
            break
        else:
            print("Неверный ввод!")

if __name__ == "__main__":
    main()