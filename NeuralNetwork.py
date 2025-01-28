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

# Установка политики смешанной точности для ускорения обучения
set_global_policy('mixed_float16')

class Config:
    # Параметры модели
    input_shape = (224, 224, 3)    # Размер входных изображений
    l1_value = 1e-4                # Коэффициент L1-регуляризации
    l2_value = 1e-5                # Коэффициент L2-регуляризации
    num_experts = 8                # Количество экспертов в слое MoE
    expert_units = 1024            # Количество нейронов в каждом эксперте
    dropout_rate = 0.3             # Процент дропаута
    
    # Параметры обучения
    initial_learning_rate = 1e-0   # Начальная скорость обучения
    batch_size = 64                # Размер батча
    min_learning_rate = 1e-10      # Минимальная скорость обучения
    reduce_lr_factor = 0.5         # Фактор уменьшения скорости обучения
    reduce_lr_patience = 2         # Количество эпох без улучшений для уменьшения LR
    early_stopping_patience = 5    # Количество эпох для ранней остановки
    epochs = 1000                  # Количество эпох обучения
    
    # Параметры Focal Loss для работы с несбалансированными классами
    focal_alpha = 0.25             # Весовой коэффициент для классов
    focal_gamma = 2.0              # Коэффициент фокусировки
    
    # Пути для сохранения результатов
    source_dir = "/media/alex/Programs/NeuralNetwork/DataSet/ARTS/Original"
    checkpoint_path = "/media/alex/Programs/NeuralNetwork/Model/best_model.keras"
    labels_path = "/media/alex/Programs/NeuralNetwork/Model/labels.txt"
    onnx_path = "/media/alex/Programs/NeuralNetwork/Model/model.onnx"

config = Config()

class MoE(Layer):
    """Слой 'Смесь экспертов' (Mixture of Experts)"""
    def __init__(self, num_experts, expert_units, **kwargs):
        super(MoE, self).__init__(**kwargs)
        self.num_experts = num_experts
        self.expert_units = expert_units

    def build(self, input_shape):
        self.experts = [
            tf.keras.Sequential([
                Dense(self.expert_units, activation='swish',
                      kernel_regularizer=l1_l2(config.l1_value, config.l2_value)),
                Dropout(config.dropout_rate),
                Dense(input_shape[-1],
                      kernel_regularizer=l1_l2(config.l1_value, config.l2_value))
            ]) for _ in range(self.num_experts)
        ]
        self.router = Dense(self.num_experts, activation='softmax',
                            kernel_regularizer=l1_l2(config.l1_value, config.l2_value))
        super(MoE, self).build(input_shape)

    def call(self, inputs):
        weights = self.router(inputs)  # (batch_size, num_experts)
        expert_outputs = tf.stack([expert(inputs) for expert in self.experts], axis=1)  # (batch_size, num_experts, units)
        weighted_outputs = tf.einsum('be,beu->bu', weights, expert_outputs)
        return weighted_outputs + inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_experts': self.num_experts,
            'expert_units': self.expert_units
        })
        return config

def focal_loss(y_true, y_pred):
    alpha = config.focal_alpha
    gamma = config.focal_gamma
    
    y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
    cross_entropy = -y_true * K.log(y_pred)
    loss = alpha * K.pow(1. - y_pred, gamma) * cross_entropy
    return K.sum(loss, axis=1)

def se_block(input_tensor, reduction=16):
    channels = input_tensor.shape[-1]
    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape((1, 1, channels))(se)
    se = Dense(channels // reduction, activation='swish',
               kernel_regularizer=l1_l2(config.l1_value, config.l2_value))(se)
    se = Dense(channels, activation='sigmoid',
               kernel_regularizer=l1_l2(config.l1_value, config.l2_value))(se)
    return Multiply()([input_tensor, se])

def residual_block(x, filters, stride=1):
    shortcut = x
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), strides=stride,
                          kernel_regularizer=l1_l2(config.l1_value, config.l2_value))(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    x = Conv2D(filters, (3, 3), strides=stride, padding='same',
               kernel_regularizer=l1_l2(config.l1_value, config.l2_value))(x)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)
    
    x = Conv2D(filters, (3, 3), padding='same',
               kernel_regularizer=l1_l2(config.l1_value, config.l2_value))(x)
    x = BatchNormalization()(x)
    x = se_block(x)
    
    x = Add()([x, shortcut])
    return Activation('swish')(x)

def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    x = Conv2D(64, (7, 7), strides=2, padding='same',
               kernel_regularizer=l1_l2(config.l1_value, config.l2_value))(inputs)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)
    
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 512, stride=2)
    
    x = GlobalAveragePooling2D()(x)
    x = LayerNormalization()(x)
    
    x = Dense(1024, activation='swish',
              kernel_regularizer=l1_l2(config.l1_value, config.l2_value))(x)
    x = Dropout(config.dropout_rate)(x)
    x = MoE(config.num_experts, config.expert_units)(x)
    
    outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    model = Model(inputs, outputs, name='CustomCNN')
    model._name = 'CustomCNN' 
    optimizer = SGD(learning_rate=config.initial_learning_rate, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer,
                  loss=focal_loss,
                  metrics=['accuracy'])
    return model

def create_dataset(subset):
    return tf.keras.utils.image_dataset_from_directory(
        config.source_dir,
        labels='inferred',
        label_mode='categorical',
        color_mode='rgb',
        batch_size=config.batch_size,
        image_size=(config.input_shape[:2]),
        validation_split=0.2,
        subset=subset,
        seed=123,
        shuffle=(subset == 'training')
    )

class EpochSpacingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('\n-------------------------------------------------------------------------------------------------------------------\n')  # Добавляет две пустые строки после каждой эпохи

def run_training():
    # Создание и проверка директорий
    os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
    
    # Загрузка данных
    train_ds = create_dataset('training')
    val_ds = create_dataset('validation')
    num_classes = len(train_ds.class_names)

    # Аугментации
    augmentations = tf.keras.Sequential([
        RandomRotation(0.2),
        RandomZoom(0.3),
        RandomContrast(0.2),
        RandomBrightness(0.3),
        tf.keras.layers.RandomFlip(mode='horizontal_and_vertical'),
    ])

    train_ds = train_ds.map(
        lambda x, y: (augmentations(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    # Расчет весов классов
    labels = np.concatenate([y.numpy().argmax(axis=1) for x, y in train_ds], axis=0)
    class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    class_weights_dict = {i: w for i, w in enumerate(class_weights)}

    # Проверяем, существует ли модель и её архитектура
    if os.path.exists(config.checkpoint_path):
        print("\nЗагрузка предобученной модели...\n")
        model = load_model(
            config.checkpoint_path,
            custom_objects={
                'MoE': MoE,
                'focal_loss': focal_loss,
                'LayerNormalization': LayerNormalization
            }
        )

    else:
        print("\nСоздание новой модели...\n")
        # Построение модели
        model = build_model(config.input_shape, num_classes)
        model.summary()

    # Колбэки
    callbacks = [
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.reduce_lr_factor,
            patience=config.reduce_lr_patience,
            min_lr=config.min_learning_rate,
            verbose=1
        ),
        ModelCheckpoint(
            config.checkpoint_path,
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=config.early_stopping_patience,
            restore_best_weights=True
        ),
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

def save_labels():
    """Сохраняет текстовые метки из директорий классов."""
    class_names = sorted(os.listdir(config.source_dir))
    with open(config.labels_path, "w") as f:
        for label in class_names:
            f.write(label + "\n")
    print(f"Метки сохранены в {config.labels_path}")
    return class_names

def convert_to_onnx():
    import tf2onnx
    import onnxruntime as ort
    
    model = tf.keras.models.load_model(
        config.checkpoint_path,
        custom_objects={'MoE': MoE, 'focal_loss': focal_loss}
    )
    
    input_signature = [tf.TensorSpec(shape=[None, *config.input_shape], dtype=tf.float32)]
    tf2onnx.convert.from_keras(model, input_signature=input_signature, output_path=config.onnx_path)
    
    # Тестирование ONNX модели
    session = ort.InferenceSession(config.onnx_path)
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, *config.input_shape).astype(np.float32)
    outputs = session.run(None, {input_name: dummy_input})
    print("\nТест ONNX модели успешно выполнен!")

    save_labels()

def main():
    while True:
        print("\nМеню:")
        print("1. Обучить модель")
        print("2. Конвертировать в ONNX")
        print("3. Выход")
        choice = input("Выберите действие (1-3): ").strip()

        if choice == '1':
            if not os.path.exists(config.source_dir):
                print(f"Ошибка: Директория с данными {config.source_dir} не существует!")
                continue
            run_training()
            print("Обучение завершено!")
        elif choice == '2':
            if not os.path.exists(config.checkpoint_path):
                print("Ошибка: Файл модели не найден! Сначала обучите модель.")
                continue
            convert_to_onnx()
        elif choice == '3':
            break
        else:
            print("Неверный ввод! Попробуйте снова.")

if __name__ == "__main__":
    main()