# model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D, Add, Reshape, Multiply
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import SGD
from sklearn.utils.class_weight import compute_class_weight

# notification
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Отключение всех сообщений
os.environ['XLA_FLAGS'] = '--xla_hlo_profile=false'  # Отключить профилирование XLA

# params
from tensorflow.keras.mixed_precision import set_global_policy

# convert
import tf2onnx
import tensorflow as tf
import onnxruntime as ort
import numpy as np

set_global_policy('mixed_float16')

# SE-блок
def se_block(input_tensor, reduction=16):
    """Добавление Squeeze-and-Excitation блока."""
    channel_axis = -1  # Ось каналов (последняя ось)
    filters = input_tensor.shape[channel_axis]
    
    # Сжатие: глобальный средний пуллинг
    se_tensor = GlobalAveragePooling2D()(input_tensor)
    se_tensor = Reshape((1, 1, filters))(se_tensor)
    
    # Excitation: полносвязные слои для вычисления важности каналов
    se_tensor = Dense(filters // reduction, activation='relu', kernel_regularizer=l2(0.001))(se_tensor)
    se_tensor = Dense(filters, activation='sigmoid', kernel_regularizer=l2(0.001))(se_tensor)

    # Масштабирование входного тензора по вычисленным весам
    return Multiply()([input_tensor, se_tensor])

# Модифицированный Bottleneck-блок с SE-блоком
def bottleneck_block(input_tensor, filters, stride=1, reduction=16):
    """Bottleneck-блок с Squeeze-and-Excitation."""
    shortcut = input_tensor

    # Проекционный shortcut (если требуется изменение размерности)
    if shortcut.shape[-1] != filters * 4 or stride > 1:
        shortcut = Conv2D(filters * 4, (1, 1), strides=stride, padding='same', kernel_regularizer=l2(0.001))(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # Слой 1: 1x1 свёртка (уменьшение числа каналов)
    x = Conv2D(filters, (1, 1), strides=stride, padding='same', kernel_regularizer=l2(0.001))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Слой 2: 3x3 свёртка
    x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Слой 3: 1x1 свёртка (восстановление размерности)
    x = Conv2D(filters * 4, (1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)

    # Применяем SE-блок
    x = se_block(x, reduction)

    # Добавление shortcut
    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x

# Функция для создания серии Bottleneck-блоков с SE
def build_residual_blocks(input_tensor, num_blocks, filters, stride=1, reduction=16):
    """Создаёт серию Bottleneck-блоков с SE-блоком."""
    x = bottleneck_block(input_tensor, filters, stride=stride, reduction=reduction)
    for _ in range(1, num_blocks):
        x = bottleneck_block(x, filters, stride=1, reduction=reduction)
    return x

# Основная модель с SE-блоками
def build_cnn(input_shape, num_classes):
    """Создаёт свёрточную нейронную сеть с Bottleneck-блоками и SE-блоками."""
    # Входной слой
    input_tensor = Input(shape=input_shape)
    x = input_tensor

    # Начальная свёртка
    x = Conv2D(64, (7, 7), strides=2, padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    # # Residual-блоки с SE (для ResNet-50)
    # x = build_residual_blocks(x, num_blocks=3, filters=64, stride=1)   # Conv2_x
    # x = build_residual_blocks(x, num_blocks=4, filters=128, stride=2)  # Conv3_x
    # x = build_residual_blocks(x, num_blocks=6, filters=256, stride=2)  # Conv4_x
    # x = build_residual_blocks(x, num_blocks=3, filters=512, stride=2)  # Conv5_x

    # # Residual-блоки с SE (для ResNet-101)
    # x = build_residual_blocks(x, num_blocks=3, filters=64, stride=1)   # Conv2_x
    # x = build_residual_blocks(x, num_blocks=4, filters=128, stride=2)  # Conv3_x
    # x = build_residual_blocks(x, num_blocks=23, filters=256, stride=2) # Conv4_x
    # x = build_residual_blocks(x, num_blocks=3, filters=512, stride=2)  # Conv5_x

    # Residual-блоки с SE (для ResNet-152)
    x = build_residual_blocks(x, num_blocks=3, filters=64, stride=1)   # Conv2_x
    x = build_residual_blocks(x, num_blocks=8, filters=128, stride=2)  # Conv3_x
    x = build_residual_blocks(x, num_blocks=36, filters=256, stride=2) # Conv4_x
    x = build_residual_blocks(x, num_blocks=3, filters=512, stride=2)  # Conv5_x

    # Глобальный усреднённый пуллинг
    x = GlobalAveragePooling2D()(x)

    # Полносвязная голова
    x = Dense(1024, kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    # Выходной слой
    output_tensor = Dense(num_classes, activation='softmax')(x)

    # Создание модели
    model = Model(inputs=input_tensor, outputs=output_tensor)

    # Компиляция модели
    optimizer = SGD(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    return model

def run_learning():
    """Обучение нейроной сети"""
    # Параметры
    img_size = (224, 224)  # Размер изображения (224x224)
    batch_size = 5
    images_per_epochs_count = 10000  # Текущее количество изображений для обучения
    steps_per_epoch = images_per_epochs_count // batch_size
    steps_per_epoch += 1
    base_epochs = 100

    # Генерация данных
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,  # Масштабирование пикселей в диапазон [0, 1]
        validation_split=0.2,  # Разделение данных на обучение и валидацию
        rotation_range=20,  # Повороты до 20 градусов
        width_shift_range=0.2,  # Горизонтальное смещение до 20% от ширины
        height_shift_range=0.2,  # Вертикальное смещение до 20% от высоты
        shear_range=0.15,  # Сдвиг (shear) до 15%
        zoom_range=0.2,  # Увеличение или уменьшение масштаба до 20%
        fill_mode='nearest'  # Заполнение новых пикселей при смещении (nearest, constant, reflect, wrap)
    )   

    validation_generator = datagen.flow_from_directory(
        source_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    input_shape = (img_size[0], img_size[1], 3)

    # Проверяем, существует ли модель и её архитектура
    if os.path.exists(checkpoint_model_filename):
        print("\nЗагрузка предобученной модели...\n")
        model = load_model(checkpoint_model_filename)
    else:
        print("\nСоздание новой модели...\n")
        model = build_cnn(input_shape, len(validation_generator.class_indices))

    # Пересоздаём train_generator на каждую эпоху
    train_generator = datagen.flow_from_directory(
        source_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True  # Перемешиваем данные
    )

    train_images_count = train_generator.samples  # Количество изображений для обучения

    # Вычисляем веса классов
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weights_dict = dict(enumerate(class_weights))

    epochs_per_image_batch = train_images_count // images_per_epochs_count
    epochs_per_image_batch += 1

    # Состояние для ручного early_stopping
    best_val_loss = float('inf')
    patience = 2 * epochs_per_image_batch
    wait = 0

    full_epoch = base_epochs * epochs_per_image_batch

    # Обучение модели
    for epoch in range(full_epoch):
        print(f"\nЗапуск эпохи {epoch + 1}/{full_epoch}")

        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_generator,
            epochs=1,  # Одна эпоха за раз
            class_weight=class_weights_dict  # Добавляем веса классов
        )

        # Логика ручного early_stopping
        val_loss = history.history['val_loss'][0]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            # Сохраняем лучшую модель
            model.save(checkpoint_model_filename)
            print(f"Лучшая модель сохранена в {checkpoint_model_filename}")
        else:
            wait += 1
            if wait >= patience:
                print("\nEarly stopping сработал!")
                break

        # Пересоздаём train_generator на каждую эпоху
        train_generator = datagen.flow_from_directory(
            source_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True  # Перемешиваем данные
        )

    # Тестирование модели
    test_loss, test_accuracy = model.evaluate(validation_generator)
    print(f"\nРезультаты тестирования:\n - Потери (Loss): {test_loss:.4f}\n - Точность (Accuracy): {test_accuracy:.4f}")

labels_path = "/media/alex/Programs/NeuralNetwork/Model/labels.txt" 
def save_labels():
    """Сохраняет текстовые метки из директорий классов."""
    class_names = sorted(os.listdir(source_dir))
    with open(labels_path, "w") as f:
        for label in class_names:
            f.write(label + "\n")
    print(f"Метки сохранены в {labels_path}")
    return class_names

onnx_model_filename = "/media/alex/Programs/NeuralNetwork/Model/model.onnx" 
def load_and_convert_model():
    """Конввертирует в ONNX"""

    # Загрузка модели из контрольной точки
    model = load_model(checkpoint_model_filename)

    # Конвертация и сохранение модели в формате ONNX
    input_signature = [tf.TensorSpec([None, 224, 224, 3], tf.float32)]
    tf2onnx.convert.from_keras(model, input_signature=input_signature, output_path=onnx_model_filename)
    test_onnx_model_with_random_tensor()

def test_onnx_model_with_random_tensor():
    # Загрузка ONNX модели
    session = ort.InferenceSession(onnx_model_filename)
    
    # Получить имена входов и выходов
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Создать случайный тензор
    random_tensor = np.random.rand(1, 224, 224, 3).astype(np.float32)  # Batch size = 1
    
    # Прогнать модель с случайным тензором
    outputs = session.run([output_name], {input_name: random_tensor})
    
    # Печать результата
    print("Результат работы модели со случайным тензором:")
    print(outputs[0])
      
def main():
    while True:
        print("\nВыберите действие:")
        print("1. Обучить модель")
        print("2. Конвертировать модель в ONNX")

        choice = input("Введите номер действия (1/2): ")

        if choice == '1':
            run_learning()

        if choice == '2':
            load_and_convert_model()
            save_labels()

main()