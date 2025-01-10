from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D, Add
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Nadam

import os

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Reshape, Multiply
from tensorflow.keras import layers

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
    x = se_block(x)

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
    #x = Dense(128, kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    # Выходной слой
    output_tensor = Dense(num_classes, activation='softmax')(x)

    # Создание модели
    model = Model(inputs=input_tensor, outputs=output_tensor)

    # Компиляция модели
    optimizer = Nadam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    return model

def run():
    # Параметры
    source_dir = r"DataSet\ARTS\Original"  # Путь к папке с исходными данными
    img_size = (224, 224)  # Размер изображения (224x224)
    batch_size = 32
    images_per_epochs_count = 5000  # Текущее количество изображений для обучения
    steps_per_epoch = images_per_epochs_count // batch_size
    base_epochs = 100

    result_model_filename = r"Model\result_model.keras"
    checkpoint_model_filename = r"Model\checkpoint_model.keras" 

    # Генерация данных
    datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    validation_generator = datagen.flow_from_directory(
        source_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    input_shape = (img_size[0], img_size[1], 3)

    # Проверяем, существует ли модель и её архитектура
    if os.path.exists(result_model_filename):
        print("\nЗагрузка предобученной модели...\n")
        model = load_model(result_model_filename)
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

    train_images_count = train_generator.samples # количество изображений для обучения

    epochs_per_image_batch = train_images_count // images_per_epochs_count

    # Состояние для ручного early_stopping
    best_val_loss = float('inf')
    patience = 1 * epochs_per_image_batch
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

    # Сохранение итоговой модели
    model.save(result_model_filename)
    print(f"Модель сохранена в {result_model_filename}")


run()
