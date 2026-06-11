import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0

IMG_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 80
DATA_DIR = "train"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.6, 1.4],
    validation_split=0.15
)

train_data = train_gen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = train_gen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

base = EfficientNetB0(include_top=False, weights='imagenet',
                      input_shape=(IMG_SIZE, IMG_SIZE, 3))

for layer in base.layers[:-20]:
    layer.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    callbacks.ModelCheckpoint("emotion_model.keras", save_best_only=True),
    callbacks.EarlyStopping(patience=8, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(patience=3)
]

model.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=callbacks)