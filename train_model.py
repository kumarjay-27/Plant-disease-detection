import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import json
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# GPU configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# choose only 10 classes
selected_classes = [
"Pepper__bell___Bacterial_spot",
"Pepper__bell___healthy",
"Potato___Early_blight",
"Potato___healthy",
"Potato___Late_blight",
"Tomato_Bacterial_spot",
"Tomato_Early_blight",
"Tomato_healthy",
"Tomato_Late_blight",
"Tomato_Leaf_Mold"
]
dataset_path = r"PlantVillage/train_set"
print("Dataset path exists:", os.path.exists(dataset_path))
for cls in selected_classes:
    cls_path = os.path.join(dataset_path, cls)
    if os.path.exists(cls_path):
        print(cls, "images:", len(os.listdir(cls_path)))
    else:
        print(cls, "folder NOT FOUND")

img_size = 224
batch_size = 32
epochs = 10


train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size,img_size),
    batch_size=batch_size,
    classes=selected_classes,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size,img_size),
    batch_size=batch_size,
    classes=selected_classes,
    class_mode='categorical',
    subset='validation'
)

print("Classes:",train_generator.class_indices)

# save class names
os.makedirs("model",exist_ok=True)

class_names = list(train_generator.class_indices.keys())

with open("class_names.json","w") as f:
    json.dump(class_names,f)

# Load MobileNetV2
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128,activation='relu')(x)
x = Dropout(0.3)(x)

predictions = Dense(len(selected_classes),activation='softmax')(x)

model = Model(inputs=base_model.input,outputs=predictions)

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "model/plant_disease_model.keras",
    monitor='val_accuracy',
    save_best_only=True
)

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=[early_stop,checkpoint]
)

print("Training Complete")
