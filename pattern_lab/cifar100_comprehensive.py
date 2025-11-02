# COMPREHENSIVE CIFAR-100 SOLUTION
# Multiple approaches for achieving good performance on CIFAR-100

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
import numpy as np
import matplotlib.pyplot as plt

# Clear session
tf.keras.backend.clear_session()

# Load CIFAR-100 data
print("Loading CIFAR-100 dataset...")
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar100.load_data()

# Data preprocessing
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

num_classes = 100
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

print(f"Training data: {train_images.shape}")
print(f"Test data: {test_images.shape}")
print(f"Number of classes: {num_classes}")

def create_basic_cnn():
    """Basic CNN for CIFAR-100"""
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(0.25),
        
        GlobalAveragePooling2D(),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def create_advanced_cnn():
    """Advanced CNN with residual-like connections"""
    inputs = Input(shape=(32, 32, 3))
    
    # Initial conv
    x = Conv2D(64, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Block 1
    shortcut = x
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Block 2
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    shortcut = Conv2D(128, (1, 1), padding='same')(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Block 3
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    shortcut = Conv2D(256, (1, 1), padding='same')(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Final layers
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model

def create_transfer_learning_model():
    """Transfer learning with ResNet50"""
    # Load pre-trained ResNet50
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(32, 32, 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

# Choose model architecture
print("Creating model...")
model_choice = "basic"  # Change to "advanced" or "transfer" for other models

if model_choice == "basic":
    model = create_basic_cnn()
elif model_choice == "advanced":
    model = create_advanced_cnn()
elif model_choice == "transfer":
    model = create_transfer_learning_model()

model.summary()

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', 'top_5_accuracy']
)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.15,
    shear_range=0.15,
    fill_mode='nearest'
)

# Callbacks
callbacks = [
    ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=8,
        min_lr=0.00001,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'best_cifar100_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Training
print("Starting training...")
print("CIFAR-100 Performance Expectations:")
print("- Basic CNN: 50-65% accuracy")
print("- Advanced CNN: 60-75% accuracy") 
print("- Transfer Learning: 65-80% accuracy")
print("- State-of-the-art: 85%+ accuracy")

history = model.fit(
    datagen.flow(train_images, train_labels, batch_size=32),
    steps_per_epoch=len(train_images) // 32,
    epochs=150,
    validation_data=(test_images, test_labels),
    callbacks=callbacks,
    verbose=1
)

# Load best model and evaluate
model.load_weights('best_cifar100_model.h5')
results = model.evaluate(test_images, test_labels, verbose=0)
test_loss, test_accuracy, test_top5_accuracy = results

print(f"\nFinal CIFAR-100 Results:")
print(f"Test Accuracy (Top-1): {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Test Accuracy (Top-5): {test_top5_accuracy:.4f} ({test_top5_accuracy*100:.2f}%)")

# Performance evaluation
if test_accuracy > 0.75:
    print("🚀 OUTSTANDING! >75% on CIFAR-100 is excellent!")
elif test_accuracy > 0.65:
    print("🎉 GREAT! >65% on CIFAR-100 is very good!")
elif test_accuracy > 0.55:
    print("✅ GOOD! >55% on CIFAR-100 is solid!")
elif test_accuracy > 0.45:
    print("👍 DECENT! >45% on CIFAR-100 is reasonable!")
else:
    print("📈 Keep improving! CIFAR-100 is very challenging.")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

print("Model saved as 'best_cifar100_model.h5'")