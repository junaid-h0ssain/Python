# CIFAR-100 IMPROVED MODEL - Optimized for 100 classes

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Clear any previous sessions
tf.keras.backend.clear_session()

# Load and properly preprocess the CIFAR-100 data
print("Loading CIFAR-100 dataset...")
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar100.load_data()

# Normalize the images to [0, 1] range
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Convert labels to categorical (one-hot encoding) - 100 classes for CIFAR-100
num_classes = 100
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

print("Data shapes after preprocessing:")
print(f"Train images: {train_images.shape}")
print(f"Train labels: {train_labels.shape}")
print(f"Test images: {test_images.shape}")
print(f"Test labels: {test_labels.shape}")

# IMPROVED MODEL ARCHITECTURE FOR CIFAR-100
# More complex architecture needed for 100 classes
model = Sequential([
    # Block 1
    Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # Block 2
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # Block 3
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # Block 4 - Additional block for CIFAR-100 complexity
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    Dropout(0.25),
    
    # Classifier - More capacity needed for 100 classes
    GlobalAveragePooling2D(),  # Better than Flatten for reducing overfitting
    Dense(1024, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # 100 classes
])

# IMPROVED COMPILATION FOR CIFAR-100
model.compile(
    optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
    loss='categorical_crossentropy',
    metrics=['accuracy', 'top_5_accuracy']  # Top-5 accuracy is important for CIFAR-100
)

model.summary()

# DATA AUGMENTATION - Critical for CIFAR-100 performance
print("Setting up data augmentation...")
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=0.1,
    shear_range=0.1,
    fill_mode='nearest'
)

# IMPROVED TRAINING WITH CALLBACKS
callbacks = [
    ReduceLROnPlateau(
        monitor='val_accuracy', 
        factor=0.5, 
        patience=7, 
        min_lr=0.00001,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy', 
        patience=15, 
        restore_best_weights=True,
        verbose=1
    )
]

# Train with data augmentation - Essential for CIFAR-100
print("Starting training with data augmentation...")
print("Note: CIFAR-100 is much more challenging than CIFAR-10")
print("Expected accuracy: 60-75% (vs 85%+ for CIFAR-10)")

history = model.fit(
    datagen.flow(train_images, train_labels, batch_size=32),
    steps_per_epoch=len(train_images) // 32,
    epochs=100,  # More epochs needed for CIFAR-100
    validation_data=(test_images, test_labels),
    callbacks=callbacks,
    verbose=1
)

# Evaluate
test_loss, test_accuracy, test_top5_accuracy = model.evaluate(test_images, test_labels)
print(f"\nCIFAR-100 Results:")
print(f"Test Accuracy (Top-1): {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Test Accuracy (Top-5): {test_top5_accuracy:.4f} ({test_top5_accuracy*100:.2f}%)")

# CIFAR-100 is much harder - good performance benchmarks:
if test_accuracy > 0.70:
    print("🎉 EXCELLENT! >70% accuracy on CIFAR-100 is very good!")
elif test_accuracy > 0.60:
    print("✅ GOOD! >60% accuracy on CIFAR-100 is solid performance!")
elif test_accuracy > 0.50:
    print("👍 DECENT! >50% accuracy on CIFAR-100 is reasonable!")
else:
    print(f"Current accuracy: {test_accuracy*100:.2f}%. CIFAR-100 is challenging!")

print(f"\nFor comparison:")
print(f"- Random guessing: 1% accuracy")
print(f"- Good performance: 60-70% accuracy") 
print(f"- Excellent performance: 70%+ accuracy")
print(f"- State-of-the-art: 85%+ accuracy (with advanced techniques)")