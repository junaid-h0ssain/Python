# =============================================================================
# CIFAR-100 NOTEBOOK CELLS - Copy these into your notebook
# CIFAR-100 is much more challenging than CIFAR-10 (100 classes vs 10)
# =============================================================================

# CELL 1: Import libraries and clear session
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Clear session
tf.keras.backend.clear_session()
print("TensorFlow version:", tf.__version__)

# CELL 2: Load and preprocess CIFAR-100 data
print("Loading CIFAR-100 dataset...")
print("Note: CIFAR-100 has 100 classes (vs 10 in CIFAR-10), making it much more challenging")

# Load CIFAR-100 data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar100.load_data()

print("Original data shapes:")
print(f"Train images: {train_images.shape}")
print(f"Train labels: {train_labels.shape}")
print(f"Test images: {test_images.shape}")
print(f"Test labels: {test_labels.shape}")

# Normalize images
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Convert labels to categorical (100 classes)
num_classes = 100
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

print(f"\nAfter preprocessing:")
print(f"Train images: {train_images.shape}, range: [{train_images.min():.2f}, {train_images.max():.2f}]")
print(f"Train labels: {train_labels.shape}")
print(f"Number of classes: {num_classes}")

# CELL 3: Create CIFAR-100 optimized model
print("Creating CIFAR-100 optimized CNN...")
print("Architecture designed for 100-class classification")

model = Sequential([
    # Block 1 - Start with more filters for complex dataset
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
    
    # Block 4 - Additional capacity for 100 classes
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    Dropout(0.3),
    
    # Classifier - More neurons needed for 100 classes
    GlobalAveragePooling2D(),  # Reduces overfitting vs Flatten
    Dense(1024, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # 100 output classes
])

model.summary()

# CELL 4: Compile model with appropriate metrics
print("Compiling model...")
model.compile(
    optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
    loss='categorical_crossentropy',
    metrics=['accuracy', 'top_5_accuracy']  # Top-5 is important for CIFAR-100
)

# CELL 5: Setup data augmentation (critical for CIFAR-100)
print("Setting up data augmentation...")
print("Data augmentation is crucial for CIFAR-100 performance")

datagen = ImageDataGenerator(
    rotation_range=25,        # More aggressive augmentation
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.15,
    shear_range=0.15,
    fill_mode='nearest'
)

# Fit the generator
datagen.fit(train_images)

# CELL 6: Setup callbacks
callbacks = [
    ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=8,          # More patience for CIFAR-100
        min_lr=0.00001,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=20,         # More patience needed
        restore_best_weights=True,
        verbose=1
    )
]

# CELL 7: Train the model
print("Starting training...")
print("\nCIFAR-100 Performance Expectations:")
print("- Random guessing: 1% accuracy")
print("- Basic CNN: 40-55% accuracy")
print("- Good CNN: 55-70% accuracy")
print("- Advanced techniques: 70-85% accuracy")
print("- State-of-the-art: 85%+ accuracy")
print("\nNote: CIFAR-100 is much harder than CIFAR-10!")

history = model.fit(
    datagen.flow(train_images, train_labels, batch_size=32),
    steps_per_epoch=len(train_images) // 32,
    epochs=100,  # More epochs needed for CIFAR-100
    validation_data=(test_images, test_labels),
    callbacks=callbacks,
    verbose=1
)

# CELL 8: Evaluate the model
print("Evaluating model...")
results = model.evaluate(test_images, test_labels, verbose=1)
test_loss, test_accuracy, test_top5_accuracy = results

print(f"\nCIFAR-100 Final Results:")
print(f"Test Accuracy (Top-1): {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Test Accuracy (Top-5): {test_top5_accuracy:.4f} ({test_top5_accuracy*100:.2f}%)")

# Performance evaluation for CIFAR-100
if test_accuracy > 0.70:
    print("🚀 OUTSTANDING! >70% on CIFAR-100 is excellent performance!")
elif test_accuracy > 0.60:
    print("🎉 GREAT! >60% on CIFAR-100 is very good!")
elif test_accuracy > 0.50:
    print("✅ GOOD! >50% on CIFAR-100 is solid performance!")
elif test_accuracy > 0.40:
    print("👍 DECENT! >40% on CIFAR-100 is reasonable!")
else:
    print("📈 Keep improving! CIFAR-100 is very challenging.")

print(f"\nContext:")
print(f"- Your result: {test_accuracy*100:.1f}% (Top-1), {test_top5_accuracy*100:.1f}% (Top-5)")
print(f"- Random guessing: 1.0% (Top-1), 5.0% (Top-5)")
print(f"- Good performance: 60%+ (Top-1), 85%+ (Top-5)")

# CELL 9: Plot training history
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy (Top-1)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history.history['top_5_accuracy'], label='Training Top-5 Accuracy')
plt.plot(history.history['val_top_5_accuracy'], label='Validation Top-5 Accuracy')
plt.title('Model Accuracy (Top-5)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# CELL 10: Save the model
model.save('cifar100_model.h5')
print("Model saved as 'cifar100_model.h5'")

# =============================================================================
# KEY DIFFERENCES FROM CIFAR-10:
#
# 1. DATASET COMPLEXITY:
#    - CIFAR-10: 10 classes, easier to distinguish
#    - CIFAR-100: 100 classes, much more challenging
#
# 2. EXPECTED PERFORMANCE:
#    - CIFAR-10: 85-95% achievable with good CNN
#    - CIFAR-100: 60-75% is good performance
#
# 3. ARCHITECTURE CHANGES:
#    - More filters in conv layers (64 vs 32 start)
#    - Additional conv block for complexity
#    - Larger dense layers (1024, 512 vs 512, 256)
#    - GlobalAveragePooling2D to reduce overfitting
#
# 4. TRAINING CHANGES:
#    - More aggressive data augmentation
#    - More epochs needed (100 vs 50)
#    - More patience in callbacks
#    - Top-5 accuracy metric added
#
# 5. EVALUATION:
#    - Top-1 accuracy: Your model's best guess
#    - Top-5 accuracy: Correct class in top 5 predictions
#    - Both metrics important for 100-class problems
# =============================================================================