# =============================================================================
# NOTEBOOK-FRIENDLY CIFAR-10 IMPROVEMENT
# Copy these cells into your notebook to fix the error and improve accuracy
# =============================================================================

# CELL 1: Clear session and import libraries
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Clear any previous sessions to avoid conflicts
tf.keras.backend.clear_session()
print("TensorFlow version:", tf.__version__)

# CELL 2: Load and preprocess data properly
print("Loading and preprocessing CIFAR-10 data...")

# Load the data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Check original data
print("Original shapes:")
print(f"Train images: {train_images.shape}")
print(f"Train labels: {train_labels.shape}")

# Normalize images to [0, 1] range
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Convert labels to categorical (one-hot encoding)
num_classes = 10
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

print("After preprocessing:")
print(f"Train images: {train_images.shape}, range: [{train_images.min():.2f}, {train_images.max():.2f}]")
print(f"Train labels: {train_labels.shape}")
print(f"Label example: {train_labels[0]} (sum: {train_labels[0].sum()})")

# CELL 3: Create improved CNN model
print("Creating improved CNN model...")

model = Sequential([
    # First Convolutional Block
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # Second Convolutional Block
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # Third Convolutional Block
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # Classifier
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.summary()

# CELL 4: Compile model
print("Compiling model...")
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# CELL 5: Setup callbacks
callbacks = [
    ReduceLROnPlateau(
        monitor='val_accuracy', 
        factor=0.5, 
        patience=5, 
        min_lr=0.00001,
        verbose=1
    )
]

# CELL 6: Train the model (with error handling)
print("Starting training...")

# Verify data before training
print("Final data check before training:")
print(f"X_train shape: {train_images.shape}, dtype: {train_images.dtype}")
print(f"y_train shape: {train_labels.shape}, dtype: {train_labels.dtype}")
print(f"X_test shape: {test_images.shape}, dtype: {test_images.dtype}")
print(f"y_test shape: {test_labels.shape}, dtype: {test_labels.dtype}")

try:
    # Train the model
    history = model.fit(
        train_images, train_labels,
        batch_size=32,
        epochs=50,
        validation_data=(test_images, test_labels),
        callbacks=callbacks,
        verbose=1
    )
    
    print("Training completed successfully!")
    
except Exception as e:
    print(f"Training error: {e}")
    print("This error might be due to data format issues or TensorFlow version conflicts.")
    print("Try restarting your kernel and running the cells again.")

# CELL 7: Evaluate the model
print("Evaluating model...")
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=1)
print(f"\nFinal Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

if test_accuracy > 0.85:
    print("🎉 SUCCESS! Achieved >85% test accuracy!")
else:
    print(f"Current accuracy: {test_accuracy*100:.2f}%. Getting closer to 85%!")

# =============================================================================
# TROUBLESHOOTING TIPS:
# 
# If you still get the ValueError:
# 1. Restart your kernel/runtime
# 2. Make sure you're using the correct TensorFlow version (2.x)
# 3. Check that your data shapes are correct
# 4. Try running with smaller batch sizes (16 instead of 32)
# 5. Make sure all imports are correct
# 
# The error "Attr 'Toutput_types' of 'OptionalFromValue' Op passed list of 
# length 0" usually indicates:
# - Data pipeline issues
# - Incorrect data types
# - TensorFlow version conflicts
# - Memory issues
# =============================================================================