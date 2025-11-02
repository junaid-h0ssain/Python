# FIXED CIFAR-10 MODEL - This should resolve the ValueError

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Clear any previous sessions
tf.keras.backend.clear_session()

print("TensorFlow version:", tf.__version__)

# Load CIFAR-10 data
print("Loading CIFAR-10 dataset...")
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

print("Original data shapes:")
print(f"Train images: {train_images.shape}, dtype: {train_images.dtype}")
print(f"Train labels: {train_labels.shape}, dtype: {train_labels.dtype}")
print(f"Test images: {test_images.shape}, dtype: {test_images.dtype}")
print(f"Test labels: {test_labels.shape}, dtype: {test_labels.dtype}")

# Normalize pixel values to [0, 1] range
print("Normalizing images...")
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Convert labels to categorical (one-hot encoding)
print("Converting labels to categorical...")
num_classes = 10
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

print("Processed data shapes:")
print(f"Train images: {train_images.shape}, dtype: {train_images.dtype}")
print(f"Train labels: {train_labels.shape}, dtype: {train_labels.dtype}")
print(f"Test images: {test_images.shape}, dtype: {test_images.dtype}")
print(f"Test labels: {test_labels.shape}, dtype: {test_labels.dtype}")

# Verify data integrity
print("Data verification:")
print(f"Train images - min: {train_images.min()}, max: {train_images.max()}")
print(f"Test images - min: {test_images.min()}, max: {test_images.max()}")
print(f"Train labels - shape: {train_labels.shape}, sum per sample: {train_labels[0].sum()}")

# Create the improved CNN model
print("Creating improved CNN model...")
model = Sequential([
    # Block 1
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # Block 2
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # Block 3
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

# Display model architecture
model.summary()

# Compile the model
print("Compiling model...")
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Setup callbacks
callbacks = [
    ReduceLROnPlateau(
        monitor='val_accuracy', 
        factor=0.5, 
        patience=5, 
        min_lr=0.00001,
        verbose=1
    )
]

# Train the model with proper error handling
print("Starting training...")
try:
    history = model.fit(
        x=train_images,
        y=train_labels,
        batch_size=32,
        epochs=30,  # Start with fewer epochs for testing
        validation_data=(test_images, test_labels),
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )
    
    print("Training completed successfully!")
    
    # Evaluate the model
    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    if test_accuracy > 0.85:
        print("🎉 SUCCESS! Achieved >85% test accuracy!")
    else:
        print(f"Current accuracy: {test_accuracy*100:.2f}%. Getting closer to 85%!")
        print("Try running for more epochs or with data augmentation for better results.")
    
except Exception as e:
    print(f"Error during training: {e}")
    print("Trying alternative approach...")
    
    # Alternative training approach
    print("Using tf.data pipeline...")
    
    # Create tf.data datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    
    # Train with tf.data
    history = model.fit(
        train_dataset,
        epochs=30,
        validation_data=test_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")