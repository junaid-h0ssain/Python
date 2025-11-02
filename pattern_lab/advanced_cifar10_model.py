import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Add, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Load and preprocess CIFAR-10 data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to [0, 1]
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Convert labels to categorical
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

def residual_block(x, filters, kernel_size=3, stride=1):
    """Create a residual block"""
    shortcut = x
    
    # First conv layer
    x = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # Second conv layer
    x = Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    
    # Adjust shortcut if needed
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    # Add shortcut
    x = Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    
    return x

def create_advanced_cnn():
    """
    Create an advanced CNN with residual connections for CIFAR-10
    This should achieve >90% test accuracy
    """
    inputs = Input(shape=(32, 32, 3))
    
    # Initial conv layer
    x = Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Residual blocks
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)
    
    x = residual_block(x, 512, stride=2)
    x = residual_block(x, 512)
    
    # Global average pooling and final layers
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(10, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model

def create_efficient_cnn():
    """
    Create a more efficient CNN that balances performance and training time
    """
    model = tf.keras.Sequential([
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
        
        # Block 4
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(0.25),
        
        # Classifier
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    
    return model

# Create the model (choose one)
print("Creating efficient CNN model...")
model = create_efficient_cnn()
model.summary()

# Compile with optimized settings
model.compile(
    optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Enhanced data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=0.15,
    shear_range=0.1,
    fill_mode='nearest'
)

# Advanced callbacks
callbacks = [
    ReduceLROnPlateau(
        monitor='val_accuracy', 
        factor=0.5, 
        patience=5, 
        min_lr=0.00001,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy', 
        patience=15, 
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'best_cifar10_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Train the model
print("Starting training with data augmentation...")
history = model.fit(
    datagen.flow(train_images, train_labels, batch_size=64),
    steps_per_epoch=len(train_images) // 64,
    epochs=150,
    validation_data=(test_images, test_labels),
    callbacks=callbacks,
    verbose=1
)

# Load the best model and evaluate
model.load_weights('best_cifar10_model.h5')
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)

print(f"\nFinal Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

if test_accuracy > 0.85:
    print("🎉 SUCCESS! Achieved >85% test accuracy!")
    if test_accuracy > 0.90:
        print("🚀 EXCELLENT! Achieved >90% test accuracy!")
else:
    print(f"Current accuracy: {test_accuracy*100:.2f}%. Need to reach 85%+")

print("Best model saved as 'best_cifar10_model.h5'")