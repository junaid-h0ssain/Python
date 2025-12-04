"""
Fixed train_single_model function for ResNet50
Copy this into your notebook Cell 6 to replace the existing train_single_model function
"""

def train_single_model(model_name, base_model_class):
    """
    Train a single model with two-stage training.
    Includes ResNet50-specific preprocessing fix.
    """
    print(f"\n{'='*80}")
    print(f"TRAINING {model_name}")
    print(f"{'='*80}")
    
    # CRITICAL FIX: Use model-specific preprocessing for ResNet50
    if model_name == 'ResNet50':
        print(f"\nCreating datasets with ResNet50-specific preprocessing...")
        from tensorflow.keras.applications.resnet50 import preprocess_input
        
        # Recreate datasets with proper preprocessing
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            train,
            seed=123,
            image_size=(224, 224),  # ResNet50 optimal size
            batch_size=batch_size,
            label_mode='categorical',
            shuffle=True
        )
        
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            valid,
            seed=123,
            image_size=(224, 224),  # ResNet50 optimal size
            batch_size=batch_size,
            label_mode='categorical',
            shuffle=False
        )
        
        total_batches = train_ds.cardinality().numpy()
        train_ds = train_ds.shuffle(buffer_size=100, reshuffle_each_iteration=True)
        
        # CRITICAL FIX: Use more data (20% instead of 1%)
        portion = 0.20  # Use 20% of data for better learning
        train_ds = train_ds.take(int(total_batches * portion))
        
        print(f"Using {int(total_batches * portion)} batches for training")
        
        # Data augmentation for training
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomFlip("vertical"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomContrast(0.2),
        ])
        
        # CRITICAL FIX: Apply ResNet-specific preprocessing
        train_ds = train_ds.map(
            lambda x, y: (preprocess_input(data_augmentation(x, training=True)), y),
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)
        
        val_ds = val_ds.map(
            lambda x, y: (preprocess_input(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)
        
        # Use these datasets for ResNet50
        train_dataset_model = train_ds
        val_dataset_model = val_ds
    else:
        # Use global datasets for other models
        train_dataset_model = train_dataset
        val_dataset_model = val_dataset

    print(f"\nCreating {model_name} model...")
    model, base_model = create_model(base_model_class, model_name, num_classes)
    print(f"✓ {model_name} model created")

    print(f"\n[STAGE 1] Initial training with frozen base model...")
    initial_epochs = 15  # INCREASED from 10

    history_stage1 = model.fit(
        train_dataset_model,
        validation_data=val_dataset_model,
        epochs=initial_epochs,
        callbacks=get_callbacks(f"{model_name}_stage1")
    )

    print(f"\n[STAGE 2] Fine-tuning with unfrozen top layers...")
    model = fine_tune_model(model, base_model, num_layers_to_unfreeze=30)  # INCREASED from 20
    fine_tune_epochs = 15  # INCREASED from 10

    history_stage2 = model.fit(
        train_dataset_model,
        validation_data=val_dataset_model,
        epochs=fine_tune_epochs,
        callbacks=get_callbacks(f"{model_name}_stage2")
    )

    model.save(f"{model_name}_plant_disease_model.h5")
    print(f"\n✓ {model_name} model saved")

    print(f"\nGenerating predictions for {model_name}...")
    y_pred = model.predict(val_dataset_model)

    y_true = []
    for image_batch, label_batch in val_dataset_model:
        y_true.append(label_batch)
    y_true = tf.concat(y_true, axis=0).numpy()

    np.save(f"{model_name}_y_pred.npy", y_pred)
    np.save(f"{model_name}_y_true.npy", y_true)
    print(f"✓ {model_name} predictions saved")

    # Calculate accuracy
    y_pred_labels = y_pred.argmax(axis=1)
    y_true_labels = y_true.argmax(axis=1)
    accuracy = np.mean(y_pred_labels == y_true_labels)

    print(f"\n{model_name} FINAL VALIDATION ACCURACY: {accuracy*100:.2f}%")

    if accuracy >= 0.985:
        print(f"✓ {model_name} achieved >=98.5% accuracy!")
    else:
        print(f"✗ {model_name} did not reach 98.5% accuracy (got {accuracy*100:.2f}%)")

    # Clear memory before next model
    del model, base_model
    if model_name == 'ResNet50':
        del train_ds, val_ds, train_dataset_model, val_dataset_model
    K.clear_session()
    gc.collect()

    return accuracy

print("✓ Fixed helper function defined")
