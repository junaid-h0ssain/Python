# Updated functions for validation AFTER training (not during)

def get_callbacks(model_name):
    """
    Callbacks that monitor TRAINING metrics only (not validation).
    Since we're not validating during training, we monitor 'accuracy' and 'loss' instead.
    """
    callbacks = [
        EarlyStopping(
            monitor='accuracy',  # Changed from 'val_accuracy'
            patience=5,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='loss',  # Changed from 'val_loss'
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
            mode='min'
        ),
        ModelCheckpoint(
            filepath=f'{model_name}_best.weights.h5',
            monitor='accuracy',  # Changed from 'val_accuracy'
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
            mode='max'
        )
    ]
    return callbacks


def train_single_model(model_name, base_model_class):
    """
    Train a model WITHOUT validation during training.
    Validation is performed ONLY AFTER training is complete.
    """
    print(f"="*80)
    print(f"TRAINING {model_name}")
    print(f"="*80)
    
    # Special preprocessing for ResNet50
    if model_name == 'ResNet50':
        print(f"\nCreating datasets with ResNet50-specific preprocessing...")
        from tensorflow.keras.applications.resnet50 import preprocess_input
        
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            train,
            seed=123,
            image_size=(224, 224),
            batch_size=batch_size,
            label_mode='categorical',
            shuffle=True
        )
        
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            valid,
            seed=123,
            image_size=(224, 224),
            batch_size=batch_size,
            label_mode='categorical',
            shuffle=False
        )
        
        total_batches = train_ds.cardinality().numpy()
        train_ds = train_ds.shuffle(buffer_size=100, reshuffle_each_iteration=True)
        
        portion = 0.5
        train_ds = train_ds.take(int(total_batches * portion))
        
        print(f"Using {int(total_batches * portion)} batches for training")
        
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomFlip("vertical"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomContrast(0.2),
        ])
        
        train_ds = train_ds.map(
            lambda x, y: (preprocess_input(data_augmentation(x, training=True)), y),
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)
        
        val_ds = val_ds.map(
            lambda x, y: (preprocess_input(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)
        
        train_dataset_model = train_ds
        val_dataset_model = val_ds
    else:
        train_dataset_model = train_dataset
        val_dataset_model = val_dataset

    print(f"\nCreating {model_name} model...")
    model, base_model = create_model(base_model_class, model_name, num_classes)
    print(f"✓ {model_name} model created")

    # ========================================
    # STAGE 1: Initial training (NO validation)
    # ========================================
    print(f"\n[STAGE 1] Initial training with frozen base model...")
    print("NOTE: No validation during training - validation will run after training completes")
    initial_epochs = 20

    history_stage1 = model.fit(
        train_dataset_model,
        # validation_data=val_dataset_model,  # ← REMOVED
        epochs=initial_epochs,
        callbacks=get_callbacks(f"{model_name}_stage1")
    )
    
    # Validate AFTER Stage 1 training
    print(f"\n[VALIDATION AFTER STAGE 1]")
    val_loss_s1, val_acc_s1 = model.evaluate(val_dataset_model, verbose=1)
    print(f"Stage 1 Validation - Loss: {val_loss_s1:.4f}, Accuracy: {val_acc_s1*100:.2f}%")

    # ========================================
    # STAGE 2: Fine-tuning (NO validation)
    # ========================================
    print(f"\n[STAGE 2] Fine-tuning with unfrozen top layers...")
    print("NOTE: No validation during training - validation will run after training completes")
    model = fine_tune_model(model, base_model, num_layers_to_unfreeze=30)
    fine_tune_epochs = 20
    
    history_stage2 = model.fit(
        train_dataset_model,
        # validation_data=val_dataset_model,  # ← REMOVED
        epochs=fine_tune_epochs,
        callbacks=get_callbacks(f"{model_name}_stage2")
    )
    
    # Validate AFTER Stage 2 training
    print(f"\n[VALIDATION AFTER STAGE 2]")
    val_loss_s2, val_acc_s2 = model.evaluate(val_dataset_model, verbose=1)
    print(f"Stage 2 Validation - Loss: {val_loss_s2:.4f}, Accuracy: {val_acc_s2*100:.2f}%")

    # Save model
    model.save(f"{model_name}_plant_disease_model.h5")
    print(f"\n✓ {model_name} model saved")

    # Generate predictions for final evaluation
    print(f"\nGenerating predictions for {model_name}...")
    y_pred = model.predict(val_dataset_model)

    y_true = []
    for image_batch, label_batch in val_dataset_model:
        y_true.append(label_batch)
    y_true = tf.concat(y_true, axis=0).numpy()

    np.save(f"{model_name}_y_pred.npy", y_pred)
    np.save(f"{model_name}_y_true.npy", y_true)
    print(f"✓ {model_name} predictions saved")

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


print("✓ Updated functions defined (validation AFTER training)")
