import sys
import os
# Project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import json
import numpy as np
from models.cnn_model import create_cnn_model


NUM_CLASSES = 36  # 6x6 possible dice combinations
IMAGE_SIZE = (1024, 1024)
TRAIN_DIR = "data/processed/train"
VAL_DIR = "data/processed/val"
EPOCHS = 50
BATCH_SIZE = 8  # Smaller batch size since images are larger

def load_dataset(data_dir, batch_size=8):
    """
    Load and preprocess the dataset from the given directory.
    Memory-efficient version with full resolution.
    """
    annotations_file = os.path.join(data_dir, 'annotations.json')
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    # Create lists of file paths and labels instead of loading all images
    image_paths = []
    labels = []
    for item in annotations:
        image_paths.append(os.path.join(data_dir, item['image']))
        die_one = max(1, min(6, item['die_one'])) - 1
        die_two = max(1, min(6, item['die_two'])) - 1
        label = die_one * 6 + die_two
        labels.append(label)

    def load_and_preprocess(path, label):
        # Load and preprocess single image
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, IMAGE_SIZE, method='bilinear')
        img = tf.cast(img, tf.float32) / 255.0
        return img, tf.one_hot(label, NUM_CLASSES)

    # Create dataset from paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_and_preprocess, 
                        num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1000).batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def train_model(train_dir, val_dir, epochs=50, batch_size=32):
    """
    Train the CNN model.
    """
    # Load datasets
    train_dataset = load_dataset(train_dir, batch_size)
    val_dataset = load_dataset(val_dir, batch_size)
    
    # Create model
    model = create_cnn_model()
    
    # Add callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            'checkpoints/model_{epoch:02d}.h5',
            save_best_only=True
        ),
        tf.keras.callbacks.TensorBoard(log_dir='logs')
    ]
    
    # Train model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return model, history

if __name__ == "__main__":
    # Training configuration
    TRAIN_DIR = "data/processed/train"
    VAL_DIR = "data/processed/val"
    EPOCHS = 50
    BATCH_SIZE = 8
    
    # Train the model
    model, history = train_model(TRAIN_DIR, VAL_DIR, EPOCHS, BATCH_SIZE)
    
    # Save the final model
    model.save('models/final_model.h5')
