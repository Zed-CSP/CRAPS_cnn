import sys
import os
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import json
import numpy as np
from models.cnn_model import create_cnn_model

# Add these constants at the top of the file
NUM_CLASSES = 36  # 6x6 possible dice combinations
IMAGE_SIZE = (1024, 1024)
TRAIN_DIR = "data/processed/train"
VAL_DIR = "data/processed/val"
EPOCHS = 50
BATCH_SIZE = 32

def load_dataset(data_dir, batch_size=32):
    """
    Load and preprocess the dataset from the given directory.
    """
    annotations_file = os.path.join(data_dir, 'annotations.json')
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    images = []
    labels = []
    for item in annotations:
        img_path = os.path.join(data_dir, item['image'])
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        images.append(img_array)
        
        # Create label (0-35 for all possible combinations)
        die_one = max(1, min(6, item['die_one'])) - 1
        die_two = max(1, min(6, item['die_two'])) - 1
        label = die_one * 6 + die_two
        labels.append(label)
    
    # Convert to numpy arrays
    images = np.array(images, dtype='float32')
    labels = np.array(labels, dtype='int32')
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(lambda x, y: (x / 255.0, tf.one_hot(y, NUM_CLASSES)))
    dataset = dataset.shuffle(1000).batch(batch_size)
    
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
    # Train the model
    model, history = train_model(TRAIN_DIR, VAL_DIR, EPOCHS, BATCH_SIZE)
    
    # Save the final model
    model.save('models/final_model.h5')
