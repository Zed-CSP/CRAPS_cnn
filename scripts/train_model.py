import sys
import os
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import json
from models.cnn_model import create_cnn_model

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
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(1024, 1024))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        images.append(img_array)
        
        # Combine dice values into single label 
        label = (item['die_one'] - 1) * 6 + (item['die_two'] - 1)
        labels.append(label)
    
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(lambda x, y: (x / 255.0, y))  # Normalize images
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
    # Training configuration
    TRAIN_DIR = "data/processed/train"
    VAL_DIR = "data/processed/val"
    EPOCHS = 50
    BATCH_SIZE = 32
    
    # Train the model
    model, history = train_model(TRAIN_DIR, VAL_DIR, EPOCHS, BATCH_SIZE)
    
    # Save the final model
    model.save('models/final_model.h5')
