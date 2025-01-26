import tensorflow as tf
from tensorflow.keras import layers, models

def create_cnn_model(input_shape=(1024, 1024, 3), num_classes=36):
    """
    Creates a CNN model for reading d6 dice rolls.

    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of output classes (6 faces x 2 dice).

    Returns:
        tf.keras.Model: Compiled CNN model.
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    # CNN layers
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    
    # Flatten and dense layers
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    # Example: Create and summarize the model
    model = create_cnn_model()
    model.summary()