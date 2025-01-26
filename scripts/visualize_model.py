import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import plot_model
import numpy as np
from tensorflow.keras.models import load_model
import cv2
from tensorflow.keras import Model
import os

def plot_model_architecture(model, output_path='visualizations/model_architecture.png'):
    """Plot the model architecture"""
    plot_model(model, 
              to_file=output_path, 
              show_shapes=True, 
              show_layer_names=True,
              show_layer_activations=True)

def plot_training_history(history, output_path='visualizations/training_history.png'):
    """Plot training and validation metrics"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def visualize_feature_maps(model, image_path, output_path='visualizations/feature_maps/'):
    """Visualize feature maps for each convolutional layer"""
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(1024, 1024))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Get outputs of all conv layers
    layer_outputs = []
    layer_names = []
    
    for layer in model.layers:
        if 'conv2d' in layer.name:
            intermediate_model = Model(inputs=model.input, outputs=layer.output)
            layer_outputs.append(intermediate_model.predict(img_array))
            layer_names.append(layer.name)

    # Plot feature maps
    for layer_name, feature_maps in zip(layer_names, layer_outputs):
        n_features = min(8, feature_maps.shape[-1])  # Display first 8 features
        
        plt.figure(figsize=(12, 8))
        for i in range(n_features):
            plt.subplot(2, 4, i + 1)
            plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
            plt.axis('off')
        
        plt.suptitle(f'Feature Maps - {layer_name}')
        plt.savefig(f'{output_path}{layer_name}_features.png')
        plt.close()

def visualize_class_activation_maps(model, image_path, output_path='visualizations/cam/'):
    """Generate Class Activation Maps"""
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(1024, 1024))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Get the score for the most likely class
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    
    # Get the output of the last conv layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break
    
    grad_model = Model([model.inputs], 
                      [last_conv_layer.output, model.output])
    
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_output = conv_output[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    
    # Overlay heatmap on original image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (1024, 1024))
    heatmap = cv2.resize(heatmap, (1024, 1024))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(f'{output_path}cam_output.png', superimposed_img)

if __name__ == "__main__":
    # Load model and sample image
    model = load_model('models/final_model.h5')
    sample_image = 'data/processed/test/0.png'  # Adjust path as needed
    
    # Create visualization directories
    for dir_path in ['visualizations/feature_maps/', 'visualizations/cam/']:
        os.makedirs(dir_path, exist_ok=True)
    
    # Generate all visualizations
    plot_model_architecture(model)
    visualize_feature_maps(model, sample_image)
    visualize_class_activation_maps(model, sample_image) 