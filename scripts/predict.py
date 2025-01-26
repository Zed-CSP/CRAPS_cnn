import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import cv2

def predict_dice_roll(model_path, image_path):
    """
    Predict dice values for a single image.
    """
    # Load model
    model = load_model(model_path)
    
    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(1024, 1024))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Get prediction
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction[0])
    
    # Convert class index back to dice values
    die_one = (class_idx // 6) + 1
    die_two = (class_idx % 6) + 1
    
    confidence = prediction[0][class_idx]
    
    return die_one, die_two, confidence

if __name__ == "__main__":
    MODEL_PATH = "models/final_model.h5"
    
    # Test multiple images
    test_images = [
        "data/processed/test/0.png",
        "data/processed/test/1.png",
        # Add more test images as needed
    ]
    
    for image_path in test_images:
        die_one, die_two, confidence = predict_dice_roll(MODEL_PATH, image_path)
        print(f"\nImage: {image_path}")
        print(f"Predicted Roll: {die_one}, {die_two}")
        print(f"Confidence: {confidence:.2%}") 