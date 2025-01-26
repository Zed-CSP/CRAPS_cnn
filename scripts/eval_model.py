import tensorflow as tf
import numpy as np
import json
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model_path, test_dir):
    """
    Evaluate the model on the test set.
    """
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Load test data
    test_dataset = load_dataset(test_dir)
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Get predictions
    y_pred = []
    y_true = []
    
    for images, labels in test_dataset:
        predictions = model.predict(images)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(labels.numpy())
    
    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('evaluation/confusion_matrix.png')
    plt.close()

if __name__ == "__main__":
    MODEL_PATH = "models/final_model.h5"
    TEST_DIR = "data/processed/test"
    
    evaluate_model(MODEL_PATH, TEST_DIR)
