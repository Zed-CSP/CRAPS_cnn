# ***WIP***

# CRAPS_cnn
(C)onvolutional (R)ecognition and (A)nalysis for (P)robabilistic (S)coring ...a diceroll CV!

## Overview

This project is a convolutional neural network (CNN) model designed to predict the outcome of a dice roll. The model is trained on a dataset of dice roll images and is able to predict the outcome of a dice roll with high accuracy.

## Install Dependencies

```bash
pip install -r requirements.txt
```


## Usage

1. **prepare the data**:

```bash
python scripts/utils.py
```
### This will prompt you for:
 - Path to XML file (e.g., "data/raw/kaggle/roll-detection/output/rolls.xml")
 - Path to images directory (e.g., "data/raw/kaggle/roll-detection/images/")
 - Output directory (e.g., "data/processed/")


2. **Train the model** using the training script:

```bash
python scripts/train_model.py
```
### This will:
 - Load and preprocess the data
 - Train the model
 - Save the trained model to 'models/final_model.h5'
 - Create checkpoints in the 'checkpoints' directory
 - Save TensorBoard logs in the 'logs' directory

3. **Evaluate the model** using the evaluation script:

```bash
python scripts/eval_model.py
```
### This will:
 - Load the trained model
 - Evaluate on test data
 - Generate classification report
 - Create confusion matrix visualization

4. **Visualize the model** using the visualization script:

```bash
python scripts/visualize_model.py
```
### This will:
 - Create model architecture diagram
 - Visualize feature maps
 - Generate class activation maps

5. (optional) **Visualize the training process** using TensorBoard:
```bash
tensorboard --logdir logs
# Then open your browser to http://localhost:6006
```

6. **Predict the outcome of a dice roll** using the prediction script:

To use the model, you can run the `predict.py` script. This script will load the model and predict the outcome of a dice roll for a given image.

```bash
python scripts/predict.py
```

