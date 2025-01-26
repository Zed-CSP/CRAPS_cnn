import xml.etree.ElementTree as ET
import json
import os
import shutil
from math import floor

def convert_xml_to_json(xml_path, json_path):
    """
    Convert an XML file containing dice roll annotations to JSON format.

    Args:
        xml_path (str): Path to the XML file.
        json_path (str): Path to save the JSON file.
    """
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extract data
    data = []
    for roll in root.findall('roll'):
        image = roll.find('image').text
        die_one = int(roll.find('die-one').text)
        die_two = int(roll.find('die-two').text)
        data.append({
            "image": image,
            "die_one": die_one,
            "die_two": die_two
        })

    # Save to JSON
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def split_dataset(json_path, images_dir, output_dir):
    """
    Split the dataset into training, validation, and testing sets.

    Args:
        json_path (str): Path to the JSON file with annotations.
        images_dir (str): Directory containing the images.
        output_dir (str): Directory to save the split datasets (train, val, test).
    """
    # Load JSON data
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)

    # Shuffle data
    data.sort(key=lambda x: x['image'])

    # Calculate split indices
    total = len(data)
    train_end = floor(total * 0.8)
    val_end = floor(total * 0.9)

    splits = {
        "train": data[:train_end],
        "val": data[train_end:val_end],
        "test": data[val_end:]
    }

    # Create output directories
    for split in splits:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        # Save images and annotations
        for item in splits[split]:
            image_src = os.path.join(images_dir, item['image'])
            image_dst = os.path.join(split_dir, item['image'])
            shutil.copy(image_src, image_dst)

        # Save annotations
        with open(os.path.join(split_dir, 'annotations.json'), 'w') as split_json:
            json.dump(splits[split], split_json, indent=4)

if __name__ == "__main__":
    # Example usage
    xml_path = "data/raw/kaggle/rolldetection/output/rolls.xml"  # Replace with a prompted XML file path later
    json_path = "data/raw/kaggle/rolldetection/output/rolls.json"  # Replace with prompted JSON file path later
    images_dir = "data/raw/kaggle/rolldetection/input"  # replace with prompted images directory later
    output_dir = "data"  # Directory to save splits (train, val, test will be created here)

    convert_xml_to_json(xml_path, json_path)
    split_dataset(json_path, images_dir, output_dir)
    print(f"Dataset split into train, val, and test sets at {output_dir}")
