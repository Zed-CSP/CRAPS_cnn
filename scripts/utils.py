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

def prompt_user_for_paths():
    """
    Prompt the user for input, output, and image directory paths.
    Returns:
        dict: A dictionary containing the paths.
    """
    xml_path = input("Enter the path to the XML file: ")
    images_dir = input("Enter the path to the images directory: ")
    output_dir = input("Enter the path to save the split datasets: ")
    
    return {
        "xml_path": xml_path,
        "images_dir": images_dir,
        "output_dir": output_dir
    }

if __name__ == "__main__":
    # Prompt user for paths
    paths = prompt_user_for_paths()
    json_path = paths["xml_path"].replace(".xml", ".json")

    # Process dataset
    convert_xml_to_json(paths["xml_path"], json_path)
    split_dataset(json_path, paths["images_dir"], paths["output_dir"])
    print(f"Dataset split into train, val, and test sets at {paths['output_dir']}")
