import xml.etree.ElementTree as ET
import json

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
