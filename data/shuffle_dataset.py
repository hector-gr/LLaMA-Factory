import json
import random
import argparse
import shutil
def shuffle_dataset(file_path: str) -> None:
    """
    Loads a JSON file containing a list of objects, shuffles the order,
    and saves it back to the same file.
    
    Args:
        file_path (str): Path to the JSON file
    """
    new_file_path = file_path.replace('.json', '_copy.json')
    shutil.copy(file_path, new_file_path)

    # Read the JSON file
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        
    # Verify that the data is a list
    if not isinstance(data, list):
        raise ValueError("The JSON file must contain a list of objects")
        
    # Shuffle the data
    random.shuffle(data)
    
    # Write back to the same file
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    # try:
    #     # create a copy of the file with a new name
    #     new_file_path = file_path.replace('.json', '_copy.json')
    #     shutil.copy(file_path, new_file_path)

    #     # Read the JSON file
    #     with open(file_path, 'r', encoding='utf-8') as file:
    #         data = json.load(file)
            
    #     # Verify that the data is a list
    #     if not isinstance(data, list):
    #         raise ValueError("The JSON file must contain a list of objects")
            
    #     # Shuffle the data
    #     random.shuffle(data)
        
    #     # Write back to the same file
    #     with open(file_path, 'w', encoding='utf-8') as file:
    #         json.dump(data, file, indent=2, ensure_ascii=False)
            
    # except json.JSONDecodeError:
    #     raise ValueError("Invalid JSON file")
    # except Exception as e:
    #     raise Exception(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Shuffle a JSON dataset file.')
    parser.add_argument('-p', '--path', required=True, help='Path to the JSON dataset file')
    
    args = parser.parse_args()
    shuffle_dataset(args.path) 