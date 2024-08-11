import os
import json

def delete_key_from_json_in_folder(input_folder, key_to_delete, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Traverse all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            input_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename)
            
            # Read the JSON data from the file
            with open(input_file_path, 'r') as file:
                data = json.load(file)
            
            # Delete the specified key if it exists
            if key_to_delete in data:
                del data[key_to_delete]
                print(f"Key '{key_to_delete}' deleted in file '{filename}'.")
            else:
                print(f"Key '{key_to_delete}' not found in file '{filename}'.")
            
            # Save the modified JSON to the output folder with the same filename
            with open(output_file_path, 'w') as file:
                json.dump(data, file, indent=4)
            print(f"Modified JSON saved to '{output_file_path}'.")

# Example usage
input_folder = 'data/resolve/set2/gpt4o'  # Replace with your input folder path
key_to_delete = 'Omitted Other Inconsistency'
output_folder = 'data/resolve-removeOI/set2/gpt4o'  # Replace with your desired output folder path

delete_key_from_json_in_folder(input_folder, key_to_delete, output_folder)
