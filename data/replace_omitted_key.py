import os
import json

def replace_key_prefix_in_json_in_folder(input_folder, key_prefix_to_replace, new_prefix, output_folder):
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
            
            # Replace keys that start with the specified prefix
            new_data = {}
            for key in data:
                if key.startswith(key_prefix_to_replace):
                    new_key = key.replace(key_prefix_to_replace, new_prefix, 1)
                    new_data[new_key] = data[key]
                    print(f"Key '{key}' changed to '{new_key}' in file '{filename}'.")
                else:
                    new_data[key] = data[key]
            
            # Save the modified JSON to the output folder with the same filename
            with open(output_file_path, 'w') as file:
                json.dump(new_data, file, indent=4)
            print(f"Modified JSON saved to '{output_file_path}'.")

# Example usage
input_folder = 'data/human-annotations/set2/llama3_removeOI'  # Replace with your input folder path
key_prefix_to_replace = 'Omitted'
new_prefix = 'Specific to General'
output_folder = 'data/human-annotations/set2/llama3_cleaned'  # Replace with your desired output folder path

replace_key_prefix_in_json_in_folder(input_folder, key_prefix_to_replace, new_prefix, output_folder)
