import json
import os

# Define the input and output folder paths
input_folder = 'data/human-annotations'
output_folder = 'data/processed-human-annotations'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Get the list of files in the input folder
input_files = os.listdir(input_folder)

# Define the target structure with initial empty lists
target_structure = {
    "Incorrect Patient Information": [],
    "Omitted Patient Information": [],
    "Incorrect Patient History": [],
    "Omitted Patient History": [],
    "Incorrect Symptoms/Diagnosis": [],
    "Omitted Symptoms/Diagnosis": [],
    "Incorrect Medicinal Instructions": [],
    "Omitted Medicinal Instructions": [],
    "Incorrect Followup": [],
    "Omitted Followup": [],
    "Incorrect Other Inconsistency": [],
    "Omitted Other Inconsistency": [],
    "Incorrect Reasoning": [],
    "Chronological Inconsistency": []
}

# Function to transform data
def transform_data(data, target_structure):
    transformed = target_structure.copy()
    for key, value in data.items():
        if key in transformed:
            if "Omitted" in key:
                for item in value:
                    transformed[key].append({"text": item["text"], "omittedDetails": item["omittedDetails"]})
            else:
                for item in value:
                    transformed[key].append({"text": item["text"]})
    return transformed

# Process each file in the input folder
for input_file in input_files:
    # Construct the full input file path
    input_file_path = os.path.join(input_folder, input_file)
    
    # Read the input JSON data
    with open(input_file_path, 'r') as file:
        data = json.load(file)
    
    # Transform the input data
    transformed_data = transform_data(data, target_structure)
    
    # Construct the full output file path
    output_file_path = os.path.join(output_folder, input_file)
    
    # Write the transformed data to the output JSON file
    with open(output_file_path, 'w') as file:
        json.dump(transformed_data, file, indent=2)

    print(f"Processed {input_file} and saved to {output_file_path}")
