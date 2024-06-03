import os
import json

def process_json_files(input_path, output_path):
    # Ensure the output path exists
    os.makedirs(output_path, exist_ok=True)

    # Iterate over all files in the input directory
    for filename in os.listdir(input_path):
        if filename.endswith('.json'):
            input_file_path = os.path.join(input_path, filename)
            output_file_path = os.path.join(output_path, filename)

            with open(input_file_path, 'r') as infile:
                data = json.load(infile)

            # Process each key in the JSON
            for key, value in data.items():
                if isinstance(value, list):
                    # Filter out entries with text length of 1
                    filtered_entries = [entry for entry in value if len(entry['text']) > 2 or not all(char in "*:,. " for char in entry['text'])]


                    # Update the main list with filtered entries
                    data[key] = filtered_entries

                    # Check for associated omitted details key and filter it
                    omitted_key = key.replace("Incorrect", "Omitted")
                    if omitted_key in data and isinstance(data[omitted_key], list):
                        # Filter out omitted details if the text is removed from the main key
                        data[omitted_key] = [
                            detail for detail in data[omitted_key] if len(detail['text']) > 1
                        ]

            # Write the cleaned data to the output file
            with open(output_file_path, 'w') as outfile:
                json.dump(data, outfile, indent=4)

# Example usage:
input_path = 'data/processed-human-annotations'
output_path = 'data/cleaned_annotations'
process_json_files(input_path, output_path)
