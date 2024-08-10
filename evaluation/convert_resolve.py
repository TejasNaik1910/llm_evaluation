#convert resove detections to JSON format of human, for each category, within list of "texts" and "omittedDetails", within "text" extract the word/words enclosed within {{}} and store it as "text", store "omittedDetails" as "omittedDetails" and extract word/words index of the {{}} from summary. So summary and resolve JSON path will be given. New JSON with "text", "omittedDetails", "WordIndex" should be added for each entry within each key.
#convert resolve detections to JSON format of human, for each category, within list of "texts" and "omittedDetails". First find the "text" part within the summary, then from "text" extract the word/words enclosed within {{}} and store it as "text" and extract word/words index or word index range of the {{}} in resolve detection from summary as "WordIndex". Also store "omittedDetails" as "omittedDetails" in the new JSON as well. Remember to preprocess "text" from resolve to not contain newline characters \n \\n . summary and resolve JSON path will be given. New JSON with "text",  "WordIndex" , "omittedDetails", should be added for each entry within each key. CAN YOU GIVE ME THE PYTHON CODE TO DO THIS TASK PLEASE

import json
import re

def preprocess_text(text):
    """Remove newline characters and double backslashes."""
    return text.replace('\n', ' ').replace('\\n', ' ')

def find_word_index(summary_text, word):
    """Find the starting word index of a specific word in the summary text."""
    words = summary_text.split()
    word_positions = [i for i, w in enumerate(words) if word in w]
    return word_positions[0] if word_positions else None

def extract_words_and_indices(summary_text, highlighted_text):
    """Extract the words and their word index or index range from text enclosed within {{}}."""
    results = []
    for match in re.finditer(r'\{\{(.*?)\}\}', highlighted_text):
        word = match.group(1)
        index = find_word_index(summary_text, word)
        if index is not None:
            results.append({"text": word, "WordIndex": index})
    return results

def convert_to_json(summary_path, resolve_path, output_path):
    # Load summary and resolve files
    with open(summary_path, 'r') as summary_file:
        summary_text = summary_file.read()
    with open(resolve_path, 'r') as resolve_file:
        resolve_data = json.load(resolve_file)
    
    # Preprocess summary text
    summary_text = preprocess_text(summary_text)
    
    # Process each category in resolve data
    output_data = {}
    for category, entries in resolve_data.items():
        output_data[category] = []
        for entry in entries:
            extracted_data = extract_words_and_indices(summary_text, preprocess_text(entry['text']))
            for data in extracted_data:
                output_entry = {
                    "text": data["text"],
                    "WordIndex": data["WordIndex"],
                    "omittedDetails": entry.get('omittedDetails', None)
                }
                output_data[category].append(output_entry)
    
    # Save to output JSON file
    with open(output_path, 'w') as output_file:
        json.dump(output_data, output_file, indent=4)

# Paths to the input and output files
summary_path = 'data/summaries/set2/gpt4o/gpt4o-doc-10002221-DS-11.txt'
resolve_path = 'data/resolve/gpt4o/set2/pt-10002221-DS-11.json'
output_path = 'data/resolve-cleaned/gpt4o/10002221-DS-11.json'

# Convert and save to JSON
convert_to_json(summary_path, resolve_path, output_path)

print(f"Converted JSON has been saved to {output_path}")
