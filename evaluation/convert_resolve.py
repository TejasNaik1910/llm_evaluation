# #convert resove detections to JSON format of human, for each category, within list of "texts" and "omittedDetails", within "text" extract the word/words enclosed within {{}} and store it as "text", store "omittedDetails" as "omittedDetails" and extract word/words index of the {{}} from summary. So summary and resolve JSON path will be given. New JSON with "text", "omittedDetails", "WordIndex" should be added for each entry within each key.
# #convert resolve detections to JSON format of human, for each category, within list of "texts" and "omittedDetails". First find the "text" part within the summary, then from "text" extract the word/words enclosed within {{}} and store it as "text" and extract word/words index or word index range of the {{}} in resolve detection from summary as "WordIndex". Also store "omittedDetails" as "omittedDetails" in the new JSON as well. Remember to preprocess "text" from resolve to not contain newline characters \n \\n . summary and resolve JSON path will be given. New JSON with "text",  "WordIndex" , "omittedDetails", should be added for each entry within each key. CAN YOU GIVE ME THE PYTHON CODE TO DO THIS TASK PLEASE

# import json
# import re
# from difflib import SequenceMatcher

# def preprocess_text(text):
#     """Remove newline characters, double backslashes, and normalize spaces."""
#     return re.sub(r'\s+', ' ', text.replace('\n', ' ').replace('\\n', ' ')).strip().lower()

# def fuzzy_match(text, summary, threshold=0.8):
#     """Perform fuzzy matching to find the best match in the summary."""
#     best_ratio = 0
#     best_match_start = -1
#     text_length = len(text)
    
#     for i in range(len(summary) - text_length + 1):
#         snippet = summary[i:i + text_length]
#         ratio = SequenceMatcher(None, text, snippet).ratio()
#         if ratio > best_ratio:
#             best_ratio = ratio
#             best_match_start = i
#         if best_ratio >= threshold:
#             break  # Exit if a good match is found
    
#     return best_match_start if best_ratio >= threshold else -1

# def find_character_indices_in_summary(summary_text, resolve_text, word_within_braces):
#     """Calculate the character indices for the characters within {{}} in the given resolve text."""
#     normalized_summary = preprocess_text(summary_text)
#     normalized_resolve = preprocess_text(resolve_text)
    
#     # Try to find an exact match first
#     start_index = normalized_summary.find(normalized_resolve)
    
#     # If no exact match, use fuzzy matching
#     if start_index == -1:
#         start_index = fuzzy_match(normalized_resolve, normalized_summary)
    
#     if start_index == -1:
#         return None
    
#     indices = []
#     for match in re.finditer(r'\{\{(.*?)\}\}', word_within_braces):
#         word = match.group(1).lower()
#         word_start = normalized_summary.find(word, start_index)
#         if word_start != -1:
#             word_end = word_start + len(word)
#             indices.append((word_start, word_end))
    
#     return indices if indices else None

# def extract_words_within_braces(resolve_text):
#     """Extract words within {{}} from resolve text."""
#     matches = re.findall(r'\{\{(.*?)\}\}', resolve_text)
#     return matches if matches else None

# def process_resolve_text(summary_text, resolve_text):
#     """Process the resolve text to calculate character indices and extract words within {{}}."""
#     resolve_text_preprocessed = preprocess_text(resolve_text)
#     extracted_words = extract_words_within_braces(resolve_text)
#     results = []
#     if extracted_words:
#         character_indices = find_character_indices_in_summary(summary_text, resolve_text, resolve_text)
#         for i, word in enumerate(extracted_words):
#             indices = character_indices[i] if character_indices and i < len(character_indices) else None
#             results.append({"text": word, "CharacterIndices": indices})
#     return results

# def convert_to_json(summary_path, resolve_path, output_path):
#     # Load summary and resolve files
#     with open(summary_path, 'r') as summary_file:
#         summary_text = summary_file.read()
#     with open(resolve_path, 'r') as resolve_file:
#         resolve_data = json.load(resolve_file)
    
#     # Preprocess summary text
#     summary_text = preprocess_text(summary_text)
    
#     # Process each category in resolve data
#     output_data = {}
#     for category, entries in resolve_data.items():
#         output_data[category] = []
#         for entry in entries:
#             resolve_text = entry['text']
#             processed_data = process_resolve_text(summary_text, resolve_text)
#             for data in processed_data:
#                 output_entry = {
#                     "text": data["text"],
#                     "CharacterIndices": data["CharacterIndices"],
#                     "omittedDetails": entry.get('omittedDetails', None)
#                 }
#                 output_data[category].append(output_entry)
    
#     # Save to output JSON file
#     with open(output_path, 'w') as output_file:
#         json.dump(output_data, output_file, indent=4)

# summary_path = 'data/summaries/set2/gpt4o/gpt4o-doc-10002221-DS-11.txt'
# resolve_path = 'data/resolve/set2/gpt4o/pt-10002221-DS-11.json'
# output_path = 'data/resolve-cleaned/set2/gpt4o/10002221-DS-11.json'

# # note_ids = [
# #     "10002221-DS-11", "10004401-DS-22", "10004401-DS-29", "10094971-DS-3",
# #     "10018052-DS-17", "10024331-DS-28", "10024331-DS-29", "10024331-DS-31",
# #     "10035631-DS-13", "10094971-DS-5", "10041127-DS-17", "10041836-DS-20",
# #     "10047172-DS-15", "10047172-DS-16", "10052992-DS-17", "10054464-DS-19",
# #     "10054464-DS-20", "10056223-DS-4", "10059192-DS-10", "10060764-DS-8",
# #     "10060764-DS-9", "10070201-DS-19", "10070594-DS-14", "10070594-DS-16",
# #     "10073847-DS-30", "10074556-DS-22", "10074858-DS-16", "10076342-DS-20",
# #     "10076617-DS-11", "10076958-DS-13", "10078297-DS-5", "10078933-DS-9",
# #     "10079616-DS-8", "10079616-DS-9", "10084586-DS-19", "10085005-DS-5",
# #     "10085725-DS-12", "10089085-DS-18", "10090755-DS-7", "10090755-DS-8",
# #     "10091141-DS-20", "10095417-DS-19", "10091385-DS-16", "10091385-DS-17",
# #     "10091873-DS-22", "10093120-DS-18", "10097898-DS-11", "10098672-DS-3",
# #     "10036086-DS-25", "10098875-DS-12"
# # ]
# #summary_path = 'data/summaries/set2/gpt4o/'
# #resolve path = 'data/resolve-replace-omitted/set2/gpt4o'
# #output = 'data/resolve-cleaned/set2/gpt4o'

# # Convert and save to JSON
# convert_to_json(summary_path, resolve_path, output_path)

# print(f"Converted JSON has been saved to {output_path}")


import json
import re
from difflib import SequenceMatcher
import os

def preprocess_text(text):
    """Remove newline characters, double backslashes, and normalize spaces."""
    return re.sub(r'\s+', ' ', text.replace('\n', ' ').replace('\\n', ' ')).strip().lower()

def fuzzy_match(text, summary, threshold=0.8):
    """Perform fuzzy matching to find the best match in the summary."""
    best_ratio = 0
    best_match_start = -1
    text_length = len(text)
    
    for i in range(len(summary) - text_length + 1):
        snippet = summary[i:i + text_length]
        ratio = SequenceMatcher(None, text, snippet).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match_start = i
        if best_ratio >= threshold:
            break  # Exit if a good match is found
    
    return best_match_start if best_ratio >= threshold else -1

def find_character_indices_in_summary(summary_text, resolve_text, word_within_braces):
    """Calculate the character indices for the characters within {{}} in the given resolve text."""
    normalized_summary = preprocess_text(summary_text)
    normalized_resolve = preprocess_text(resolve_text)
    
    # Try to find an exact match first
    start_index = normalized_summary.find(normalized_resolve)
    
    # If no exact match, use fuzzy matching
    if start_index == -1:
        start_index = fuzzy_match(normalized_resolve, normalized_summary)
    
    if start_index == -1:
        return None
    
    indices = []
    for match in re.finditer(r'\{\{(.*?)\}\}', word_within_braces):
        word = match.group(1).lower()
        word_start = normalized_summary.find(word, start_index)
        if word_start != -1:
            word_end = word_start + len(word)
            indices.append((word_start, word_end))
    
    return indices if indices else None

def extract_words_within_braces(resolve_text):
    """Extract words within {{}} from resolve text."""
    matches = re.findall(r'\{\{(.*?)\}\}', resolve_text)
    return matches if matches else None

def process_resolve_text(summary_text, resolve_text):
    """Process the resolve text to calculate character indices and extract words within {{}}."""
    resolve_text_preprocessed = preprocess_text(resolve_text)
    extracted_words = extract_words_within_braces(resolve_text)
    results = []
    if extracted_words:
        character_indices = find_character_indices_in_summary(summary_text, resolve_text, resolve_text)
        for i, word in enumerate(extracted_words):
            indices = character_indices[i] if character_indices and i < len(character_indices) else None
            results.append({"text": word, "CharacterIndices": indices})
    return results

def convert_to_json(note_id, summary_dir, resolve_dir, output_dir):
    # Construct file paths
    summary_path = os.path.join(summary_dir, f'gpt4o-doc-{note_id}.txt')
    resolve_path = os.path.join(resolve_dir, f'pt-{note_id}.json')
    output_path = os.path.join(output_dir, f'{note_id}.json')

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
            resolve_text = entry['text']
            processed_data = process_resolve_text(summary_text, resolve_text)
            for data in processed_data:
                output_entry = {
                    "text": data["text"],
                    "CharacterIndices": data["CharacterIndices"],
                    "omittedDetails": entry.get('omittedDetails', None)
                }
                output_data[category].append(output_entry)
    
    # Save to output JSON file
    with open(output_path, 'w') as output_file:
        json.dump(output_data, output_file, indent=4)

def process_all_notes(note_ids, summary_dir, resolve_dir, output_dir):
    for note_id in note_ids:
        convert_to_json(note_id, summary_dir, resolve_dir, output_dir)
        print(f"Processed and saved: {note_id}")

# Directories for input and output
summary_dir = 'data/summaries/set2/gpt4o'
resolve_dir = 'data/resolve/set2/gpt4o'
output_dir = 'data/resolve-cleaned/set2/gpt4o'

# List of note_ids
note_ids = [
    "10002221-DS-11", "10004401-DS-22", "10004401-DS-29", "10094971-DS-3",
    "10018052-DS-17", "10024331-DS-28", "10024331-DS-29", "10024331-DS-31",
    "10035631-DS-13", "10094971-DS-5", "10041127-DS-17", "10041836-DS-20",
    "10047172-DS-15", "10047172-DS-16", "10052992-DS-17", "10054464-DS-19",
    "10054464-DS-20", "10056223-DS-4", "10059192-DS-10", "10060764-DS-8",
    "10060764-DS-9", "10070201-DS-19", "10070594-DS-14", "10070594-DS-16",
    "10073847-DS-30", "10074556-DS-22", "10074858-DS-16", "10076342-DS-20",
    "10076617-DS-11", "10076958-DS-13", "10078297-DS-5", "10078933-DS-9",
    "10079616-DS-8", "10079616-DS-9", "10084586-DS-19", "10085005-DS-5",
    "10085725-DS-12", "10089085-DS-18", "10090755-DS-7", "10090755-DS-8",
    "10091141-DS-20", "10095417-DS-19", "10091385-DS-16", "10091385-DS-17",
    "10091873-DS-22", "10093120-DS-18", "10097898-DS-11", "10098672-DS-3",
    "10036086-DS-25", "10098875-DS-12"
]

# Process all notes
process_all_notes(note_ids, summary_dir, resolve_dir, output_dir)


