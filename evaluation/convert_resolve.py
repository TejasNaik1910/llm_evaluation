# #convert resove detections to JSON format of human, for each category, within list of "texts" and "omittedDetails", within "text" extract the word/words enclosed within {{}} and store it as "text", store "omittedDetails" as "omittedDetails" and extract word/words index of the {{}} from summary. So summary and resolve JSON path will be given. New JSON with "text", "omittedDetails", "WordIndex" should be added for each entry within each key.
# #convert resolve detections to JSON format of human, for each category, within list of "texts" and "omittedDetails". First find the "text" part within the summary, then from "text" extract the word/words enclosed within {{}} and store it as "text" and extract word/words index or word index range of the {{}} in resolve detection from summary as "WordIndex". Also store "omittedDetails" as "omittedDetails" in the new JSON as well. Remember to preprocess "text" from resolve to not contain newline characters \n \\n . summary and resolve JSON path will be given. New JSON with "text",  "WordIndex" , "omittedDetails", should be added for each entry within each key. CAN YOU GIVE ME THE PYTHON CODE TO DO THIS TASK PLEASE

# INDICES EXTRACTED CORRECTLY IF ONLY ON OCCURANCE OF WORD IN SUMMARY
# import json
# import re

# def preprocess_text(text):
#     """Remove newline characters and double backslashes, and skip the first word."""
#     # Remove newline characters and double backslashes
#     processed_text = text.replace('\n', '').replace('\\n', '')  
#     # Find the first whitespace character and skip the first word
#     first_space_index = processed_text.find(' ')
#     if first_space_index != -1:
#         processed_text = processed_text[first_space_index + 1:]
#     return processed_text

# def find_word_index(summary_text, word):
#     """Find the starting word index of a specific word in the summary text."""
#     words = summary_text.split()
#     word_positions = [i for i, w in enumerate(words) if word in w]
#     return word_positions[0] if word_positions else None

# def extract_words_and_indices(summary_text, highlighted_text):
#     """Extract the words and their word index or index range from text enclosed within {{}}."""
#     results = []
#     for match in re.finditer(r'\{\{(.*?)\}\}', highlighted_text):
#         word = match.group(1)
#         index = find_word_index(summary_text, word)
#         if index is not None:
#             results.append({"text": word, "WordIndex": index})
#     return results

# def convert_to_json(summary_path, resolve_path, output_path):
#     # Load summary and resolve files
#     with open(summary_path, 'r') as summary_file:
#         summary_text = summary_file.read()
#     with open(resolve_path, 'r') as resolve_file:
#         resolve_data = json.load(resolve_file)
    
#     # Preprocess summary text
#     summary_text = preprocess_text(summary_text)      #skipping preprocess
    
#     # Process each category in resolve data
#     output_data = {}
#     for category, entries in resolve_data.items():
#         output_data[category] = []
#         for entry in entries:
#             extracted_data = extract_words_and_indices(summary_text, preprocess_text(entry['text']))
#             for data in extracted_data:
#                 output_entry = {
#                     "text": data["text"],
#                     "WordIndex": data["WordIndex"],
#                     "omittedDetails": entry.get('omittedDetails', None)
#                 }
#                 output_data[category].append(output_entry)
    
#     # Save to output JSON file
#     with open(output_path, 'w') as output_file:
#         json.dump(output_data, output_file, indent=4)

# # Paths to the input and output files
# summary_path = 'data/summaries/set2/gpt4o/gpt4o-doc-10002221-DS-11.txt'
# resolve_path = 'data/resolve/gpt4o/set2/pt-10002221-DS-11.json'
# output_path = 'data/resolve-cleaned/gpt4o/10002221-DS-11.json'

# # Convert and save to JSON
# convert_to_json(summary_path, resolve_path, output_path)

# print(f"Converted JSON has been saved to {output_path}")


#ALL {{}} EXTRACTED
# import json
# import re
# from difflib import SequenceMatcher

# def preprocess_text(text):
#     """Remove newline characters, double backslashes, and normalize spaces."""
#     return re.sub(r'\s+', ' ', text.replace('\n', ' ').replace('\\n', ' ')).strip().lower()

# def fuzzy_match(text, summary, threshold=0.8):
#     """Perform fuzzy matching to find the best match in the summary."""
#     match = None
#     best_ratio = 0
#     words = summary.split()
#     snippet_length = len(text.split())
    
#     for i in range(len(words) - snippet_length + 1):
#         snippet = ' '.join(words[i:i + snippet_length])
#         ratio = SequenceMatcher(None, text, snippet).ratio()
#         if ratio > best_ratio:
#             best_ratio = ratio
#             match = snippet
#         if best_ratio >= threshold:
#             break  # Exit if a good match is found
    
#     return match if best_ratio >= threshold else None

# def find_word_indices_in_summary(summary_text, resolve_text, word_within_braces):
#     """Calculate the word indices for the word(s) within {{}} in the given resolve text."""
#     normalized_summary = preprocess_text(summary_text)
#     normalized_resolve = preprocess_text(resolve_text)
    
#     # Try to find an exact match first
#     start_index = normalized_summary.find(normalized_resolve)
    
#     # If no exact match, use fuzzy matching
#     if start_index == -1:
#         match = fuzzy_match(normalized_resolve, normalized_summary)
#         if match:
#             start_index = normalized_summary.find(match)
    
#     if start_index == -1:
#         return None
    
#     # Calculate word indices relative to the summary text
#     summary_words = normalized_summary.split()
#     indices = []
#     for match in re.finditer(r'\{\{(.*?)\}\}', word_within_braces):
#         word = match.group(1).lower()
#         for i, summary_word in enumerate(summary_words):
#             if word == summary_word:
#                 indices.append(i)
#                 break
    
#     if len(indices) == 1:
#         return indices[0]
#     elif len(indices) > 1:
#         return f"{indices[0]}-{indices[-1]}"
#     else:
#         return None

# def extract_words_within_braces(resolve_text):
#     """Extract words within {{}} from resolve text."""
#     matches = re.findall(r'\{\{(.*?)\}\}', resolve_text)
#     return matches if matches else None

# def process_resolve_text(summary_text, resolve_text):
#     """Process the resolve text to calculate word indices and extract words within {{}}."""
#     resolve_text_preprocessed = preprocess_text(resolve_text)
#     extracted_words = extract_words_within_braces(resolve_text)
#     results = []
#     if extracted_words:
#         word_indices = find_word_indices_in_summary(summary_text, resolve_text_preprocessed, resolve_text)
#         for word in extracted_words:
#             results.append({"text": word, "WordIndex": word_indices})
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
#                     "WordIndex": data["WordIndex"],
#                     "omittedDetails": entry.get('omittedDetails', None)
#                 }
#                 output_data[category].append(output_entry)
    
#     # Save to output JSON file
#     with open(output_path, 'w') as output_file:
#         json.dump(output_data, output_file, indent=4)

# # Paths to the input and output files
# summary_path = 'data/summaries/set2/gpt4o/gpt4o-doc-10002221-DS-11.txt'
# resolve_path = 'data/resolve/gpt4o/set2/pt-10002221-DS-11.json'
# output_path = 'data/resolve-cleaned/gpt4o/10002221-DS-11_new.json'

# # Convert and save to JSON
# convert_to_json(summary_path, resolve_path, output_path)

# print(f"Converted JSON has been saved to {output_path}")

import json
import re
from difflib import SequenceMatcher

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

# Paths to the input and output files
summary_path = 'data/summaries/set2/gpt4o/gpt4o-doc-10002221-DS-11.txt'
resolve_path = 'data/resolve/gpt4o/set2/pt-10002221-DS-11.json'
output_path = 'data/resolve-cleaned/gpt4o/10002221-DS-11.json'

# Convert and save to JSON
convert_to_json(summary_path, resolve_path, output_path)

print(f"Converted JSON has been saved to {output_path}")


