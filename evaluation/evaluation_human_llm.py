#in human annotations, change omitted label to "Specific to General" and omittedDetails to explanation.
#in human annotations, remove other inconsistency category

#only for chronological inconsistency, incorrect reasoning compare human-texts and llm-explanations, for everything else compare human-texts and llm-texts.
#considere LLM detection as a hit even if one non stop word from "text" section in LLM annotation is present in human annotation
#if indexes need to be considered for LLM annotation, how to do that?
#need to do resolve evaluation as well

# True Positives: Incremented when there is a match between an entry in humananno and llmanno.
# False Negatives: Incremented when an entry in humananno is not matched by any entry in llmanno.
# False Positives: Incremented when an entry in llmanno does not match any entry in humananno.

# #######################################################################

# import json
# import pandas as pd
# from collections import Counter
# from nltk.corpus import stopwords

# # Load stopwords
# stop_words = set(stopwords.words('english'))

# def remove_stopwords(text):
#     """Remove stopwords from a given text."""
#     tokens = text.split()
#     filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
#     return " ".join(filtered_tokens)

# def one_gram_overlap(text1, text2):
#     """Calculate the one-gram overlap between two texts after removing stopwords."""
#     text1 = remove_stopwords(text1)
#     text2 = remove_stopwords(text2)
#     tokens1 = text1.split()
#     tokens2 = text2.split()
#     count1 = Counter(tokens1)
#     count2 = Counter(tokens2)
#     overlap = sum((count1 & count2).values())
#     return overlap

# def check_annotations(humananno, llmanno):
#     results = []
#     true_positives = 0
#     false_positives = 0
#     false_negatives = 0
    
#     # Traverse through keys in humananno
#     for key in humananno:
#         if key in llmanno:
#             # Compare each entry in humananno with all entries in llmanno
#             for human_entry in humananno[key]:
#                 text_human = human_entry["text"]
#                 hit = 0
#                 for llm_entry in llmanno[key]:
#                     text_llm = llm_entry["text"]
#                     if one_gram_overlap(text_human, text_llm) > 0:
#                         hit = 1
#                         true_positives += 1
#                         break
#                 if hit == 0:
#                     false_negatives += 1
#                 results.append({
#                     "Key": key,
#                     "Text Entry": text_human,
#                     "Hit": hit
#                 })
#         else:
#             # If the key is not present in llmanno, all hits are 0 and all are false negatives
#             for human_entry in humananno[key]:
#                 results.append({
#                     "Key": key,
#                     "Text Entry": human_entry["text"],
#                     "Hit": 0
#                 })
#                 false_negatives += 1
    
#     # Calculate false positives
#     for key in llmanno:
#         if key in humananno:
#             for llm_entry in llmanno[key]:
#                 text_llm = llm_entry["text"]
#                 hit = 0
#                 for human_entry in humananno[key]:
#                     text_human = human_entry["text"]
#                     if one_gram_overlap(text_human, text_llm) > 0:
#                         hit = 1
#                         break
#                 if hit == 0:
#                     false_positives += 1
#         else:
#             false_positives += len(llmanno[key])

#     # Calculate Precision and Recall
#     precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
#     recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

#     # Calculate F1 Score
#     if precision + recall > 0:
#         f1_score = 2 * (precision * recall) / (precision + recall)
#     else:
#         f1_score = 0

#     # Convert results to a pandas DataFrame for better readability
#     df_results = pd.DataFrame(results)
    
#     return df_results, precision, recall, f1_score

# # Example JSON data
# humananno = {
#     "A": [{"text": "The patient has a headache.", "explanation": "This is a common symptom."}, 
#           {"text": "my dog stepped on a bee", "explanation": "This is funny"}],
#     "B": [{"text": "Prescribed 500mg of medication.", "explanation": "Standard dosage."}]
# }

# llmanno = {
#     "A": [{"text": "The patient complains of a headache.", "explanation": "Symptom noted."}],
#     "C": [{"text": "Blood pressure is normal.", "explanation": "No concerns here."}]
# }

# # Check the annotations and generate the new table
# result_df, precision, recall, f1_score = check_annotations(humananno, llmanno)

# # Display the result
# print(result_df)
# print(f"Precision: {precision:.2f}")
# print(f"Recall: {recall:.2f}")
# print(f"F1 Score: {f1_score:.2f}")

#############################################################################

import os
import json
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords

# Load stopwords
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    """Remove stopwords from a given text."""
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return " ".join(filtered_tokens)

def one_gram_overlap(text1, text2):
    """Calculate the one-gram overlap between two texts after removing stopwords."""
    text1 = remove_stopwords(text1)
    text2 = remove_stopwords(text2)
    tokens1 = text1.split()
    tokens2 = text2.split()
    count1 = Counter(tokens1)
    count2 = Counter(tokens2)
    overlap = sum((count1 & count2).values())
    return overlap

def check_annotations(humananno, llmanno):
    results = []
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Traverse through keys in humananno
    for key in humananno:
        if key in llmanno:
            # Compare each entry in humananno with all entries in llmanno
            for human_entry in humananno[key]:
                text_human = human_entry["text"]
                hit = 0
                for llm_entry in llmanno[key]:
                    text_llm = llm_entry["text"]
                    if one_gram_overlap(text_human, text_llm) > 0:
                        hit = 1
                        true_positives += 1
                        break
                if hit == 0:
                    false_negatives += 1
                results.append({
                    "Key": key,
                    "Text Entry": text_human,
                    "Hit": hit
                })
        else:
            # If the key is not present in llmanno, all hits are 0 and all are false negatives
            for human_entry in humananno[key]:
                results.append({
                    "Key": key,
                    "Text Entry": human_entry["text"],
                    "Hit": 0
                })
                false_negatives += 1
    
    # Calculate false positives
    for key in llmanno:
        if key in humananno:
            for llm_entry in llmanno[key]:
                text_llm = llm_entry["text"]
                hit = 0
                for human_entry in humananno[key]:
                    text_human = human_entry["text"]
                    if one_gram_overlap(text_human, text_llm) > 0:
                        hit = 1
                        break
                if hit == 0:
                    false_positives += 1
        else:
            false_positives += len(llmanno[key])

    # Calculate Precision and Recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    # Calculate F1 Score
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0

    # Convert results to a pandas DataFrame for better readability
    df_results = pd.DataFrame(results)
    
    return df_results, precision, recall, f1_score

def process_annotations(noteid_list, human_folder, llm_folder, output_csv):
    results_list = []

    for note_id in noteid_list:
        # Construct file paths
        human_file = os.path.join(human_folder, f"annotations-gpt4o-{note_id}.json")      #change filename format for gpt4o and llama3
        llm_file = os.path.join(llm_folder, f"{note_id}.json")          #change filename format for gpt4o and llama3

        # Load the JSON data
        with open(human_file, 'r') as hf:
            humananno = json.load(hf)
        with open(llm_file, 'r') as lf:
            llmanno = json.load(lf)

        # Calculate evaluation metrics
        eval_df, precision, recall, f1_score = check_annotations(humananno, llmanno)
        
        # Append results to list
        results_list.append({
            "note_id": note_id,
            "eval_df": eval_df.to_json(orient='records'),
            "precision": precision,
            "recall": recall,
            "f1": f1_score
        })

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results_list)

    # Save the DataFrame to a CSV file
    results_df.to_csv(output_csv, index=False)


noteid_list = [
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
human_folder = 'data/human-annotations/set2/gpt4o'  # Replace with the path to your human annotations folder
llm_folder = 'single_prompts/single-prompts-annotations/set2/gpt4o'  # Replace with the path to your LLM annotations folder
output_csv = 'evaluation/human-llm-gpt4o_evaluation_results.csv'  # Replace with the desired output CSV file path

process_annotations(noteid_list, human_folder, llm_folder, output_csv)

print(f"Results saved to {output_csv}")

