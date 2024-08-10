# #in human annotations, change omitted label to "Specific to General" and omittedDetails to explanation.
# #in human annotations, remove other inconsistency category

# #only for chronological inconsistency compare explanations, for everything else compare texts.
# #considere LLM detection as a hit even if one non stop word from "text" section in LLM annotation is present in human annotation
# #if indexes need to be considered for LLM annotation, how to do that?

# #######################################################################
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
                        break
                results.append({
                    "Key": key,
                    "Text Entry": text_human,
                    "Hit": hit
                })
        else:
            # If the key is not present in llmanno, all hits are 0
            for human_entry in humananno[key]:
                results.append({
                    "Key": key,
                    "Text Entry": human_entry["text"],
                    "Hit": 0
                })
    
    # Convert results to a pandas DataFrame for better readability
    df_results = pd.DataFrame(results)
    return df_results

# Example JSON data
humananno = {
    "A": [{"text": "The patient has a headache.", "explanation": "This is a common symptom."}, {"text": "apple keeps doctor away", "explanation": "its a common saying"}],
    "B": [{"text": "Prescribed 500mg of medication.", "explanation": "Standard dosage."}]
}

llmanno = {
    "A": [{"text": "The patient complains of a headache.", "explanation": "Symptom noted."}],
    "C": [{"text": "Blood pressure is normal.", "explanation": "No concerns here."}]
}

# Check the annotations and generate the new table
result_df = check_annotations(humananno, llmanno)

# Display the result
print(result_df)
