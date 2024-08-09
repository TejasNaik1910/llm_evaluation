#in human annotations, change omitted label to "Specific to General" and omittedDetails to explanation.
#in human annotations, remove other inconsistency category

#only for chronological inconsistency compare explanations, for everything else compare texts.
#considere LLM detection as a hit even if one non stop word from "text" section in LLM annotation is present in human annotation
#if indexes need to be considered for LLM annotation, how to do that?

#######################################################################

import json
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer, util
import logging
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download stopwords if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')

# List of note IDs
gpt4o_note_ids = ["gpt4o-10001401-DS-20"]

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    if not text.strip():
        return text
    words = word_tokenize(text)
    filtered_text = ' '.join(word for word in words if word.lower() not in stop_words)
    return filtered_text

# Initialize SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def ngram_overlap(text1, text2, n=3):
    if not text1.strip() or not text2.strip():
        return 0.0  # Return 0 overlap if either text is empty or contains only whitespace
    vectorizer = CountVectorizer(ngram_range=(n, n))
    ngrams1 = set(vectorizer.fit_transform([text1]).toarray()[0])
    ngrams2 = set(vectorizer.fit_transform([text2]).toarray()[0])
    intersection = ngrams1.intersection(ngrams2)
    overlap = len(intersection) / len(ngrams1.union(ngrams2))
    return len(intersection) > 0  # Return True if there is an overlap, otherwise False
    # return overlap                # return this to get overlap score for n-gram

def sentence_bert_similarity(text1, text2):
    if not text1.strip() or not text2.strip():
        return 0.0  # Return 0 similarity if either text is empty or contains only whitespace
    embeddings1 = model.encode(text1, convert_to_tensor=True)
    embeddings2 = model.encode(text2, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    return cosine_scores.item()

# Analysis containers
analysis_all = {}
analysis_single = {}

# Helper function to process each category
def process_category(category, human_texts, json_single):
    overlap_scores_all = []
    overlap_scores_single = []
    similarity_scores_all = []
    similarity_scores_single = []

    if category.startswith("Incorrect") and category != "Incorrect Reasoning":
        texts_single = json_single.get(category, [])
        for human_data in human_texts:
            # human_text = human_data.get('text', "")
            human_text = remove_stopwords(human_data.get('text', ""))
            for text_single_data in texts_single:
                 # text_single = text_single_data.get('text', "")
                text_single = remove_stopwords(text_single_data.get('text', ""))
                overlap_single_vs_human = ngram_overlap(human_text, text_single)
                overlap_scores_single.append((human_text, text_single, overlap_single_vs_human))
        analysis_single[category] = overlap_scores_single

    elif category.startswith("Omitted"):
        omitted_single = json_single.get(category, [])
        for human_omitted_data in human_texts:
             # human_omitted_text = human_omitted_data.get('omittedDetails', "")
            human_omitted_text = remove_stopwords(human_omitted_data.get('omittedDetails', ""))
            for omitted_single_data in omitted_single:
                # omitted_single_text = omitted_single_data.get('omittedDetails', "")
                omitted_single_text = remove_stopwords(omitted_single_data.get('omittedDetails', ""))
                similarity_single_vs_human = sentence_bert_similarity(human_omitted_text, omitted_single_text)
                similarity_scores_single.append((human_omitted_text, omitted_single_text, similarity_single_vs_human))
        analysis_single[category] = similarity_scores_single

    elif category.startswith("Incorrect Reasoning") or category.startswith("Chronological Inconsistency"):
        annotation_single = json_single.get(category, [])
        for human_data in human_texts:
            # human_text = human_data.get('text', "")
            human_text = remove_stopwords(human_data.get('text', ""))
            for text_single_data in annotation_single:
                # text_single = text_single_data.get('text', "")
                text_single = remove_stopwords(text_single_data.get('text', ""))
                overlap_single_vs_human = sentence_bert_similarity(human_text, text_single)
                overlap_scores_single.append((human_text, text_single, overlap_single_vs_human))
        analysis_single[category] = overlap_scores_single

# Process each note_id
for note_id in gpt4o_note_ids:
    # Load JSON files
    try:
        with open(f'single_prompts/single-prompts-annotations/llm-annotated-{note_id}-single.json', 'r') as file:
            json_llm_annotation = json.load(file)
        with open(f'data/cleaned_annotations/annotations-{note_id}.json', 'r') as file:
            json_human_annotation = json.load(file)
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        exit()
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON: {e}")
        exit()

    # Process each category
    for category in json_human_annotation.keys():
        human_texts = json_human_annotation.get(category, [])
        if not human_texts:
            human_texts = [{"text": "", "omittedDetails": ""}]
        process_category(category, human_texts, json_human_annotation)
        
    json_all= {
    "Incorrect Patient Information": [
      {
        "text": ""
      }
    ],
    "Specific to General Patient Information": [
        {
          "text": ""
        }
      ],
    "Incorrect Patient History": [
        {
          "text": ""
        }
      ],
    "Specific to General Patient History": [
      {
        "text": ""
      }
    ],
    "Incorrect Symptoms/Diagnosis": [
        {
          "text": ""
        }
      ],
    "Specific to General Symptoms/Diagnosis": [
      {
        "text": ""
      }
    ],
    "Incorrect Medicinal Instructions": [
        {
          "text": ""
        }
      ],
    "Specific to General Medicinal Instructions": [
      {
        "text": ""
      }
    ],
    "Incorrect Followup": [
        {
          "text": ""
        }
      ],
    "Specific to General Followup": [
      {
        "text": ""
      }
    ],
    "Incorrect Reasoning": [
      {
        "text": "",
        "explanation":""
      }
    ],
    "Chronological Inconsistency": [
      {
        "text": "",
        "explanation":""
      }
    ] 
  }

    # Ensure to process categories even if human_texts are empty
    for category in json_all.keys():
        if category not in analysis_single:
            process_category(category, [{"text": "", "omittedDetails": ""}], json_human_annotation)
            # Writing analysis to a text file
    try:
            with open(f'evaluation/analysis_files/{note_id}_analysis.txt', 'w') as file:
                file.write("COMPARISON OF HUMAN ANNOTATION WITH LLM SINGLE PROMPT DETECTION:\n\n")
                for category, results in analysis_single.items():
                    file.write(f"\nCATEGORY: {category}\n")
                    for result in results:
                        if category.startswith("Incorrect") and category != "Incorrect Reasoning":
                            human_text, text_single, overlap_single_vs_human = result
                            file.write(f"Human: {human_text}\nText Single: {text_single}\nN-gram Overlap (Human vs Single): {overlap_single_vs_human}\n\n")
                        elif category.startswith("Omitted"):
                            human_omitted_text, omitted_single_text, similarity_single_vs_human = result
                            file.write(f"Human: {human_omitted_text}\nOmitted Single: {omitted_single_text}\nSimilarity Score (Human vs Single): {similarity_single_vs_human}\n\n")
                        elif category.startswith("Incorrect Reasoning") or category.startswith("Chronological Inconsistency"):
                            human_text, text_single, similarity_single_vs_human = result
                            file.write(f"Human: {human_text}\nText Single: {text_single}\nSimilarity Score (Human vs Single): {similarity_single_vs_human}\n\n")

            logging.info(f"Analysis completed and written to {note_id}.txt")
    except Exception as e:
            logging.error(f"Error writing analysis to file: {e}")