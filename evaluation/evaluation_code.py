# import json
# from sklearn.feature_extraction.text import CountVectorizer
# from sentence_transformers import SentenceTransformer, util

# # Load JSON files
# with open('multiple_prompts/llm-annotated-gpt4o-10001401-DS-20_all.json', 'r') as file:
#     json_all = json.load(file)

# with open('single_prompt/llm-annotated-gpt4o-10001401-DS-20.json', 'r') as file:
#     json_single = json.load(file)
    
# with open('data/human-annotations/annotations-gpt4o-10001401-DS-20.json', 'r') as file:
#     json_human_annotation = json.load(file)

# # Initialize SentenceTransformer model
# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# def ngram_overlap(text1, text2, n=3):
#     if not text1.strip() or not text2.strip():
#         return 0.0  # Return 0 overlap if either text is empty or contains only whitespace
#     vectorizer = CountVectorizer(ngram_range=(n, n)).fit([text1, text2])
#     ngrams1 = set(vectorizer.transform([text1]).toarray()[0])
#     ngrams2 = set(vectorizer.transform([text2]).toarray()[0])
#     intersection = ngrams1.intersection(ngrams2)
#     overlap = len(intersection) / len(ngrams1.union(ngrams2))
#     return overlap

# def sentence_bert_similarity(text1, text2):
#     if not text1.strip() or not text2.strip():
#         return 0.0  # Return 0 similarity if either text is empty or contains only whitespace
#     embeddings1 = model.encode(text1, convert_to_tensor=True)
#     embeddings2 = model.encode(text2, convert_to_tensor=True)
#     cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
#     return cosine_scores.item()

# # Analysis container
# analysis = {}

# for category in json_all.keys():
#     if category.startswith("Incorrect"):
#         texts_all = json_all[category]
#         texts_single = json_single.get(category, [])
#         overlap_scores = []
#         for text_data in texts_all:
#             text_all = text_data.get('text', "")
#             for text_single_data in texts_single:
#                 text_single = text_single_data.get('text', "")
#                 overlap = ngram_overlap(text_all, text_single)
#                 overlap_scores.append((text_all, text_single, overlap))
#         analysis[category] = overlap_scores
#     elif category.startswith("Omitted"):
#         omitted_all = json_all[category]
#         omitted_single = json_single.get(category, [])
#         similarity_scores = []
#         for omitted_data in omitted_all:
#             omitted_all_text = omitted_data.get('omittedDetails', "")
#             for omitted_single_data in omitted_single:
#                 omitted_single_text = omitted_single_data.get('omittedDetails', "")
#                 similarity = sentence_bert_similarity(omitted_all_text, omitted_single_text)
#                 similarity_scores.append((omitted_all_text, omitted_single_text, similarity))
#         analysis[category] = similarity_scores

# # Writing analysis to a text file
# with open('evaluation/llm-annotated-gpt4o-10001401-DS-20_analysis.txt', 'w') as file:
#     for category, results in analysis.items():
#         file.write(f"Category: {category}\n")
#         for result in results:
#             if category.startswith("Incorrect"):
#                 text_all, text_single, overlap = result
#                 file.write(f"Text All: {text_all}\nText Single: {text_single}\nN-gram Overlap: {overlap}\n\n")
#             elif category.startswith("Omitted"):
#                 text_all, text_single, similarity = result
#                 file.write(f"Omitted All: {text_all}\nOmitted Single: {text_single}\nSimilarity Score: {similarity}\n\n")

# print("Analysis completed and written to llm-annotated-gpt4o-10001401-DS-20_analysis.txt")

import json
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer, util
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load JSON files
try:
    with open('multiple_prompts/llm-annotated-gpt4o-10001401-DS-20_all.json', 'r') as file:
        json_all = json.load(file)
    with open('single_prompt/llm-annotated-gpt4o-10001401-DS-20.json', 'r') as file:
        json_single = json.load(file)
    with open('data/processed-human-annotations/annotations-gpt4o-10001401-DS-20.json', 'r') as file:
        json_human_annotation = json.load(file)
except FileNotFoundError as e:
    logging.error(f"File not found: {e}")
    exit()
except json.JSONDecodeError as e:
    logging.error(f"Error decoding JSON: {e}")
    exit()

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
    return overlap

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
def process_category(category, human_texts, json_all, json_single):
    overlap_scores_all = []
    overlap_scores_single = []
    similarity_scores_all = []
    similarity_scores_single = []

    if category.startswith("Incorrect") and category != "Incorrect Reasoning":
        texts_all = json_all.get(category, [])
        texts_single = json_single.get(category, [])
        for human_data in human_texts:
            human_text = human_data.get('text', "")
            for text_data in texts_all:
                text_all = text_data.get('text', "")
                overlap_all_vs_human = ngram_overlap(human_text, text_all)
                overlap_scores_all.append((human_text, text_all, overlap_all_vs_human))
            for text_single_data in texts_single:
                text_single = text_single_data.get('text', "")
                overlap_single_vs_human = ngram_overlap(human_text, text_single)
                overlap_scores_single.append((human_text, text_single, overlap_single_vs_human))
        analysis_all[category] = overlap_scores_all
        analysis_single[category] = overlap_scores_single

    elif category.startswith("Omitted"):
        omitted_all = json_all.get(category, [])
        omitted_single = json_single.get(category, [])
        for human_omitted_data in human_texts:
            human_omitted_text = human_omitted_data.get('omittedDetails', "")
            for omitted_data in omitted_all:
                omitted_all_text = omitted_data.get('omittedDetails', "")
                similarity_all_vs_human = sentence_bert_similarity(human_omitted_text, omitted_all_text)
                similarity_scores_all.append((human_omitted_text, omitted_all_text, similarity_all_vs_human))
            for omitted_single_data in omitted_single:
                omitted_single_text = omitted_single_data.get('omittedDetails', "")
                similarity_single_vs_human = sentence_bert_similarity(human_omitted_text, omitted_single_text)
                similarity_scores_single.append((human_omitted_text, omitted_single_text, similarity_single_vs_human))
        analysis_all[category] = similarity_scores_all
        analysis_single[category] = similarity_scores_single
        
    elif category.startswith("Incorrect Reasoning") or category.startswith("Chronological Inconsistency"):
        annotation_all = json_all.get(category, [])
        annotation_single = json_single.get(category, [])
        for human_data in human_texts:
            human_text = human_data.get('text', "")
            for text_data in annotation_all:
                text_all = text_data.get('text', "")
                overlap_all_vs_human = sentence_bert_similarity(human_text, text_all)
                overlap_scores_all.append((human_text, text_all, overlap_all_vs_human))
            for text_single_data in annotation_single:
                text_single = text_single_data.get('text', "")
                overlap_single_vs_human = sentence_bert_similarity(human_text, text_single)
                overlap_scores_single.append((human_text, text_single, overlap_single_vs_human))
        analysis_all[category] = overlap_scores_all
        analysis_single[category] = overlap_scores_single

# Process each category
for category in json_human_annotation.keys():
    human_texts = json_human_annotation.get(category, [])
    if human_texts:
        process_category(category, human_texts, json_all, json_single)
    else:
        logging.warning(f"No human annotations found for category: {category}")

# Writing analysis to a text file
try:
    with open('evaluation/gpt4o-10001401-DS-20_analysis.txt', 'w') as file:
        file.write("COMPARISON OF HUMAN ANNOTATION WITH LLM MULTIPLE PROMPT DETECTION:\n\n")
        for category, results in analysis_all.items():
            file.write(f"CATEGORY: {category}\n")
            if results:
                for result in results:
                    if category.startswith("Incorrect") and category != "Incorrect Reasoning":
                        human_text, text_all, overlap_all_vs_human = result
                        file.write(f"Human: {human_text}\nText All: {text_all}\nN-gram Overlap (Human vs All): {overlap_all_vs_human}\n\n")
                    elif category.startswith("Omitted"):
                        human_omitted_text, omitted_all_text, similarity_all_vs_human = result
                        file.write(f"Human: {human_omitted_text}\nOmitted All: {omitted_all_text}\nSimilarity Score (Human vs All): {similarity_all_vs_human}\n\n")
                    elif category.startswith("Incorrect Reasoning") or category.startswith("Chronological Inconsistency"):
                        human_text, text_all, similarity_all_vs_human = result
                        file.write(f"Human: {human_text}\nText All: {text_all}\nSimilarity Score (Human vs All): {similarity_all_vs_human}\n\n")
            else:
                file.write("Human annotation is present but Multiple_prompt annotation is missing for this category.\n\n")

        file.write("\n######################################################################################################################################\n\n")
        file.write("COMPARISON OF HUMAN ANNOTATION WITH LLM SINGLE PROMPT DETECTION:\n\n")
        for category, results in analysis_single.items():
            file.write(f"CATEGORY: {category}\n")
            if results:
                for result in results:
                    if category.startswith("Incorrect") and category != "Incorrect Reasoning":
                        human_text, text_single, overlap_single_vs_human = result
                        file.write(f"Human: {human_text}\nText Single: {text_single}\nN-gram Overlap (Human vs Single): {overlap_single_vs_human}\n\n")
                    elif category.startswith("Omitted"):
                        human_omitted_text, omitted_single_text, similarity_single_vs_human = result
                        file.write(f"Human: {human_omitted_text}\nOmitted Single: {omitted_single_text}\nSimilarity Score (Human vs Single): {similarity_single_vs_human}\n\n")
                    elif category.startswith("Incorrect Reasoning") or category.startswith("Chronological Inconsistency"):
                        hhuman_text, text_single, similarity_single_vs_human = result
                        file.write(f"Human: {human_text}\nText All: {text_all}\nSimilarity Score (Human vs Single): {similarity_single_vs_human}\n\n")
            else:
                file.write("Human annotation is present but Single_prompt annotation is missing for this category.\n\n")

    logging.info("Analysis completed and written to gpt4o-10001401-DS-20_analysis.txt")
except Exception as e:
    logging.error(f"Error writing analysis to file: {e}")