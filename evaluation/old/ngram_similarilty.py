# import json
# from sklearn.feature_extraction.text import CountVectorizer
# from sentence_transformers import SentenceTransformer, util
# import logging
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # List of note IDs
# # gpt4o_note_ids = ["gpt4o-10001401-DS-20"]
# gpt4o_note_ids = ["gpt4o-10001401-DS-20", "gpt4o-10003299-DS-10", "gpt4o-10006029-DS-16", "gpt4o-10006431-DS-24", "gpt4o-10006820-DS-18"]
# # gpt4o_note_ids = ["gpt4o-10007795-DS-13"]   #throws ValueError: empty vocabulary, need to analyze

# stop_words = set(stopwords.words('english'))

# def remove_stopwords(text):
#     if not text.strip():
#         return text
#     words = word_tokenize(text)
#     filtered_text = ' '.join(word for word in words if word.lower() not in stop_words)
#     return filtered_text
              
# # Process each note_id
# for note_id in gpt4o_note_ids:
#         # Load JSON files
#         try:
#             with open(f'multiple_prompts/multiple-prompts-annotations/llm-annotated-{note_id}-multiple.json', 'r') as file:
#                 json_all = json.load(file)
#             with open(f'single_prompts/single-prompts-annotations/llm-annotated-{note_id}-single.json', 'r') as file:
#                 json_single = json.load(file)
#             with open(f'data/cleaned_annotations/annotations-{note_id}.json', 'r') as file:
#                 json_human_annotation = json.load(file)
#         except FileNotFoundError as e:
#             logging.error(f"File not found: {e}")
#             exit()
#         except json.JSONDecodeError as e:
#             logging.error(f"Error decoding JSON: {e}")
#             exit()

#         # Initialize SentenceTransformer model
#         model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

#         def ngram_overlap(text1, text2, n=3):
#             if not text1.strip() or not text2.strip():
#                 return 0.0  # Return 0 overlap if either text is empty or contains only whitespace
#             vectorizer = CountVectorizer(ngram_range=(n, n))
#             ngrams1 = set(vectorizer.fit_transform([text1]).toarray()[0])
#             ngrams2 = set(vectorizer.fit_transform([text2]).toarray()[0])
#             intersection = ngrams1.intersection(ngrams2)
#             overlap = len(intersection) / len(ngrams1.union(ngrams2))
#             return overlap

#         def sentence_bert_similarity(text1, text2):
#             if not text1.strip() or not text2.strip():
#                 return 0.0  # Return 0 similarity if either text is empty or contains only whitespace
#             embeddings1 = model.encode(text1, convert_to_tensor=True)
#             embeddings2 = model.encode(text2, convert_to_tensor=True)
#             cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
#             return cosine_scores.item()

#         # Analysis containers
#         analysis_all = {}
#         analysis_single = {}

#         # Helper function to process each category
#         def process_category(category, human_texts, json_all, json_single):
#             overlap_scores_all = []
#             overlap_scores_single = []
#             similarity_scores_all = []
#             similarity_scores_single = []

#             if category.startswith("Incorrect") and category != "Incorrect Reasoning":
#                 texts_all = json_all.get(category, [])
#                 texts_single = json_single.get(category, [])
#                 for human_data in human_texts:
#                     # human_text = human_data.get('text', "")
#                     human_text = remove_stopwords(human_data.get('text', ""))
#                     for text_data in texts_all:
#                         # text_all = text_data.get('text', "")
#                         text_all = remove_stopwords(text_data.get('text', ""))
#                         overlap_all_vs_human = ngram_overlap(human_text, text_all)
#                         overlap_scores_all.append((human_text, text_all, overlap_all_vs_human))
#                     for text_single_data in texts_single:
#                         # human_text = human_data.get('text', "")
#                         # text_all = text_data.get('text', "")
#                         # text_single = text_single_data.get('text', "")
#                         text_single = remove_stopwords(text_single_data.get('text', ""))
#                         overlap_single_vs_human = ngram_overlap(human_text, text_single)
#                         overlap_scores_single.append((human_text, text_single, overlap_single_vs_human))
#                 analysis_all[category] = overlap_scores_all
#                 analysis_single[category] = overlap_scores_single

#             elif category.startswith("Omitted"):
#                 omitted_all = json_all.get(category, [])
#                 omitted_single = json_single.get(category, [])
#                 for human_omitted_data in human_texts:
#                     # human_omitted_text = human_omitted_data.get('omittedDetails', "")
#                     human_omitted_text = remove_stopwords(human_omitted_data.get('omittedDetails', ""))
#                     for omitted_data in omitted_all:
#                         # omitted_all_text = omitted_data.get('omittedDetails', "")
#                         omitted_all_text = remove_stopwords(omitted_data.get('omittedDetails', ""))
#                         similarity_all_vs_human = sentence_bert_similarity(human_omitted_text, omitted_all_text)
#                         similarity_scores_all.append((human_omitted_text, omitted_all_text, similarity_all_vs_human))
#                     for omitted_single_data in omitted_single:
#                         # human_omitted_text = human_omitted_data.get('omittedDetails', "")
#                          # omitted_all_text = omitted_data.get('omittedDetails', "")
#                         # omitted_single_text = omitted_single_data.get('omittedDetails', "")
#                         omitted_single_text = remove_stopwords(omitted_single_data.get('omittedDetails', ""))
#                         similarity_single_vs_human = sentence_bert_similarity(human_omitted_text, omitted_single_text)
#                         similarity_scores_single.append((human_omitted_text, omitted_single_text, similarity_single_vs_human))
#                 analysis_all[category] = similarity_scores_all
#                 analysis_single[category] = similarity_scores_single
                
#             elif category.startswith("Incorrect Reasoning") or category.startswith("Chronological Inconsistency"):
#                 annotation_all = json_all.get(category, [])
#                 annotation_single = json_single.get(category, [])
#                 for human_data in human_texts:
#                     # human_text = human_data.get('text', "")
#                     human_text = remove_stopwords(human_data.get('text', ""))
#                     for text_data in annotation_all:
#                         # text_all = text_data.get('text', "")
#                         text_all = remove_stopwords(text_data.get('text', ""))
#                         overlap_all_vs_human = sentence_bert_similarity(human_text, text_all)
#                         overlap_scores_all.append((human_text, text_all, overlap_all_vs_human))
#                     for text_single_data in annotation_single:
#                         # human_text = human_data.get('text', "")
#                         # text_all = text_data.get('text', "")
#                         # text_single = text_single_data.get('text', "")
#                         text_single = remove_stopwords(text_single_data.get('text', ""))
#                         overlap_single_vs_human = sentence_bert_similarity(human_text, text_single)
#                         overlap_scores_single.append((human_text, text_single, overlap_single_vs_human))
#                 analysis_all[category] = overlap_scores_all
#                 analysis_single[category] = overlap_scores_single

#         # Process each category
#         for category in json_human_annotation.keys():
#             human_texts = json_human_annotation.get(category, [])
#             if not human_texts:
#                 human_texts = [{"text": "", "omittedDetails": ""}]
#             process_category(category, human_texts, json_all, json_single)

#         # Ensure to process categories even if human_texts are empty
#         for category in json_all.keys():
#             if category not in analysis_all:
#                 process_category(category, [{"text": "", "omittedDetails": ""}], json_all, json_single)

#         # Writing analysis to a text file
#         try:
#             with open(f'evaluation/analysis_files/{note_id}_analysis.txt', 'w') as file:
#                 file.write("COMPARISON OF HUMAN ANNOTATION WITH LLM MULTIPLE PROMPT DETECTION:\n\n")
#                 for category, results in analysis_all.items():
#                     file.write(f"\nCATEGORY: {category}\n")
#                     for result in results:
#                         if category.startswith("Incorrect") and category != "Incorrect Reasoning":
#                             human_text, text_all, overlap_all_vs_human = result
#                             file.write(f"Human: {human_text}\nText All: {text_all}\nN-gram Overlap (Human vs All): {overlap_all_vs_human}\n\n")
#                         elif category.startswith("Omitted"):
#                             human_omitted_text, omitted_all_text, similarity_all_vs_human = result
#                             file.write(f"Human: {human_omitted_text}\nOmitted All: {omitted_all_text}\nSimilarity Score (Human vs All): {similarity_all_vs_human}\n\n")
#                         elif category.startswith("Incorrect Reasoning") or category.startswith("Chronological Inconsistency"):
#                             human_text, text_all, similarity_all_vs_human = result
#                             file.write(f"Human: {human_text}\nText All: {text_all}\nSimilarity Score (Human vs All): {similarity_all_vs_human}\n\n")

#                 file.write("\n######################################################################################################################################\n\n")
#                 file.write("COMPARISON OF HUMAN ANNOTATION WITH LLM SINGLE PROMPT DETECTION:\n\n")
#                 for category, results in analysis_single.items():
#                     file.write(f"\nCATEGORY: {category}\n")
#                     for result in results:
#                         if category.startswith("Incorrect") and category != "Incorrect Reasoning":
#                             human_text, text_single, overlap_single_vs_human = result
#                             file.write(f"Human: {human_text}\nText Single: {text_single}\nN-gram Overlap (Human vs Single): {overlap_single_vs_human}\n\n")
#                         elif category.startswith("Omitted"):
#                             human_omitted_text, omitted_single_text, similarity_single_vs_human = result
#                             file.write(f"Human: {human_omitted_text}\nOmitted Single: {omitted_single_text}\nSimilarity Score (Human vs Single): {similarity_single_vs_human}\n\n")
#                         elif category.startswith("Incorrect Reasoning") or category.startswith("Chronological Inconsistency"):
#                             human_text, text_single, similarity_single_vs_human = result
#                             file.write(f"Human: {human_text}\nText Single: {text_single}\nSimilarity Score (Human vs Single): {similarity_single_vs_human}\n\n")

#             logging.info(f"Analysis completed and written to {note_id}.txt")
#         except Exception as e:
#             logging.error(f"Error writing analysis to file: {e}")

##################################################################################

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
gpt4o_note_ids = ["gpt4o-10001401-DS-20", "gpt4o-10003299-DS-10", "gpt4o-10006029-DS-16", "gpt4o-10006431-DS-24", "gpt4o-10006820-DS-18"]
# gpt4o_note_ids = ["gpt4o-10007795-DS-13"]   #throws ValueError: empty vocabulary, need to analyze

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
            # human_text = human_data.get('text', "")
            human_text = remove_stopwords(human_data.get('text', ""))
            for text_data in texts_all:
                # text_all = text_data.get('text', "")
                text_all = remove_stopwords(text_data.get('text', ""))
                overlap_all_vs_human = ngram_overlap(human_text, text_all)
                overlap_scores_all.append((human_text, text_all, overlap_all_vs_human))
            for text_single_data in texts_single:
                 # text_single = text_single_data.get('text', "")
                text_single = remove_stopwords(text_single_data.get('text', ""))
                overlap_single_vs_human = ngram_overlap(human_text, text_single)
                overlap_scores_single.append((human_text, text_single, overlap_single_vs_human))
        analysis_all[category] = overlap_scores_all
        analysis_single[category] = overlap_scores_single

    elif category.startswith("Omitted"):
        omitted_all = json_all.get(category, [])
        omitted_single = json_single.get(category, [])
        for human_omitted_data in human_texts:
             # human_omitted_text = human_omitted_data.get('omittedDetails', "")
            human_omitted_text = remove_stopwords(human_omitted_data.get('omittedDetails', ""))
            for omitted_data in omitted_all:
                # omitted_all_text = omitted_data.get('omittedDetails', "")
                omitted_all_text = remove_stopwords(omitted_data.get('omittedDetails', ""))
                similarity_all_vs_human = sentence_bert_similarity(human_omitted_text, omitted_all_text)
                similarity_scores_all.append((human_omitted_text, omitted_all_text, similarity_all_vs_human))
            for omitted_single_data in omitted_single:
                # omitted_single_text = omitted_single_data.get('omittedDetails', "")
                omitted_single_text = remove_stopwords(omitted_single_data.get('omittedDetails', ""))
                similarity_single_vs_human = sentence_bert_similarity(human_omitted_text, omitted_single_text)
                similarity_scores_single.append((human_omitted_text, omitted_single_text, similarity_single_vs_human))
        analysis_all[category] = similarity_scores_all
        analysis_single[category] = similarity_scores_single

    elif category.startswith("Incorrect Reasoning") or category.startswith("Chronological Inconsistency"):
        annotation_all = json_all.get(category, [])
        annotation_single = json_single.get(category, [])
        for human_data in human_texts:
            # human_text = human_data.get('text', "")
            human_text = remove_stopwords(human_data.get('text', ""))
            for text_data in annotation_all:
                # text_all = text_data.get('text', "")
                text_all = remove_stopwords(text_data.get('text', ""))
                overlap_all_vs_human = sentence_bert_similarity(human_text, text_all)
                overlap_scores_all.append((human_text, text_all, overlap_all_vs_human))
            for text_single_data in annotation_single:
                # text_single = text_single_data.get('text', "")
                text_single = remove_stopwords(text_single_data.get('text', ""))
                overlap_single_vs_human = sentence_bert_similarity(human_text, text_single)
                overlap_scores_single.append((human_text, text_single, overlap_single_vs_human))
        analysis_all[category] = overlap_scores_all
        analysis_single[category] = overlap_scores_single

# Process each note_id
for note_id in gpt4o_note_ids:
    # Load JSON files
    try:
        with open(f'multiple_prompts/multiple-prompts-annotations/llm-annotated-{note_id}-multiple.json', 'r') as file:
            json_all = json.load(file)
        with open(f'single_prompts/single-prompts-annotations/llm-annotated-{note_id}-single.json', 'r') as file:
            json_single = json.load(file)
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
        process_category(category, human_texts, json_all, json_single)

    # Ensure to process categories even if human_texts are empty
    for category in json_all.keys():
        if category not in analysis_all:
            process_category(category, [{"text": "", "omittedDetails": ""}], json_all, json_single)
    
            # Writing analysis to a text file
    try:
            with open(f'evaluation/analysis_files/{note_id}_analysis.txt', 'w') as file:
                file.write("COMPARISON OF HUMAN ANNOTATION WITH LLM MULTIPLE PROMPT DETECTION:\n\n")
                for category, results in analysis_all.items():
                    file.write(f"\nCATEGORY: {category}\n")
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

                file.write("\n######################################################################################################################################\n\n")
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

# # Calculate precision and recall
# def calculate_precision_recall(analysis, thresholds):
#     precision = {t: {} for t in thresholds}
#     recall = {t: {} for t in thresholds}

#     for t in thresholds:
#         for category, results in analysis.items():
#             true_positives = sum(1 for _, _, score in results if score >= t)
#             false_positives = len(results) - true_positives
#             false_negatives = len(results) - true_positives

#             precision[t][category] = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
#             recall[t][category] = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

#     return precision, recall

# # Define thresholds
# thresholds = [i / 100.0 for i in range(50, 101, 25)]

# # Calculate precision and recall for all and single prompt analyses
# precision_all, recall_all = calculate_precision_recall(analysis_all, thresholds)
# precision_single, recall_single = calculate_precision_recall(analysis_single, thresholds)

# # Plot precision-recall curves
# def plot_precision_recall(precision, recall, title, categories):
#     plt.figure()
#     for category in categories:
#         plt.plot(
#             [recall[t][category] for t in thresholds],
#             [precision[t][category] for t in thresholds],
#             marker='o',
#             label=category
#         )
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title(title)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.show()

# # Get list of categories
# categories = list(analysis_all.keys())

# # Plot for all prompt analyses
# plot_precision_recall(precision_all, recall_all, "Precision-Recall Curve for Multiple Prompts", categories)
# plot_precision_recall(precision_single, recall_single, "Precision-Recall Curve for Single Prompts", categories)
