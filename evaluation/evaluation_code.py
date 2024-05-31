import json
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer, util

# Load JSON files
with open('multiple_prompts/llm-annotated-gpt4o-10001401-DS-20_all.json', 'r') as file:
    json_all = json.load(file)

with open('single_prompt/llm-annotated-gpt4o-10001401-DS-20.json', 'r') as file:
    json_single = json.load(file)
    
# with open('data/human-annotations/annotations-gpt4o-10001401-DS-20.json', 'r') as file:
#     json_human_annotation = json.load(file)

# Initialize SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def ngram_overlap(text1, text2, n=3):
    if not text1.strip() or not text2.strip():
        return 0.0  # Return 0 overlap if either text is empty or contains only whitespace
    vectorizer = CountVectorizer(ngram_range=(n, n)).fit([text1, text2])
    ngrams1 = set(vectorizer.transform([text1]).toarray()[0])
    ngrams2 = set(vectorizer.transform([text2]).toarray()[0])
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

# Analysis container
analysis = {}

for category in json_all.keys():
    if category.startswith("Incorrect"):
        texts_all = json_all[category]
        texts_single = json_single.get(category, [])
        overlap_scores = []
        for text_data in texts_all:
            text_all = text_data.get('text', "")
            for text_single_data in texts_single:
                text_single = text_single_data.get('text', "")
                overlap = ngram_overlap(text_all, text_single)
                overlap_scores.append((text_all, text_single, overlap))
        analysis[category] = overlap_scores
    elif category.startswith("Omitted"):
        omitted_all = json_all[category]
        omitted_single = json_single.get(category, [])
        similarity_scores = []
        for omitted_data in omitted_all:
            omitted_all_text = omitted_data.get('omittedDetails', "")
            for omitted_single_data in omitted_single:
                omitted_single_text = omitted_single_data.get('omittedDetails', "")
                similarity = sentence_bert_similarity(omitted_all_text, omitted_single_text)
                similarity_scores.append((omitted_all_text, omitted_single_text, similarity))
        analysis[category] = similarity_scores

# Writing analysis to a text file
with open('evaluation/llm-annotated-gpt4o-10001401-DS-20_analysis.txt', 'w') as file:
    for category, results in analysis.items():
        file.write(f"Category: {category}\n")
        for result in results:
            if category.startswith("Incorrect"):
                text_all, text_single, overlap = result
                file.write(f"Text All: {text_all}\nText Single: {text_single}\nN-gram Overlap: {overlap}\n\n")
            elif category.startswith("Omitted"):
                text_all, text_single, similarity = result
                file.write(f"Omitted All: {text_all}\nOmitted Single: {text_single}\nSimilarity Score: {similarity}\n\n")

print("Analysis completed and written to llm-annotated-gpt4o-10001401-DS-20_analysis.txt")