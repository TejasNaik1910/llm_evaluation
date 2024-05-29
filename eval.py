import os
import json
from openai import AzureOpenAI

# Load the content from files
with open('guidelines.txt', 'r') as file:
    guidelines_content = file.read()

with open('output_format.json', 'r') as file:
    output_format_content = file.read()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

def use_azureai(text):
    response = client.chat.completions.create(
        model="gpt4o-deployment", 
        messages=[{"role": "user", "content": text}]
    )
    return response.choices[0].message.content

# List of note IDs
note_ids = ["10001401-DS-20", "10054464-DS-17", 
    "10002221-DS-12", "10003299-DS-10", "10056223-DS-14", "10004401-DS-26", 
    "10056612-DS-8", "10006029-DS-16", "10006431-DS-24", "10006580-DS-21", 
    "10006820-DS-18", "10007795-DS-13", "10008628-DS-3", "10010440-DS-5", 
    "10011938-DS-16", "10012292-DS-9", "10014354-DS-23", "10016142-DS-19", 
    "10017285-DS-3", "10018052-DS-16", "10057126-DS-7", "10020306-DS-9"]

# Process each note_id
for note_id in note_ids:
    # Load the EHR note and summary content based on note_id
    ehr_note_file = f'data/notes/oncology-report-{note_id}.txt'
    summary_file = f'data/summaries/gpt4o-doc-{note_id}.txt'
    
    with open(ehr_note_file, 'r') as file:
        ehr_note_content = file.read()
    
    with open(summary_file, 'r') as file:
        summary_content = file.read()

    prompt = f"""
    Given below is a TASK OVERVIEW followed by the GUIDELINES, JSON_FORMAT, EHR_NOTE and then finally another piece of text which is called the SUMMARY. You are an annotator, and you have to annotate the EHR_NOTE and SUMMARY based on the GUIDELINES. You will be provided multiple examples of which instances are referred to as hallucinations. The examples will contain the explanation of why a particular instance would be considered hallucinated or not.

    TASK OVERVIEW
    You will be given an EHR note and a piece of text which is supposed to be a summary for the EHR Note. Your task is to check if the summary has any missing or inconsistent information with the EHR note.

    Instructions:

    1. Your task is to provide phrases/words for the below-mentioned kinds of hallucinations. Each of these hallucinations has two sub-categories: OMITTED and INCORRECT. For OMITTED sub-categories, please provide a piece of the sentence from the EHR_NOTE or the SUMMARY. You do not need to provide the same for INCORRECT sub-categories.
        a) Patient Information
        b) Patient History
        c) Symptoms/Diagnosis/Surgical Procedures
        d) Medicine related instructions
        e) Followup

    2. Your task is to provide an explanation for the below-mentioned hallucination categories: 
        a) Chronological Inconsistency
        b) Incorrect Reasoning

    3. If you spot some other inconsistencies that do not fall into any of the above categories, mark it as OMITTED or INCORRECT “Other Inconsistency”. For the OMITTED sub-category, please provide a piece of the sentence from the EHR NOTE or the SUMMARY. You do not need to provide the same for the INCORRECT sub-category.

    4. Use the GUIDELINES mentioned as follows:
    {guidelines_content}

    5. Please provide your response in the JSON_FORMAT as mentioned in this file:
    {output_format_content}

    Use this EHR_NOTE:
    {ehr_note_content}

    Use this as the SUMMARY:
    {summary_content}
    """
    
    response = use_azureai(prompt)
    
    # Save the response
    output_file = f'llm-annotated-gpt4o-{note_id}.json'
    with open(output_file, 'w') as file:
        file.write(response)
        print("Done")
