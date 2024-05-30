import os
import json
import re
from openai import AzureOpenAI

# Initialize Azure OpenAI client
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

def extract_json(response):
    stack = []
    json_str = ""
    in_json = False
    for char in response:
        if char == '{':
            stack.append(char)
            in_json = True
        if in_json:
            json_str += char
        if char == '}':
            stack.pop()
            if not stack:
                break
    return json_str if json_str else None

# List of note IDs
# note_ids = ["10001401-DS-20", "10054464-DS-17", 
#     "10002221-DS-12", "10003299-DS-10", "10056223-DS-14", "10004401-DS-26", 
#     "10056612-DS-8", "10006029-DS-16", "10006431-DS-24", "10006580-DS-21", 
#     "10006820-DS-18", "10007795-DS-13", "10008628-DS-3", "10010440-DS-5", 
#     "10011938-DS-16", "10012292-DS-9", "10014354-DS-23", "10016142-DS-19", 
#     "10017285-DS-3", "10018052-DS-16", "10057126-DS-7", "10020306-DS-9"]
note_ids = ["10001401-DS-20"]

# Process each note_id
for note_id in note_ids:
    # Load the EHR note and summary content based on note_id
    ehr_note_file = f'data/notes/oncology-report-{note_id}.txt'
    summary_file = f'data/summaries/gpt4o-doc-{note_id}.txt'
    
    with open(ehr_note_file, 'r') as file:
        ehr_note_content = file.read()
    
    with open(summary_file, 'r') as file:
        summary_content = file.read()
    
    # Initialize the list to store responses
    all_responses = []

    # Process each guideline category
    for i in range(8):
        with open(f"""multiple_prompts/guidelines/guidelines_{i}.txt""", 'r') as file:
            guideline_content = file.read()

        with open(f"""multiple_prompts/op_files/op_{i}.json""", 'r') as file:
            output_format_content = file.read()

        prompt = f"""
        Given below is a TASK OVERVIEW followed by the GUIDELINES, EHR_NOTE, and then the SUMMARY. 
        You are an annotator, and you have to annotate the EHR_NOTE and SUMMARY based on the GUIDELINES. 
        You will be provided multiple examples of which instances are referred to as hallucinations. 
        The examples will contain the explanation of why a particular instance would be considered hallucinated or not.

        TASK OVERVIEW
        You will be given an EHR note and a piece of text which is supposed to be a summary for the EHR Note. 
        Your task is to check if the summary has any missing or inconsistent information with the EHR note.

        Instructions:
        {guideline_content}

        Use this EHR_NOTE:
        {ehr_note_content}

        Use this as the SUMMARY:
        {summary_content}

        Please provide your response in the JSON_FORMAT as mentioned in this file:
        {output_format_content}
        """

        response = use_azureai(prompt)

        # Extract JSON from the response
        json_content = extract_json(response)

        if json_content:
            try:
                response_dict = json.loads(json_content)
                all_responses.append(response_dict)
                print(f"Guideline {i} Done")
            except json.JSONDecodeError:
                print(f"Error decoding JSON for guideline {i}: {json_content}")
        else:
            print(f"No JSON found for guideline {i}: {response}")
    
    # Save all responses to a single JSON file after processing all guidelines
    output_file = f'multiple_prompts/llm-annotated-gpt4o-{note_id}_all.json'
    with open(output_file, 'w') as file:
        json.dump(all_responses, file, indent=4)