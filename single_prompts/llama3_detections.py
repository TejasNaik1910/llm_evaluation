import os
import json
from openai import AzureOpenAI

# Load the content from files
with open('single_prompts/guidelines.txt', 'r') as file:
    guidelines_content = file.read()

with open('single_prompts/output_format.json', 'r') as file:
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

#update this to use llama3 for the detection task. Once done, update the method name at Line 148.
def use_llama3(text): 
    return

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
note_ids = [ "10000935-DS-21"]#for testing code, uncomment below variable for full execution
# note_ids = [
#     "10000935-DS-21",
#     "10000980-DS-23",
#     "10001401-DS-20",
#     "10054464-DS-17",
#     "10002221-DS-12",
#     "10003299-DS-10",
#     "10056223-DS-14",
#     "10004401-DS-26",
#     "10056612-DS-8",
#     "10006029-DS-16",
#     "10006431-DS-24",
#     "10006580-DS-21",
#     "10006820-DS-18",
#     "10007795-DS-13",
#     "10008628-DS-3",
#     "10010440-DS-5",
#     "10011938-DS-16",
#     "10012292-DS-9",
#     "10014354-DS-23",
#     "10016142-DS-19",
#     "10017285-DS-3",
#     "10018052-DS-16",
#     "10057126-DS-7",
#     "10020306-DS-9",
#     "10021312-DS-20",
#     "10021493-DS-18",
#     "10022373-DS-5",
#     "10057731-DS-7",
#     "10059192-DS-11",
#     "10024331-DS-30",
#     "10060733-DS-12",
#     "10025862-DS-12",
#     "10027957-DS-18",
#     "10032176-DS-14",
#     "10034049-DS-20",
#     "10035631-DS-12",
#     "10060764-DS-6",
#     "10061124-DS-12",
#     "10041127-DS-16",
#     "10062597-DS-7",
#     "10041408-DS-18",
#     "10062981-DS-5",
#     "10041836-DS-21",
#     "10043750-DS-6",
#     "10067059-DS-15",
#     "10067195-DS-13",
#     "10047172-DS-17",
#     "10052938-DS-2",
#     "10052992-DS-11",
#     "10052992-DS-16"
# ]

# Process each note_id
for note_id in note_ids:
    # Load the EHR note and summary content based on note_id
    ehr_note_file = f'data/ehrs/oncology-report-{note_id}.txt'
    summary_file = f'data/summaries/llama3/{note_id}.txt'
    
    with open(ehr_note_file, 'r') as file:
        ehr_note_content = file.read()
    
    with open(summary_file, 'r') as file:
        summary_content = file.read()

    prompt = f"""
    Given below is a TASK OVERVIEW followed by the GUIDELINES, JSON_FORMAT, EHR_NOTE and then finally another piece of text which is called the SUMMARY. You are an annotator, and you have to annotate the EHR_NOTE and SUMMARY based on the GUIDELINES. You will be provided multiple examples of which instances are referred to as hallucinations. The examples will contain the explanation of why a particular instance would be considered hallucinated or not.

    TASK OVERVIEW
    You will be given an EHR note and a piece of text which is supposed to be a summary for the EHR Note. Your task is to check if the summary has any missing or inconsistent information with the EHR note.

    Instructions:

    1. Your task is to provide phrases/words for the below-mentioned kinds of hallucinations. 
        a) Patient Information
        b) Patient History
        c) Symptoms/Diagnosis/Surgical Procedures
        d) Medicine related instructions
        e) Followup
    Each of these hallucinations has two sub-categories: SPECIFIC TO GENERAL and INCORRECT. Please map hallucinations with theor respective sub-categories based on their definitions below:
    SPECIFIC TO GENERAL - Any detail within the clinical note that goes from specific to a more generalized description or if it is an oversimplification of medical events in the summary. 
    INCORRECT - Any detail within the clinical note that is twisted or incorrectly stated in the summary (a discharge instruction stated wrongly). An incorrect condition would also mean that the information was generalized in EHR but was more specific in the summarized content.


    2. Your task is to provide a phrase and a logical explanation for the below-mentioned hallucination categories: 
        a) Chronological Inconsistency
        b) Incorrect Reasoning

    3. Use the GUIDELINES mentioned as follows:
    {guidelines_content}

    4. Please provide your response in the JSON_FORMAT as mentioned in this file:
    {output_format_content}

    Use this EHR_NOTE:
    {ehr_note_content}

    Use this as the SUMMARY:
    {summary_content}
    """
    
    response = use_azureai(prompt) # replace this with use_llama3 method for llama3 detections
    
    # Extract JSON from the response
    json_content = extract_json(response)
    
    if json_content:
        try:
            response_dict = json.loads(json_content)
            # Save the response
            # output_file = f'single_prompts/annotations/gpt4o/{note_id}.json'
            output_file = f'single_prompts/annotations/llama3/{note_id}.json'
            with open(output_file, 'w') as file:
                json.dump(response_dict, file, indent=4)
            print("Done")
        except json.JSONDecodeError:
            print(f"Error decoding JSON for note {note_id}: {json_content}")
    else:
        print(f"No JSON found for note {note_id}: {response}")

