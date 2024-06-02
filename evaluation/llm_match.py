import os
import json
from openai import AzureOpenAI # type: ignore

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

# note_ids = ["10001401-DS-20", "10054464-DS-17", 
#     "10002221-DS-12", "10003299-DS-10", "10056223-DS-14", "10004401-DS-26", 
#     "10056612-DS-8", "10006029-DS-16", "10006431-DS-24", "10006580-DS-21", 
#     "10006820-DS-18", "10007795-DS-13", "10008628-DS-3", "10010440-DS-5", 
#     "10011938-DS-16", "10012292-DS-9", "10014354-DS-23", "10016142-DS-19", 
#     "10017285-DS-3", "10018052-DS-16", "10057126-DS-7", "10020306-DS-9"]

note_ids= ["10001401-DS-20", "10003299-DS-10", "10006029-DS-16", "10006431-DS-24", "10006820-DS-18", "10007795-DS-13"]

def compute(prompt_meth, note_id):
    with open(f"""../data/processed-human-annotations/annotations-gpt4o-{note_id}.json""", 'r') as file:
            human_json = json.load(file)

    with open(f"""../{prompt_meth}_prompts/{prompt_meth}-prompts-annotations/llm-annotated-gpt4o-{note_id}-{prompt_meth}.json""", 'r') as file:
            llm_json = json.load(file)

    def compare(human_text,llm_text):

        prompt = f"""Please compare these 2 texts and if they have the same meaning, give the response else give 0
        Text 1:
        {human_text}

        Text 2:
        {llm_text}

        Please respond in binary values only."""
        response = use_azureai(prompt)
        return response

    def llm_match(human_json, llm_json):
        results = {}
        num_human = {}
        num_llm= {}
        for key in human_json:
            human_entries = human_json[key]
            llm_entries = llm_json.get(key, [])
            h=len(human_entries)
            l=len(llm_entries)
            if llm_entries==[]or llm_entries==[{'text': ''}]:
                llm_entries=[{'text': '',"omittedDetails": ''}]
                l=0
            if human_entries==[] or human_entries==[{'text': ''}]:
                human_entries=[{'text': '',"omittedDetails": ''}]
                h=0
            if key.startswith("Incorrect"):
                results[key] = count_matches(human_entries, llm_entries, 'text')
            elif key.startswith("Omitted"):
                results[key] = count_matches(human_entries, llm_entries, 'omittedDetails')
            else:
                results[key] = count_matches(human_entries, llm_entries, 'text')
            num_human[key]=h
            num_llm[key]=l
        return results,num_human,num_llm

    def count_matches(human_entries, llm_entries, key_to_compare):
        matches = 0
        for human_item in human_entries:
            for llm_item in llm_entries:
                human_text = human_item.get(key_to_compare, '')
                llm_text = llm_item.get(key_to_compare, '')
                if human_text=='' or llm_text=='':
                    matches=0
                else:
                    response = compare(human_text, llm_text)
                    if response == "1":
                        matches += 1
        return matches
    
    return llm_match(human_json, llm_json)


tot_mat={}
tot_human={}
tot_llm={}
     
for note_id in note_ids:
    results, num_human, num_llm = compute("single",note_id) # change to "multiple" for other prompting method
    for key in results:
        if key not in tot_mat:
            tot_mat[key] = 0
            tot_human[key] = 0
            tot_llm[key] = 0
        tot_mat[key] += results[key]
        tot_human[key] += num_human[key]
        tot_llm[key] += num_llm[key]
    print("Done: ",note_id)

recall = {}
for key in tot_mat:
    if tot_human[key] > 0:
        recall[key] = tot_mat[key] / tot_human[key]
    elif tot_human[key] == 0:
        recall[key] = 0

precision = {}
for key in tot_mat:
    if tot_llm[key] > 0:
        precision[key] = tot_mat[key] / tot_llm[key]
    elif tot_llm[key] == 0:
        precision[key] = 0

print("Total Matches: ", tot_mat)
print("Total Human Entries: ", tot_human)
print("Total LLM Entries: ", tot_llm)

print("Precision: ", precision)
print("Recall: ", recall)