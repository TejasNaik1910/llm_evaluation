# LLM Evaluation 

This project analyzes LLM capabilities to detect hallucinations in the given summary. 

- data: Contains `ehrs` and `summaries` subfolders with ehr notes and summary files used for the detection.
- single_prompts: Contains scripts and files for LLM detection using single prompts.
  - `guidelines.txt`: Text file with guidelines for the detection process.
  - `output_format.json`: JSON file defining the output format for the annotations.
  - `gpt4o_detections.py`: Python script for processing single prompts using gpt4o.
  - `llama3_detections.py`: Python script for processing single prompts using llama3.
- .gitignore: Specifies files to be ignored by git.
- requirements.txt: Lists the Python dependencies for the project.

## Installation Instructions

1. Clone the repository:\
   git clone https://github.com/TejasNaik1910/llm_evaluation.git \
   cd llm_evaluation

2. Create and activate a virtual environment:\
   python3 -m venv myenv\
   source myenv/bin/activate  # On Windows, use `myenv\Scripts\activate`

3. Install the required dependencies:
   pip install -r requirements.txt

NOTE: STEP 2 AND STEP 3 ARE ONLY REQUIRED IF OPENAI MODULE IS NOT INSTALLED ON YOUR MACHINE.

## Usage Instructions

### Single Prompt Detection

1. Navigate to the `single_prompts` folder: \
   `cd single_prompts`

2. Before executing the scripts, please make the necessary code changes in them.

3. Next, run the `{model_name}_detections.py` script:\
   `python3 {model_name}_detections.py`

4. It will create the output JSON files within the `single_prompts/annotations/{model_name}` folder.