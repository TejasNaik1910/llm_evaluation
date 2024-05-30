# LLM Detection 

The project analyzes LLM capabilities to detect hallucinations in the given summary. 

## Folder Structure

```
LLM_DETECTION/
│
├── data/
│   ├── notes/
│   └── summaries/
│
├── multiple_prompts/
│   ├── guidelines/
│   ├── op_files/
│   ├── llm-annotated-gpt4o-10001401-DS-20_all.json
│   ├── multiple_prompts.py
│   └── output_format.json
│
├── single_prompt/
│   ├── guidelines.txt
│   ├── llm-annotated-gpt4o-10001401-DS-20_old.json
│   ├── llm-annotated-gpt4o-10001401-DS-20.json
│   ├── output_format.json
│   └── single_prompt.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

- data: Contains `notes` and `summaries` subfolders with additional information and summary files used in the detection.
- multiple_prompts: Contains scripts and data for evaluating multiple prompts.
  - `guidelines`: Folder with individual guidelines for the detection process.
  - `op_files`: Output format for individual guieline files.
  - `llm-annotated-gpt4o-10001401-DS-20_all.json`: JSON file with annotated responses for multiple prompts.
  - `multiple_prompts.py`: Python script for processing multiple prompts.
  - `output_format.json`: JSON file defining the output format for the annotations.
- single_prompt: Contains scripts and data for evaluating single prompts.
  - `guidelines.txt`: Text file with guidelines for the detection process.
  - `llm-annotated-gpt4o-10001401-DS-20_old.json`: JSON file with old annotated responses for single prompts.
  - `llm-annotated-gpt4o-10001401-DS-20.json`: JSON file with current annotated responses for single prompts.
  - `output_format.json`: JSON file defining the output format for the annotations.
  - `single_prompt.py`: Python script for processing single prompts.
- .gitignore: Specifies files to be ignored by git.
- README.md: Current file.
- requirements.txt: Lists the Python dependencies for the project.

## Installation Instructions

1. Clone the repository:
   git clone <repository_url>
   cd LLM_DETECTION

2. Create and activate a virtual environment:
   python3 -m venv myenv
   source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`

3. Install the required dependencies:
   pip install -r requirements.txt

NOTE: STEP 2 AND STEP 3 ARE REQUIRED IF OPENAI MODULE IS NOT INSTALLED ON YOUR LOCAL MACHINE.


## Usage Instructions

### Multiple Prompts Detection

1. Navigate to the `multiple_prompts` folder:
   cd multiple_prompts

2. Run the `multiple_prompts.py` script:
   python multiple_prompts.py

3. Creates a output JSON files within the `multiple_prompts` folder.


### Single Prompt Detection

1. Navigate to the `single_prompt` folder:
   cd single_prompt

2. Run the `single_prompt.py` script:
   python single_prompt.py

3. Creates a output JSON files within the `single_prompt` folder.

## Data Description

- **llm-annotated-gpt4o-10001401-DS-20_all.json**: Contains annotations for multiple prompt responses in JSON format.
- **llm-annotated-gpt4o-10001401-DS-20.json**: Contains annotations for single prompt responses combined in a single JSON file.
- **output_format.json**: Defines the structure of the output annotations.