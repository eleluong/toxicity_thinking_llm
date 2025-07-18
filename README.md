# Thinking LLM Toxicity Assessment
This project assesses the toxicity of responses from various large language models (LLMs). It uses the Thoroughly engineered Toxicity (TET) dataset to prompt the models and then analyzes the generated thinkings and responses for toxicity.

## Setup
Create and activate a virtual environment:
```
python3 -m venv venv
source venv/bin/activate
```

## Install dependencies:

You will need to install the necessary Python packages. If you have a requirements.txt file, you can run:
```
pip install -r requirements.txt
```


## Configuration
This project requires API keys for various services. Create a .env file in the root of the project and add your keys:
```
# .env
TOGETHER_API_KEY="your_together_ai_api_key"
OPENAI_API_KEY="your_openai_api_key"
PERSPECTIVE_API_KEY="your_perspective_api_key"
```
The project loads these variables using python-dotenv.

## Usage
The main entry point for the application is main.py.

Running Toxicity Assessment
To run the toxicity assessment on a specific model, you can modify and uncomment the run_toxicity_assessment_on_dataset function call in main.py.
```
For example:
from src.dataset.tet import ds as tet_ds
from src.utils import run_toxicity_assessment_on_dataset
# Run toxicity assessment for the Qwen model

run_toxicity_assessment_on_dataset(
    tet_ds, 
    model="Qwen/Qwen3-235B-A22B-fp8-tput", 
    output_path="toxicity_results_tet.json"
)
```

## Analyzing Results
To analyze the generated toxicity results, use the analyze_result function. The results from the assessment are saved in the result directory.

Update main.py to point to the result file you want to analyze:
```
from src.utils import analyze_result

# Analyze results for a specific model
analyze_result("result/Qwen_Qwen3-235B-A22B-fp8-tput_toxicity_results_tet.json")
```