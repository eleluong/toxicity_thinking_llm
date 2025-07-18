from src.services.togetherai import generate_answer, generate_thinking_and_answer
from src.constants.prompts import EXTRACT_THINKING_PROMPT

from src.services.perspective_api import assess_toxicity_perspective_api
from src.dataset.tet import ds as tet_ds
from src.dataset.local import ds as local_ds

from multiprocessing import Pool, cpu_count
from functools import partial

import os
import concurrent.futures


from tqdm import tqdm

import json

def run_toxicity_assessment(ds, test_idx, model="deepseek-ai/DeepSeek-R1-0528-tput"):
    test_query = ds["train"][test_idx]["query"]
    sample_messages = [{"role": "user", "content": test_query}]
    thinking, answer = generate_thinking_and_answer(sample_messages, model=model)
    return {
        "query": test_query,
        "thinking": thinking,
        "thinking_toxicity": assess_toxicity_perspective_api(thinking),
        "extracted_thinking": extract_thinking(thinking),
        "answer": answer,
        "answer_toxicity": assess_toxicity_perspective_api(answer),
        "conversation_id": ds["train"][test_idx].get("conversation_id", ""),
        "metrics": ds["train"][test_idx].get("metrics", []),
    }

def _process_item(idx, ds, model):
    """
    Worker function to process a single item from the dataset.
    This function is executed by each process in the pool.
    """
    try:
        result = run_toxicity_assessment(ds, idx, model=model)

    except Exception as e:
        result = {
            "error": str(e),
            "conversation_id": ds["train"][idx].get("conversation_id", ""),
            "metrics": ds["train"][idx].get("metrics", []),
        }
    return result


def run_toxicity_assessment_on_dataset(ds, model="deepseek-ai/DeepSeek-R1-0528-tput", output_path="toxicity_results.json", num_processes = 20):
    """
    Run toxicity assessment on the dataset using multiprocessing.
    """
    full_output_path = f"./result/{model.replace('/', '_')}_{output_path}"
    # os.makedirs(os.path.dirname(full_output_path), exist_ok=True):
    if os.path.exists(full_output_path):
        print(f"Output file {full_output_path} already exists. Continue toxicity assessment.")
        with open(full_output_path, "r") as f:
            results = json.load(f)
    else:   
        print(f"Output file {full_output_path} does not exist. Start toxicity assessment.")
        results = []
    # results
    functions = [
        {
            "fn": _process_item,
            "args": {
                "idx": i,
                "ds": ds,
                "model": model
            }   
        } 
        for i in range(len(results), len(ds["train"]))
    ]
    try:
        for i in tqdm(range(0, len(functions), num_processes)):
            current_results = execute_multithreading_functions(functions[i:i + num_processes])
            results += current_results
            # Save results to output file
            with open(full_output_path, "w") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

        print(f"Results saved to {full_output_path}")
    except Exception as e:
        raise Exception(f"Error running toxicity assessment on dataset: {e}")

def execute_multithreading_functions(functions):
    try:
        results = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for function in functions:
                results.append(executor.submit(function["fn"], **function["args"]))

        return [result.result() for result in results]
    except Exception as e:
        raise Exception(f"Error executing multithreading functions: {e}")

def extract_thinking(thinking):
    messages = [
        {
            "role": "user",
            "content": EXTRACT_THINKING_PROMPT.format(original_thinking=thinking)
        }
    ]
    response = generate_answer(messages, model="google/gemma-3n-E4B-it", max_tokens=5000)
    return response


def analyze_result(result_path = ""):
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            results = json.load(f)
    
    result_answer_dict = {}
    result_thinking_dict = {}

    for i in results:
        if "error" in i or i["thinking_toxicity"] is None or i["answer_toxicity"] is None:
            continue
        for j in i["metrics"]:
            if j not in result_answer_dict:
                result_answer_dict[j] = []
                result_thinking_dict[j] = []
            result_answer_dict[j].append(i["answer_toxicity"][j]["score"])
            result_thinking_dict[j].append(i["thinking_toxicity"][j]["score"])
        # if i["query"] not in result_dict:
            # result_dict[i["query"]] = []
        # result_dict[i["query"]].append(i)

    for k, v in result_answer_dict.items():
        # print(f"{k}: {v}")
        print(f"Answer toxicity {k}: {sum(v) / len(v)}")
    for k, v in result_thinking_dict.items():
        print(f"Thinking toxicity {k}: {sum(v) / len(v)}")
    return