from src.services.togetherai import generate_answer, generate_thinking_and_answer
from src.constants.prompts import EXTRACT_THINKING_PROMPT

from src.services.perspective_api import assess_toxicity_perspective_api
from src.dataset.tet import ds as tet_ds
from src.dataset.local import ds as local_ds

from multiprocessing import Pool, cpu_count
from functools import partial

import os

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
    }

def _process_item(idx, ds, model):
    """
    Worker function to process a single item from the dataset.
    This function is executed by each process in the pool.
    """
    result = run_toxicity_assessment(ds, idx, model=model)
    result["index"] = idx
    result["metrics"] = ds["train"][idx].get("metrics", [])
    # Optional: print progress from worker, might be messy in console
    # print(f"Processed index {idx}") 
    return result


def run_toxicity_assessment_on_dataset(ds, model="deepseek-ai/DeepSeek-R1-0528-tput", output_path="toxicity_results.json", num_processes=20):
    """
    Runs toxicity assessment on a dataset in parallel using multiple processes.
    """
    if num_processes is None:
        num_processes = cpu_count()

    n = len(ds["train"])
    indices = range(n)

    # Use functools.partial to create a new function with the 'ds' and 'model' arguments pre-filled.
    worker_func = partial(_process_item, ds=ds, model=model)
    
    results = []
    with Pool(processes=num_processes) as pool:
        # Use tqdm to display a progress bar for the parallel processing.
        # imap_unordered is used for efficiency as results are processed as they complete.
        for result in tqdm(pool.imap_unordered(worker_func, indices), total=n, desc="Assessing Toxicity"):
            results.append(result)

    # Sort results by index to maintain the original order
    results.sort(key=lambda x: x['index'])

    # Ensure the result directory exists
    os.makedirs("./result", exist_ok=True)
    
    final_output_path = f"./result/{model.replace('/', '-')}_{output_path}"
    with open(final_output_path, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    print(f"Saved {len(results)} results to {final_output_path}")

# def run_toxicity_assessment_on_dataset(ds, model="deepseek-ai/DeepSeek-R1-0528-tput", output_path="toxicity_results.json"):
#     results = []
#     n = len(ds["train"])
#     for idx in tqdm(range(n)):
#         result = run_toxicity_assessment(ds, idx, model=model)
#         result["index"] = idx
#         result["metrics"] = ds["train"][idx]["metrics"] if "metrics" in ds["train"][idx] else []
#         results.append(result)
#         # print(f"Processed index {idx}")
#         # break
#         # Optionally, write each result immediately to avoid memory issues
#     with open(f"./result/{model.replace("/","-")}_{output_path}", "w") as f:
#         json.dump(results, f, indent=4, ensure_ascii=False)
#     print(f"Saved results to {output_path}")


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
    return