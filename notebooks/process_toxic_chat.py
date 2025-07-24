import os
import json
import random
import concurrent.futures
from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm

def extract_thinking(text_response):
    """Extracts the thinking and answer parts from a model's response."""
    try:
        start_ = "<think>\n"
        end_ = "</think>\n"
        start_index = text_response.index(start_)
        end_index = text_response.index(end_)
        thinking = text_response[start_index + len(start_):end_index]
        answer = text_response[end_index + len(end_):].strip()
        return thinking, answer
    except ValueError:
        # If the tags are not found, return the whole response as the answer.
        return "", text_response

def generate_thinking_and_answer(client, messages: list, model: str, max_tokens: int = 5000):
    """Generates a response from the model and extracts thinking and answer."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        stream=False
    )
    return extract_thinking(response.choices[0].message.content)

def execute_multithreading_functions(functions):
    """Executes a list of functions in parallel using a thread pool."""
    try:
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_func = {executor.submit(f["fn"], **f["args"]): f for f in functions}
            for future in concurrent.futures.as_completed(future_to_func):
                results.append(future.result())
        return results
    except Exception as e:
        raise Exception(f"Error executing multithreading functions: {e}")

def _process_item(item, model, client):
    """Worker function to process a single item from the dataset."""
    try:
        messages = [{"role": "user", "content": item["query"]}]
        thinking, answer = generate_thinking_and_answer(client=client, messages=messages, model=model)
        return {
            "conversation_id": item["conversation_id"],
            "query": item["query"],
            "thinking": thinking,
            "answer": answer,
            "model": model
        }
    except Exception as e:
        return {
            "error": str(e),
            "conversation_id": item["conversation_id"],
        }

def synthesize_toxic_chat_data(output_path: str = "./synthesis_toxic_chat.json", num_processes: int = 5):
    """
    Loads the toxic-chat dataset, filters for potentially toxic or interesting samples,
    generates new responses using specified models, and saves the results.
    """
    # Initialize OpenAI client
    # client = OpenAI(
    #     base_url="https://api.together.xyz/v1",
    #     api_key=api_key
    # )

    # Load dataset
    print("Loading dataset...")
    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124")
    
    # Filter failed samples
    print("Filtering samples...")
    failed_samples = []
    train_data = ds["train"]
    for i in range(len(train_data)):
        flag = False
        # Flag if OpenAI moderation score is high
        for item in eval(train_data[i]["openai_moderation"]):
            if item[1] > 0.05:
                flag = True
                break
        # Flag if human-annotated toxicity is present
        if train_data[i]["toxicity"] == 1:
            flag = True
        
        if flag:
            failed_samples.append({
                "conversation_id": train_data[i]["conv_id"],
                "query": train_data[i]["user_input"]
            })
        # Also include a random subset of non-flagged samples
        elif random.random() > 0.9:
            failed_samples.append({
                "conversation_id": train_data[i]["conv_id"],
                "query": train_data[i]["user_input"]
            })

    print(f"Found {len(failed_samples)} samples to process.")

    # Prepare functions for multithreading
    models = ["deepseek-ai/DeepSeek-R1-0528-tput", "Qwen/Qwen3-235B-A22B-fp8-tput"]
    functions = [
        {
            "fn": _process_item,
            "args": {
                "item": sample,
                "model": random.choice(models),
            }
        } for sample in failed_samples
    ]

    # Process in batches
    results = []
    print(f"Processing functions in batches of {num_processes}...")
    try:
        for i in tqdm(range(0, len(functions), num_processes), desc="Processing Batches"):
            batch_functions = functions[i:i + num_processes]
            current_results = execute_multithreading_functions(batch_functions)
            results.extend(current_results)
            
            # Save intermediate results
            with open(output_path, "w", encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

        print(f"Processing complete. Results saved to {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"Partial results saved to {output_path}")

if __name__ == '__main__':
    # Example usage:
    # Make sure to set the TOGETHER_API_KEY environment variable
    together_api_key = os.getenv("TOGETHER_API_KEY")
    if not together_api_key:
        raise ValueError("Please set the TOGETHER_API_KEY environment variable.")
    
    synthesize_toxic_chat_data(api_key=together_api_key)