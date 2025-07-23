from openai import OpenAI
from src.config import selfhost_api_key, selfhost_base_url

client = OpenAI(
    base_url=selfhost_base_url,
    api_key = selfhost_api_key
)

def generate_thinking_and_answer(
        input: str = "Tell me a three sentence bedtime story about a unicorn.",
        model = "microsoft/Phi-4-reasoning",
        max_tokens = 4000
):

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": input}
        ],
        max_tokens = max_tokens
    )

    return response.choices[0].message.reasoning_content, response.choices[0].message.content


def generate_with_budget_forcing(
    input = "Tell me a three sentence bedtime story about a unicorn.",
    model="Qwen/Qwen3-4B",
    max_tokens_thinking=2000,
    num_ignore=1,
    max_final_tokens=1000
):
    """
    Generate responses using budget forcing technique with OpenAI API
    
    Args:
        prompts: List of user prompts to process
        model_name: OpenAI model to use (default: "gpt-3.5-turbo")
        system_message: System role message (default: Qwen system message)
        max_tokens_thinking: Max tokens per thinking step (default: 200)
        num_ignore: Number of times to force continued thinking (default: 1)
        max_final_tokens: Max tokens for final answer (default: 100)
    
    Returns:
        List of generated responses
    """
    
    # Construct initial prompt
    prompt_str = (
        f"<|im_start|>user\n{input}<|im_end|>\n"
        f"<|im_start|>assistant\n<|im_start|>\n"
        f"<think>\n"
    )
    
    # Thinking steps
    current_prompt = prompt_str
    reasoning = ""
    for step in range(num_ignore + 1):  # +1 for initial thinking step
        response = client.completions.create(
            model=model,
            prompt = current_prompt,
            max_tokens=max_tokens_thinking,
            stop=["</think>"],
            temperature=0.0,
        )
        # print(response)
        generated_text = response.choices[0].text
        reasoning += generated_text
        current_prompt += generated_text
        
        # Force continuation if not last step
        if step < num_ignore:
            current_prompt += "\nWait"
            reasoning += "\nWait"
    
    # Final answer generation
    response_final = client.completions.create(
        model=model,
        prompt = current_prompt + "\n</think>\n",
        max_tokens=max_final_tokens,
        stop=["<|im_end|>"],
        temperature=0.0,
    )
    final_text = response_final.choices[0].text
    
    return reasoning, final_text

