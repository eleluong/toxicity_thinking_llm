from openai import OpenAI
from src.config import together_api_key
client = OpenAI(
    base_url = "https://api.together.xyz/v1",
    api_key = together_api_key
)

def extract_thinking(text_response):
    start_ = "<think>\n"
    end_ = "</think>\n"
    thinking = text_response[text_response.index(start_): text_response.index(end_)].strip(end_).strip(start_)
    answer = text_response[text_response.index(end_):].strip(end_)
    return thinking, answer

def generate_thinking_and_answer(
        messages: list = [],
        model = "Qwen/Qwen3-235B-A22B-fp8-tput",
        max_tokens = 5000
):

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        stream=False
    )
    return extract_thinking(response.choices[0].message.content)


def generate_answer(
        messages: list = [],
        model = "google/gemma-3n-E4B-it",
        max_tokens = 5000
):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        stream=False
    )
    return response.choices[0].message.content

