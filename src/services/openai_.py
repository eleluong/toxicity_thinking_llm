from openai import OpenAI
from src.config import openai_api_key, openai_base_url
client = OpenAI(
    base_url=openai_base_url,
    api_key = openai_api_key
)


def generate_thinking_and_answer(
        input: str = "Tell me a three sentence bedtime story about a unicorn.",
        model = "o4-mini-2025-04-16",
):

    response = client.responses.create(
        model=model,
        input=input,
        reasoning={
            "effort": "medium",
            "summary": "detailed"
        }
    )

    try:
        if response.output[0].summary == []:
            return "", response.output[1].content[0].text
        else:
            return response.output[0].summary[0].text, response.output[1].content[0].text
            
    except Exception as e:
        print(f"Error processing response: {e}")
        return "", response.output[1].content[0].text


