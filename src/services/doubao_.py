from openai import OpenAI
from src.config import doubao_api_key, doubao_base_url
client = OpenAI(
    base_url=doubao_base_url,
    api_key = doubao_api_key
)


def generate_thinking_and_answer(
        input: str = "Tell me a three sentence bedtime story about a unicorn.",
        model = "o4-mini-2025-04-16",
):

    response = client.chat.completions.create(
        model="doubao-1-5-thinking-pro-250415",
        messages=[
            {"role": "user", "content": input}
        ],
    )
    return response.choices[0].message.reasoning_content, response.choices[0].message.content
    # try:
    #     if response.output[0].summary == []:
    #         return "", response.output[1].content[0].text
    #     else:
    #         return response.output[0].summary[0].text, response.output[1].content[0].text
            
    # except Exception as e:
    #     print(f"Error processing response: {e}")
    #     return "", response.output[1].content[0].text


