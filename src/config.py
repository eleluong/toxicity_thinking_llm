import os 

together_api_key = os.getenv("TOGETHER_API_KEY", "")

perspective_api_key = os.getenv("PERSPECTIVE_API_KEY", "")

openai_api_key = os.getenv("OPENAI_API_KEY", "")

openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

hf_token = os.getenv("HF_TOKEN", "")

doubao_api_key = os.getenv("DOUBAO_API_KEY", "")
doubao_base_url = os.getenv("DOUBAO_BASE_URL", "https://api.doubao.ai/v1")

selfhost_base_url = os.getenv("SELFHOST_BASE_URL", "http://localhost:8000/v1")
selfhost_api_key = os.getenv("SELFHOST_API_KEY", "")