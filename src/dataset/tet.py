from datasets import load_dataset
from src.config import hf_token
# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset(path = "/Users/tinhluong/work_dir/do_thinking_llms_defend_against_toxicity/src/dataset/local_dataset/tet", token = hf_token)

