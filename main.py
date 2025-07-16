from dotenv import load_dotenv
load_dotenv()

from src.dataset.tet import ds as tet_ds
from src.utils import run_toxicity_assessment_on_dataset
if __name__ == "__main__":
    run_toxicity_assessment_on_dataset(tet_ds, model = "Qwen/Qwen3-235B-A22B-fp8-tput")