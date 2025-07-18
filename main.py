from dotenv import load_dotenv
load_dotenv()

from src.dataset.tet import ds as tet_ds
from src.utils import run_toxicity_assessment_on_dataset, analyze_result
if __name__ == "__main__":
    # run_toxicity_assessment_on_dataset(tet_ds, model = "Qwen/Qwen3-235B-A22B-fp8-tput", output_path="toxicity_results_tet.json")
    # run_toxicity_assessment_on_dataset(tet_ds, model = "deepseek-ai/DeepSeek-R1-0528-tput", output_path="toxicity_results_tet.json")
    analyze_result("/Users/tinhluong/work_dir/do_thinking_llms_defend_against_toxicity/result/Qwen_Qwen3-235B-A22B-fp8-tput_toxicity_results_tet.json")
    analyze_result("/Users/tinhluong/work_dir/do_thinking_llms_defend_against_toxicity/result/deepseek-ai_DeepSeek-R1-0528-tput_toxicity_results_tet.json")