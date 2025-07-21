from dotenv import load_dotenv
load_dotenv()

from src.dataset.tet import ds as tet_ds
from src.utils import run_toxicity_assessment_on_dataset, analyze_result

from src.services.openai_ import generate_thinking_and_answer, client

if __name__ == "__main__":
    # run_toxicity_assessment_on_dataset(tet_ds, model = "Qwen/Qwen3-235B-A22B-fp8-tput", output_path="toxicity_results_tet.json")
    # run_toxicity_assessment_on_dataset(tet_ds, model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", output_path="toxicity_results_tet.json")

    # run_toxicity_assessment_on_dataset(tet_ds, model = "deepseek-ai/DeepSeek-R1-0528-tput", output_path="toxicity_results_tet.json")

    # run_toxicity_assessment_on_dataset(tet_ds, model = "claude-sonnet-4-20250514-thinking", output_path="toxicity_results_tet.json")
    run_toxicity_assessment_on_dataset(tet_ds, model = "doubao-1-5-thinking-pro-250415", output_path="toxicity_results_tet.json")



    # run_toxicity_assessment_on_dataset(tet_ds, model = "o4-mini-2025-04-16", output_path="toxicity_results_tet.json")

    # analyze_result("/Users/tinhluong/work_dir/do_thinking_llms_defend_against_toxicity/result/Qwen_Qwen3-235B-A22B-fp8-tput_toxicity_results_tet.json")
    # analyze_result("/Users/tinhluong/work_dir/do_thinking_llms_defend_against_toxicity/result/deepseek-ai_DeepSeek-R1-0528-tput_toxicity_results_tet.json")
    # analyze_result("/Users/tinhluong/work_dir/do_thinking_llms_defend_against_toxicity/result/deepseek-ai_DeepSeek-R1-Distill-Qwen-14B_toxicity_results_tet.json")
    # analyze_result("/Users/tinhluong/work_dir/do_thinking_llms_defend_against_toxicity/result/o4-mini-2025-04-16_toxicity_results_tet.json")
