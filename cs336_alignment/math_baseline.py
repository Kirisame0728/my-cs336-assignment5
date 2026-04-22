import json
from vllm import LLM, SamplingParams
from typing import Callable, List, Dict, Any
import os
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

R1_ZERO_PROMPT = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""

def format_r1_prompt(question):
    question = question.strip()
    return R1_ZERO_PROMPT.format(question=question)

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            item = json.loads(line)
            data.append(item)

    return data

def batched_format(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    if n == 0:
        raise ValueError("No results to summarize")

    n_correct_both = 0
    n_format_only = 0
    n_both_wrong = 0

    sum_format_reward = 0.0
    sum_answer_reward = 0.0
    sum_reward = 0.0

    for r in results:
        fr = r["format_reward"]
        ar = r["answer_reward"]
        rr = r["reward"]

        sum_format_reward += fr
        sum_answer_reward += ar
        sum_reward += rr

        if fr == 1.0 and ar == 1.0:
            n_correct_both += 1
        elif fr == 1.0 and ar == 0.0:
            n_format_only += 1
        elif fr == 0.0 and ar == 0.0:
            n_both_wrong += 1

    summary = {
        "num_examples": n,
        "avg_format_reward": sum_format_reward / n,
        "avg_answer_reward": sum_answer_reward / n,
        "avg_reward": sum_reward / n,
        "accuracy": sum_answer_reward / n,  # baseline math accuracy
        "count_format_1_answer_1": n_correct_both,
        "count_format_1_answer_0": n_format_only,
        "count_format_0_answer_0": n_both_wrong,
    }
    return summary

def evaluate_vllm(
        vllm_model: LLM,
        reward_fn: Callable[[str,str], dict[str, float]],
        prompts: List[str],
        ground_truths: List[str],
        eval_sampling_params: SamplingParams,
        results_path: str | None = None,
        summary_path: str | None = None,
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics,and serialize results to disk.
    """
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    results = []
    for output, ground_truth in zip(outputs, ground_truths):
        prompt = output.prompt
        generated_text = output.outputs[0].text

        reward_info = reward_fn(generated_text, ground_truth)

        result = {
            "prompt": prompt,
            "ground_truth": ground_truth,
            "model_output": generated_text,
            "format_reward": float(reward_info.get("format_reward")),
            "answer_reward": float(reward_info.get("answer_reward")),
            "reward": float(reward_info.get("reward")),
        }
        results.append(result)
    summary = summarize_results(results)
    with open(results_path, "w", encoding="utf8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    with open(summary_path, "w", encoding="utf8") as f:
        f.write(json.dumps(summary, indent=2))

def main():
    validation_path = "data/a5-alignment/MATH/validation.jsonl"
    output_dir = "outputs/math_baseline"
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "results.jsonl")
    summary_path = os.path.join(output_dir, "summary.json")

    examples = load_jsonl(validation_path)
    prompts = [format_r1_prompt(example["question"]) for example in examples]
    ground_truths = [example["ground_truth"] for example in examples]

    model_name = "Qwen/Qwen2.5-Math-1.5B"

    vllm_model = LLM(model_name, trust_remote_code=True, dtype="bfloat16",)
    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True)

    evaluate_vllm(
        vllm_model=vllm_model,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampling_params,
        results_path=results_path,
        summary_path=summary_path
    )

if __name__ == "__main__":
    main()
