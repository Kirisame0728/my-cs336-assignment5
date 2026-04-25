import os
import json
from vllm import LLM, SamplingParams
from typing import Callable, Dict, List, Optional, Any, Tuple
from transformers import PreTrainedTokenizerBase


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)

    summary = {
        "num_examples": n,
        "avg_format_reward": sum(r["format_reward"] for r in results) / n,
        "avg_answer_reward": sum(r["answer_reward"] for r in results) / n,
        "avg_reward": sum(r["reward"] for r in results) / n,
        "accuracy": sum(r["answer_reward"] for r in results) / n,
        "format_1_answer_1": sum(
            r["format_reward"] == 1.0 and r["answer_reward"] == 1.0 for r in results
        ),
        "format_1_answer_0": sum(
            r["format_reward"] == 1.0 and r["answer_reward"] == 0.0 for r in results
        ),
        "format_0_answer_0": sum(
            r["format_reward"] == 0.0 and r["answer_reward"] == 0.0 for r in results
        ),
        "avg_length": sum(r["response_length"] for r in results) / n,
    }

    lengths = [r["response_length"] for r in results]
    correct_lengths = [r["response_length"] for r in results if r["answer_reward"] == 1.0]
    incorrect_lengths = [r["response_length"] for r in results if r["answer_reward"] == 0.0]
    summary["avg_response_length"] = sum(lengths) / len(lengths)
    if correct_lengths:
        summary["avg_response_length_correct"] = sum(correct_lengths) / len(correct_lengths)
    if incorrect_lengths:
        summary["avg_response_length_incorrect"] = sum(incorrect_lengths) / len(incorrect_lengths)

    return summary


def log_generations(
        vllm_model: LLM,
        reward_fn: Callable[[str, str], Dict[str, float]],
        prompts: List[str],
        ground_truths: List[str],
        eval_sample_params: SamplingParams,
        tokenizer: PreTrainedTokenizerBase,
        step: Optional[int] = None,
        split: str = "validation",
        results_path: Optional[str] = None,
        summary_path: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    outputs = vllm_model.generate(prompts, eval_sample_params)
    responses = [o.outputs[0].text for o in outputs]

    results = []
    for output, response, ground_truth in zip(outputs, responses, ground_truths):
        reward_info = reward_fn(response, ground_truth)
        response_length = len(tokenizer(response, add_special_tokens=False)["input_ids"])
        row = {
            "step": step,
            "split": split,
            "prompt": output.prompt,
            "ground_truth": ground_truth,
            "model_output": response,
            "format_reward": float(reward_info.get("format_reward")),
            "answer_reward": float(reward_info.get("answer_reward")),
            "reward": float(reward_info.get("reward", 0.0)),
            "response_length": response_length
        }

        results.append(row)

    summary = summarize_results(results)
    summary["step"] = step
    summary["split"] = split

    if results_path is not None:
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if summary_path is not None:
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


    return results, summary




