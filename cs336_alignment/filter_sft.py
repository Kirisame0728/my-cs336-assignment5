import os
import json
from typing import List, Dict, Any

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


TRAIN_PATH = "data/a5-alignment/MATH/train.jsonl"
INPUT_SFT_PATH = "data/a5-alignment/MATH/sft.jsonl"
OUTPUT_SFT_PATH = "data/a5-alignment/MATH/sft_filtered.jsonl"

R1_ZERO_PROMPT = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_r1_zero_prompt(question: str) -> str:
    return R1_ZERO_PROMPT.format(question=question.strip())


def build_prompt_to_ground_truth(train_rows: List[Dict[str, Any]]) -> Dict[str, str]:
    mapping = {}
    for row in train_rows:
        prompt = format_r1_zero_prompt(row["question"])
        mapping[prompt] = row["ground_truth"]
    return mapping


def main() -> None:
    train_rows = load_jsonl(TRAIN_PATH)
    sft_rows = load_jsonl(INPUT_SFT_PATH)

    prompt_to_ground_truth = build_prompt_to_ground_truth(train_rows)

    kept_rows = []

    total = 0
    kept = 0
    skipped_missing_ground_truth = 0

    for row in sft_rows:
        total += 1

        prompt = row["prompt"]
        response = row["response"]

        ground_truth = prompt_to_ground_truth.get(prompt)
        if ground_truth is None:
            skipped_missing_ground_truth += 1
            continue

        reward_info = r1_zero_reward_fn(response, ground_truth)
        format_reward = float(reward_info.get("format_reward", 0.0))
        answer_reward = float(reward_info.get("answer_reward", 0.0))

        if format_reward == 1.0 and answer_reward == 1.0:
            kept_rows.append(row)
            kept += 1

    write_jsonl(OUTPUT_SFT_PATH, kept_rows)

    print("Done.")
    print(f"Input rows: {total}")
    print(f"Kept rows: {kept}")
    print(f"Skipped missing ground truth: {skipped_missing_ground_truth}")
    print(f"Wrote filtered SFT to: {OUTPUT_SFT_PATH}")


if __name__ == "__main__":
    main()