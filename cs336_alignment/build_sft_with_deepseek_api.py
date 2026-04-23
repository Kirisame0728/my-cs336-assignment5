import os
import re
import json
import time
from typing import List, Dict, Any, Tuple

from openai import OpenAI


INPUT_PATH = "data/a5-alignment/MATH/train.jsonl"
OUTPUT_PATH = "data/a5-alignment/MATH/sft.jsonl"

MODEL_NAME = "deepseek-r1"

MAX_EXAMPLES = 1024
SLEEP_SECONDS = 1.0
PRINT_EVERY = 4
MAX_RETRIES = 6

R1_ZERO_PROMPT = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_r1_zero_prompt(question: str) -> str:
    return R1_ZERO_PROMPT.format(question=question.strip())


def build_client() -> OpenAI:
    api_key = os.environ["DASHSCOPE_API_KEY"]
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def generate_reasoning_and_answer(
    client: OpenAI,
    prompt: str,
    max_retries: int = MAX_RETRIES,
) -> Tuple[str, str]:
    last_err = None

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
            message = resp.choices[0].message
            reasoning = getattr(message, "reasoning_content", "") or ""
            answer = message.content or ""
            return reasoning.strip(), answer.strip()

        except Exception as e:
            last_err = e
            err_str = str(e)

            retryable = (
                "Too many requests" in err_str
                or "throttled" in err_str
                or "ServiceUnavailable" in err_str
                or "InternalError.Algo" in err_str
                or "503" in err_str
                or "500" in err_str
            )

            if not retryable:
                raise

            sleep_seconds = min(60, 2 ** attempt)
            print(f"[retry {attempt + 1}/{max_retries}] sleeping {sleep_seconds}s due to: {e}")
            time.sleep(sleep_seconds)

    raise last_err


def extract_final_answer_from_raw_answer(answer: str) -> str:
    answer = answer.strip()
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", answer, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    return answer


def build_response_text(reasoning: str, raw_answer: str) -> str:
    final_answer = extract_final_answer_from_raw_answer(raw_answer)
    # 注意：grader 要求这里必须是 </think> <answer>，中间有空格
    return f"{reasoning}</think> <answer>{final_answer}</answer>"


def get_completed_prompt_set(path: str) -> set[str]:
    existing_rows = load_jsonl(path)
    completed = set()
    for row in existing_rows:
        prompt = row.get("prompt")
        if isinstance(prompt, str):
            completed.add(prompt)
    return completed


def main() -> None:
    client = build_client()
    examples = load_jsonl(INPUT_PATH)

    if MAX_EXAMPLES is not None:
        examples = examples[:MAX_EXAMPLES]

    completed_prompts = get_completed_prompt_set(OUTPUT_PATH)
    total_existing = len(completed_prompts)

    print(f"Found {total_existing} existing rows in {OUTPUT_PATH}")

    generated_this_run = 0
    skipped_existing = 0
    errors = 0

    for idx, ex in enumerate(examples, start=1):
        question = ex["question"]
        prompt = format_r1_zero_prompt(question)

        if prompt in completed_prompts:
            skipped_existing += 1
            if idx % PRINT_EVERY == 0:
                print(
                    f"[{idx}/{len(examples)}] "
                    f"existing={skipped_existing} generated_this_run={generated_this_run} errors={errors}"
                )
            continue

        try:
            reasoning, raw_answer = generate_reasoning_and_answer(client, prompt)
            response = build_response_text(reasoning, raw_answer)

            row = {
                "prompt": prompt,
                "response": response,
            }

            # 关键：每成功一条就立刻写入
            append_jsonl(OUTPUT_PATH, row)
            completed_prompts.add(prompt)
            generated_this_run += 1

        except Exception as e:
            errors += 1
            print(f"[{idx}] ERROR: {e}")

        if idx % PRINT_EVERY == 0:
            print(
                f"[{idx}/{len(examples)}] "
                f"existing={skipped_existing} generated_this_run={generated_this_run} errors={errors}"
            )

        if SLEEP_SECONDS > 0:
            time.sleep(SLEEP_SECONDS)

    print("\nDone.")
    print(f"Model: {MODEL_NAME}")
    print(f"Input examples considered: {len(examples)}")
    print(f"Skipped existing: {skipped_existing}")
    print(f"Generated this run: {generated_this_run}")
    print(f"Errors: {errors}")
    print(f"Output file: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()