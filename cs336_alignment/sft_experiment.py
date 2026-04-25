import json
import wandb
from tqdm import tqdm
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from vllm import LLM, SamplingParams
import torch
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
from cs336_alignment.tokenize_prompt_and_output import tokenize_prompt_and_output
from cs336_alignment.get_response_log_probs import get_response_log_probs
from cs336_alignment.sft_microbatch_train_step import sft_microbatch_train_step
from cs336_alignment.log_generations import log_generations
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from unittest.mock import patch


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )

    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype="bfloat16",
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

class SFTDataset(Dataset):
    def __init__(self, path, num_examples=None):
        self.examples = []
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                item = json.loads(line)
                self.examples.append({
                    "prompt": item["prompt"],
                    "response": item["response"]
                })
                if num_examples is not None and len(self.examples) >= num_examples:
                    break


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

def build_sft_dataloader(path, batch_size, num_examples, shuffle=True):
    dataset = SFTDataset(path, num_examples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_one_epoch(
        model,
        tokenizer,
        train_loader,
        optimizer,
        device,
        gradient_accumulation_steps,
        normalize_constant=1.0,
        max_grad_norm=1.0
):
    model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    total_num_response_tokens = 0.0
    total_mean_log_prob = 0.0
    num_batches = 0
    num_optimizer_steps = 0

    for idx, batch in enumerate(train_loader):
        prompts = batch["prompt"]
        responses = batch["response"]
        tokenized = tokenize_prompt_and_output(prompts, responses, tokenizer)

        input_ids = tokenized["input_ids"].to(device)
        labels = tokenized["labels"].to(device)
        response_mask = tokenized["response_mask"].to(device)

        log_probs = get_response_log_probs(model, input_ids, labels)["log_probs"]

        loss, metadata = sft_microbatch_train_step(log_probs, response_mask, gradient_accumulation_steps, normalize_constant)

        total_loss += float(loss.detach().cpu())
        total_num_response_tokens += float(metadata["num_response_tokens"].detach().cpu())
        total_mean_log_prob += metadata["mean_log_prob"].detach().float().reshape(-1).mean().item()
        num_batches += 1

        should_step = ((idx + 1) % gradient_accumulation_steps == 0)
        is_last_batch = (idx + 1 == len(train_loader))

        if should_step or is_last_batch:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            num_optimizer_steps += 1


    return {
        "avg_loss": total_loss / num_batches if num_batches > 0 else 0.0,
        "avg_num_response_tokens": (
            total_num_response_tokens / num_batches if num_batches > 0 else 0.0
        ),
        "avg_mean_log_prob": (
            total_mean_log_prob / num_batches if num_batches > 0 else 0.0
        ),
        "num_batches": num_batches,
        "num_optimizer_steps": num_optimizer_steps,
    }

R1_ZERO_PROMPT = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""

def format_r1_prompt(question):
    question = question.strip()
    return R1_ZERO_PROMPT.format(question=question)

def load_validation_jsonl(path, max_examples=None):
    prompts = []
    ground_truths = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            question = item["question"]
            prompts.append(format_r1_prompt(question))
            ground_truths.append(item["ground_truth"])
            if max_examples is not None and len(prompts) >= max_examples:
                break

    return prompts, ground_truths

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--train_path", default="data/a5-alignment/MATH/sft.jsonl")
    parser.add_argument("--val_path", default="data/a5-alignment/MATH/validation.jsonl")
    parser.add_argument("--output_dir", default="outputs/sft_run")
    parser.add_argument("--model_id", default="data/Qwen2.5-Math-1.5B")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--num_train_examples", type=int, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    train_device = "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
    ).to(train_device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    train_loader = build_sft_dataloader(args.train_path, args.batch_size, args.num_train_examples)
    val_prompts, val_ground_truths = load_validation_jsonl(args.val_path, 16)

    vllm_device = "cuda:1"
    vllm_model = init_vllm(
        model_id=args.model_id,
        device=vllm_device,
        seed=42,
        gpu_memory_utilization=0.85,
    )

    eval_sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    wandb.init(
        project="cs336-a5-sft",
        name=f"sft-{args.num_train_examples}",
        config={
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "num_train_examples": args.num_train_examples,
        },
    )

    for epoch in tqdm(range(args.num_epochs), desc="Training epochs"):
        train_metrics = train_one_epoch(
            model=model,
            tokenizer=tokenizer,
            train_loader=train_loader,
            optimizer=optimizer,
            device=train_device,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            normalize_constant=1.0,
            max_grad_norm=1.0,
        )

        wandb.log({
            "train_step": epoch,
            "train/avg_loss": train_metrics["avg_loss"],
            "train/avg_num_response_tokens": train_metrics["avg_num_response_tokens"],
            "train/avg_mean_log_prob": train_metrics["avg_mean_log_prob"],
            "train/num_batches": train_metrics["num_batches"],
            "train/num_optimizer_steps": train_metrics["num_optimizer_steps"],
        })

        load_policy_into_vllm_instance(model, vllm_model)

        results, summary = log_generations(
            vllm_model=vllm_model,
            reward_fn=r1_zero_reward_fn,
            prompts=val_prompts,
            ground_truths=val_ground_truths,
            eval_sample_params=eval_sampling_params,
            tokenizer=tokenizer,
            step=epoch,
            split="validation",
            results_path=os.path.join(args.output_dir, f"val_generations_epoch_{epoch}.jsonl"),
            summary_path=os.path.join(args.output_dir, f"val_summary_epoch_{epoch}.json"),
        )

        wandb.log({
            "eval_step": epoch,
            "eval/accuracy": summary["accuracy"],
            "eval/avg_reward": summary["avg_reward"],
            "eval/avg_format_reward": summary["avg_format_reward"],
            "eval/avg_answer_reward": summary["avg_answer_reward"],
            "eval/avg_response_length": summary["avg_response_length"],
        })

        save_dir = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}")
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
    wandb.finish()


if __name__ == "__main__":
    main()



