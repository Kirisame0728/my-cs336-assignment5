import torch

def padding(batch_sequence, padding_value):
    max_len = max(line.size(0) for line in batch_sequence)
    out = torch.full((len(batch_sequence), max_len), padding_value, dtype=batch_sequence[0].dtype)
    for i, x in enumerate(batch_sequence):
        out[i, :len(x)] = x
    return out



def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    all_sequences = []
    all_masks = []
    for prompt, output in zip(prompt_strs, output_strs):
        enc_prompt = tokenizer(prompt)["input_ids"]
        enc_output = tokenizer(output)["input_ids"]
        prompt_and_output = torch.tensor(enc_prompt+enc_output, dtype=torch.long)
        mask = torch.tensor([0]*len(enc_prompt)+[1]*len(enc_output), dtype=torch.bool)

        all_sequences.append(prompt_and_output)
        all_masks.append(mask)

    pad_token_id = tokenizer.pad_token_id
    padded_sequence = padding(all_sequences, pad_token_id)
    padded_mask = padding(all_masks, 0)

    input_ids = padded_sequence[:, :-1]
    labels = padded_sequence[:, 1:]
    response_mask = padded_mask[:, 1:]

    dict = {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask
    }

    return dict