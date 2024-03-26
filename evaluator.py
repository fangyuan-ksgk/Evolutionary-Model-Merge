import torch
from tqdm import tqdm as tqdm
import os, json

# model_name = "xxx"

def formatting_prompts_func(example):
    output_texts = []
    query_texts = []
    for i in range(len(example['prompt'])):
        query_text = f"### Question: {example['prompt'][i]}\n ### Answer: "
        answer_text = f"{example['completion'][i]}"
        text = query_text + answer_text
        output_texts.append(text)
        query_texts.append(query_text)
    # calculate number of tokens 
    return output_texts, query_texts

def format_prompt(prompt, completion):
    return f"### Question: {prompt}\n ### Answer: {completion}", f"### Question: {prompt}\n ### Answer: "


def evaluate_perplexity(model, tokenizer, dataset):
    # device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    # model.to(device)
    model.eval()

    perplexities = []

    for data in tqdm(dataset, desc="Evaluating Perplexity on the Instruction-Tuning Dataset"):
        prompt = data["prompt"]
        completion = data["completion"] + tokenizer.eos_token

        target_sequence, query_sequence = format_prompt(prompt, completion)
        query_sequence_length = tokenizer.encode(query_sequence, return_tensors="pt").shape[1]

        sequence_ids = tokenizer.encode(target_sequence, return_tensors="pt")

        with torch.no_grad():
            sequence_logits = model(sequence_ids).logits
            target_logits = sequence_logits[:, (query_sequence_length-1):-1]
            target_ids = sequence_ids[:, query_sequence_length:].view(-1)

        target_logits = sequence_logits[:, (query_sequence_length-1):-1]
        target_ids = sequence_ids[:, query_sequence_length:].view(-1)

        loss = torch.nn.functional.cross_entropy(target_logits.reshape(-1, target_logits.size(-1)), target_ids, reduction="none")

        perplexity = loss.mean()
        perplexities.append(perplexity)

    return sum(perplexities) / len(perplexities)


def save_info_to_json(info, file_path="./merge_info/info.json"):
    os.makedirs("./merge_info", exist_ok=True)
    try:
        with open(file_path, "r") as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = []
    existing_data.append(info)
    with open(file_path, "w") as file:
        json.dump(existing_data, file, indent=4)

def get_info_from_json(file_path="./merge_info/info.json"):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print("Info file not found.")
        return []
