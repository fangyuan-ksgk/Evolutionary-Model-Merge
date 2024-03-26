# Modal-based Evaluation Script
import modal
from modal import Image, Stub, gpu
import torch
from tqdm import tqdm
from huggingface_hub import login
import argparse
from evaluator import format_prompt
import os

login(os.environ["HF_TOKEN"])

stub = modal.Stub(
    image = Image.debian_slim(python_version="3.11")
    .pip_install(
        ["transformers", "datasets", "huggingface_hub", "torch", "tqdm", "psutil", "sentencepiece"]
    )
    .apt_install("git")
    .apt_install( "gcc")
    .run_commands("git config --global user.name ksgk-fangyuan",
                  "git config --global user.email fangyuan.yu18@gmail.com",
                  )
)

@stub.function(gpu = modal.gpu.A100(size="40GB"),
               secrets=[modal.Secret.from_name("ksgk-secret")],
               timeout=2400)
def evaluate_perplexity(model_id, dataset_name):


    from huggingface_hub import login
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    
    login(os.environ["HF_TOKEN"])
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dataset = load_dataset(dataset_name, split="test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Updated to use CPU as fallback
    model.to(device)
    model.eval()

    perplexities = []

    for data in tqdm(dataset, desc="Evaluating Perplexity on the Instruction-Tuning Dataset"):
        prompt = data["prompt"]
        completion = data["completion"] + tokenizer.eos_token

        target_sequence, query_sequence = format_prompt(prompt, completion)
        query_sequence_length = tokenizer.encode(query_sequence, return_tensors="pt").shape[1]

        sequence_ids = tokenizer.encode(target_sequence, return_tensors="pt").to(device)  # Ensure sequence_ids are on the correct device

        with torch.no_grad():
            sequence_logits = model(sequence_ids).logits
            target_logits = sequence_logits[:, (query_sequence_length-1):-1]
            target_ids = sequence_ids[:, query_sequence_length:].view(-1)

        target_logits = sequence_logits[:, (query_sequence_length-1):-1]
        target_ids = sequence_ids[:, query_sequence_length:].view(-1)

        loss = torch.nn.functional.cross_entropy(target_logits.reshape(-1, target_logits.size(-1)), target_ids, reduction="none")

        perplexity = loss.mean().cpu()
        perplexities.append(perplexity)

    # for some reason it needs to go back to CPU before returning
    del model
    del tokenizer

    return sum(perplexities) / len(perplexities)



@stub.local_entrypoint()
def main(model_id = "HuggingFaceH4/zephyr-7b-beta", 
         dataset_name = "Ksgk-fy/alignment-sft-test01"):

    avg_perplexity = evaluate_perplexity.remote(model_id, dataset_name)
    print("Average Perplexity:", avg_perplexity)
    info = {"Model ID": model_id, "Dataset Name": dataset_name, "Average Perplexity": avg_perplexity}


    os.makedirs("./merge_info", exist_ok=True)

    # Save info to a text file
    os.makedirs("./merge_info", exist_ok=True)
    with open("./merge_info/info.txt", "a") as file:
        file.write(str(info) + "\n")

    # # Save info to a text file
    # os.makedirs("./merge_info", exist_ok=True)
    # with open("./merge_info/info.txt", "w") as file:
    #     file.write(str(info))
