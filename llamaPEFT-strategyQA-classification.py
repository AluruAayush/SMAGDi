# LLaMA + PEFT - StrategyQA Classification (Final Version) 

# 1. Install necessary packages
!pip install -U peft bitsandbytes scikit-learn accelerate
!pip install datasets==3.6
!pip install transformers==4.6.1 

# 2. Imports
import torch
import datasets 
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForSeq2Seq, EvalPrediction
)
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score
import numpy as np
import re

# 3. Load & Format StrategyQA Dataset
dataset = load_dataset("wics/strategy-qa")["test"]

def format_strategyqa(example):
    prompt = f"Q: {example['question']}\nA:"
    label = "yes" if example["answer"] else "no"
    return {"prompt": prompt, "label": label}

dataset = dataset.map(format_strategyqa)
dataset = dataset.remove_columns([col for col in dataset.column_names if col not in {"prompt", "label"}])

# Split
train_test = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test["train"]
eval_dataset = train_test["test"]

# 4. Load LLaMA 3.1 8B + Tokenizer
hf_token = ENV[‘AUTH_TOKEN’]
model_name = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
    token=hf_token
)

# 5. Apply LoRA
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 6. Tokenize Prompt Only and retain original prompt + label
def tokenize(example):
    tokenized = tokenizer(
        example["prompt"],
        max_length=256,
        truncation=True,
        padding="max_length",
    )
    tokenized["prompt"] = example["prompt"]
    tokenized["gold"] = example["label"]
    return tokenized

train_dataset = train_dataset.map(tokenize)
eval_dataset = eval_dataset.map(tokenize)

# 7. TrainingArguments (no generation)
training_args = TrainingArguments(
    output_dir="./llama-strategyqa-lora",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    learning_rate=5e-5,
    fp16=True,
    save_total_limit=1,
    report_to="none",
    logging_strategy="epoch",
    save_strategy="epoch"
)

# 8. Data Collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,
    return_tensors="pt"
)

# 9. Classification-style Accuracy
def compute_metrics(eval_pred: EvalPrediction):
    inputs = eval_dataset["prompt"]
    golds = eval_dataset["gold"]
    preds = []

    for prompt in inputs:
        # Score YES
        yes_input = tokenizer(prompt + " yes", return_tensors="pt").to(model.device)
        yes_logits = model(**yes_input).logits[0]
        yes_tokens = yes_input["input_ids"][0]
        yes_logprobs = yes_logits.log_softmax(dim=-1)
        yes_score = sum([
            yes_logprobs[i, token.item()].item()
            for i, token in enumerate(yes_tokens[-2:])  # just score "yes"
        ])

        # Score NO
        no_input = tokenizer(prompt + " no", return_tensors="pt").to(model.device)
        no_logits = model(**no_input).logits[0]
        no_tokens = no_input["input_ids"][0]
        no_logprobs = no_logits.log_softmax(dim=-1)
        no_score = sum([
            no_logprobs[i, token.item()].item()
            for i, token in enumerate(no_tokens[-2:])  # just score "no"
        ])

        pred = "yes" if yes_score > no_score else "no"
        preds.append(pred)

    exact_matches = [p == g for p, g in zip(preds, golds)]
    accuracy = sum(exact_matches) / len(exact_matches)
    return {"classification_accuracy": accuracy}

# 10. Trainer Setup
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 11. Train & Evaluate
trainer.train()
results = trainer.evaluate()
print("Evaluation Results:", results)
