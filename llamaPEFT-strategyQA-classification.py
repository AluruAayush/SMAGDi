# LLaMA + PEFT - StrategyQA Classification

# 1. Install necessary packages
!pip install -U peft bitsandbytes scikit-learn accelerate
!pip install datasets==3.6
!pip install transformers==4.46.1

# 2. Imports
import torch
import datasets
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, EvalPrediction
)
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score
import numpy as np
import re
import os

# 3. Load & Format StrategyQA Dataset
dataset = load_dataset("wics/strategy-qa")["test"]

def format_strategyqa(example):
    prompt = f"Q: {example['question']}\nA:"
    label = "yes" if example["answer"] else "no"
    return {"prompt": prompt, "label": label}

dataset = dataset.map(format_strategyqa)
dataset = dataset.select_columns(["prompt", "label"])

# Split
train_test = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test["train"]
eval_dataset_original = train_test["test"]

# 4. Load LLaMA 3.1 8B + Tokenizer
hf_token = ENV[‘AUTH_TOKEN’]
model_name = "meta-llama/Llama-3.2-3B"

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

# 6. Tokenization functions
def tokenize_for_training(examples):
    full_texts = [prompt + " " + label for prompt, label in zip(examples["prompt"], examples["label"])]
    
    # Tokenize with explicit padding to max_length
    tokenized = tokenizer(
        full_texts,
        max_length=256,
        truncation=True,
        padding="max_length",  # Ensure all sequences are padded to max_length
        return_tensors=None,  # Return Python lists
        return_special_tokens_mask=True  # Helpful for debugging
    )
    
    # Ensure labels are properly formatted and padded
    tokenized["labels"] = []
    for input_ids in tokenized["input_ids"]:
        # Copy input_ids for labels, ensuring same length
        labels = input_ids.copy()
        tokenized["labels"].append(labels)
    
    # Debug: Check lengths
    for i, (input_ids, labels) in enumerate(zip(tokenized["input_ids"], tokenized["labels"])):
        if len(input_ids) != len(labels) or len(input_ids) != 256:
            print(f"Warning: Example {i} has input_ids length {len(input_ids)}, labels length {len(labels)}")
    
    return tokenized

def tokenize_for_eval(examples):
    tokenized = tokenizer(
        examples["prompt"],
        max_length=256,
        truncation=True,
        padding="max_length",  # Ensure consistent padding
        return_tensors=None
    )
    
    # Create labels as -100 for evaluation
    tokenized["labels"] = [[-100] * 256 for _ in tokenized["input_ids"]]  # Fixed length
    return tokenized

# 7. Apply tokenization with batching
print("Tokenizing training dataset...")
train_dataset = train_dataset.map(
    tokenize_for_training,
    batched=True,
    batch_size=1000,  # Process in batches for efficiency
    remove_columns=train_dataset.column_names
)

print("Tokenizing evaluation dataset...")
eval_dataset = eval_dataset_original.map(
    tokenize_for_eval,
    batched=True,
    batch_size=1000,
    remove_columns=eval_dataset_original.column_names
)

# Debug dataset structure
print("Train dataset sample:", train_dataset[0])
print("Eval dataset sample:", eval_dataset[0])
print("Train dataset columns:", train_dataset.column_names)
print("Eval dataset columns:", eval_dataset.column_names)

# Rest of the code remains the same, but ensure the DataCollator is configured correctly
# 9. Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8
)

# 10. Evaluation function
def evaluate_model_properly(model, original_dataset, num_samples=50):
    model.eval()
    correct = 0
    total = 0
    
    eval_subset = original_dataset.select(range(min(num_samples, len(original_dataset))))
    
    print(f"Evaluating on {len(eval_subset)} samples...")
    
    with torch.no_grad():
        for i, example in enumerate(eval_subset):
            prompt = example["prompt"]
            gold = example["label"]
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256, padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            try:
                outputs = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=20,  # Increased for robustness
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = generated_text[len(prompt):].strip().lower()
                
                # Improved parsing
                pred = "yes" if re.search(r"\b(yes|true)\b", response) else "no"
                
                if pred == gold:
                    correct += 1
                total += 1
                
                if i < 3:
                    print(f"Example {i+1}:")
                    print(f"  Prompt: {prompt}")
                    print(f"  Gold: {gold}")
                    print(f"  Generated: '{response}'")
                    print(f"  Prediction: {pred}")
                    print()
                    
            except Exception as e:
                print(f"Error processing example {i}: {e}")
                total += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"Final Accuracy: {accuracy:.4f} ({correct}/{total})")
    model.train()
    return {"eval_accuracy": accuracy}

# 11. Compute metrics for training
def compute_metrics(eval_pred: EvalPrediction):
    # Use evaluate_model_properly for real metrics
    metrics = evaluate_model_properly(trainer.model, eval_dataset_original, num_samples=50)
    return metrics

training_args = TrainingArguments(
    output_dir="./llama-strategyqa-lora",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    eval_accumulation_steps=1, 
    num_train_epochs=1,
    learning_rate=5e-5,
    fp16=True,
    save_total_limit=1,
    report_to="none",
    logging_strategy="epoch",
    save_strategy="epoch"
)

# 12. Trainer Setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 13. Train & Evaluate
print("Starting training...")
trainer.train()

print("Training completed. Running final evaluation...")
accuracy = evaluate_model_properly(model, eval_dataset_original, num_samples=50)
print(f"Final Evaluation Accuracy: {accuracy['eval_accuracy']:.4f}")
