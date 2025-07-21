# Llama-PEFT with StrategyQA (Generation - 63.5% 3B, 72.1% 8B)

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
    Seq2SeqTrainingArguments, Trainer,
    DataCollatorForSeq2Seq, EvalPrediction
)
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score
import numpy as np

# 3. Load & Format StrategyQA Dataset
dataset = load_dataset("wics/strategy-qa")["test"]

def format_strategyqa(example):
    prompt = f"Q: {example['question']}\nA:"
    label = "yes" if example["answer"] else "no"
    return {"prompt": prompt, "label": label}

# Format and keep only the new prompt/label fields
dataset = dataset.map(format_strategyqa)
dataset = dataset.remove_columns([col for col in dataset.column_names if col not in {"prompt", "label"}])

# Split AFTER formatting
train_test = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test["train"]
eval_dataset = train_test["test"]

# 4. Load LLaMA 3.2 3B + Tokenizer
hf_token = ENV[‘AUTH_TOKEN’]
model_name = "meta-llama/Llama-3.1-8B-Instruct" # for 3B: meta-llama/Llama-3.2-3B

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

# Version Checks 
import transformers
import datasets 
import accelerate
import bitsandbytes
import torch
import sklearn

print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("accelerate:", accelerate.__version__)
print("bitsandbytes:", bitsandbytes.__version__)
print("torch:", torch.__version__)
print("sklearn:", sklearn.__version__)

# 6. Tokenize prompt + label together for causal LM
def tokenize(example):
    # tokenize full text: prompt + answer
    full_text = example["prompt"] + " " + example["label"]
    tokenized = tokenizer(
        full_text,
        max_length=256,
        truncation=True,
        padding="max_length",
    )
    # get length of prompt tokens
    prompt_len = len(tokenizer(example["prompt"], truncation=True)["input_ids"])
    
    # create labels: -100 for prompt tokens, then answer tokens
    labels = [-100] * prompt_len + tokenized["input_ids"][prompt_len:]
    labels = labels[:256]  # ensure max length
    
    tokenized["labels"] = labels
    return tokenized

train_dataset = train_dataset.map(tokenize, remove_columns=["prompt", "label"])
eval_dataset = eval_dataset.map(tokenize, remove_columns=["prompt", "label"])

# 7. TrainingArguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./llama-strategyqa-lora",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    learning_rate=5e-5,
    fp16=True,
    save_total_limit=1,
    report_to="none",
    logging_strategy="epoch",
    #evaluation_strategy="epoch",
    save_strategy="epoch",
    predict_with_generate=True,           # <-- Enable generation
    generation_max_length=10              # <-- Short since answers are yes/no
)

# 8. Data Collator for Seq2Seq
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,
    return_tensors="pt"
)

# 9. Exact Match Accuracy Metric
def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    torch.cuda.empty_cache()

    # If logits are returned, we need to argmax over vocab dim
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # Convert logits -> token IDs
    predictions = np.argmax(predictions, axis=-1)

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Normalize for exact match
    def normalize(text):
        text = text.strip().lower()
        return "yes" if text.startswith("yes") else "no" if text.startswith("no") else text

    import re 

    def parse_final_answer(generated_text: str) -> str | None:
        """Enhanced answer parsing that handles multiple formats"""
        # Look for structured formats first
        patterns = [
            r'ANSWER:\s*(yes|no)',
            r'CONCLUSION:\s*(yes|no)',
            r'Answer:\s*(yes|no)',
            r'Final answer:\s*(yes|no)',
            r'\b(yes|no)\s*$',  # Last True/False in text
            r'\b(yes|no)\b'     # Any True/False
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, generated_text, re.IGNORECASE)
            if matches:
                return matches[-1].capitalize()
        return None

    norm_preds = [parse_final_answer(p) for p in decoded_preds]
    norm_labels = [parse_final_answer(l) for l in decoded_labels]

    print(norm_preds)
    print(norm_labels)

    exact_matches = [p == l for p, l in zip(norm_preds, norm_labels)]
    accuracy = sum(exact_matches) / len(exact_matches)

    return {"exact_match_accuracy": accuracy}

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
