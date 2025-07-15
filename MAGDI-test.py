import os
import torch
import numpy as np
import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==== User-defined variables ====
CUDA_DEVICES = "0,1,2,3"
BATCH_SIZE = 10
BASE_MODEL = 'meta-llama/Llama-3.2-3B'
LORA_MODEL = 'MAGDi_Llama3_Distilled'
CACHE_DIR = './hf_models'
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 600
SEED = 42

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICES

print(f"=== Evaluating {LORA_MODEL} on 20% of StrategyQA (wics/strategy-qa) (seed={SEED}) ===")

# ==== Load distilled model (decoder only, no GCN) ====
model = AutoModelForCausalLM.from_pretrained(
    LORA_MODEL,
    cache_dir=CACHE_DIR,
    device_map='auto'
)
model.eval()

# ==== Load tokenizer ====
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    padding_side='left',
    add_eos_token=False
)
tokenizer.pad_token_id = tokenizer.eos_token_id

# ==== Load and split StrategyQA test set ====
test_data = load_dataset("wics/strategy-qa", split="test")
test_data = test_data.train_test_split(test_size=0.20, seed=SEED)
test_data = test_data["test"]
questions = test_data['question']
answers = test_data['answer']  # Boolean: True/False

# Build prompts for LLM
prompts = [f"Question: {q}\nAnswer:" for q in questions]
test_prompts = prompts
test_answers = answers

def parse_answer_strategyqa(generated_text):
    # Heuristic: look for "true" or "false" (case-insensitive) in the output
    text = generated_text.lower()
    if "true" in text:
        return True
    elif "false" in text:
        return False
    # fallback: guess False
    return False

def calc_acc(pred_ans, gold_ans):
    return sum([p == g for p, g in zip(pred_ans, gold_ans)]) / len(gold_ans) if gold_ans else 0

# ==== Evaluation loop ====
result = []
for idx in range(0, len(test_prompts), BATCH_SIZE):
    batch_prompts = test_prompts[idx: idx+BATCH_SIZE]
    batch_tok = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    batch_tok = {k: v.to(model.device) for k, v in batch_tok.items()}
    with torch.no_grad():
        output_tokens = model.generate(
            **batch_tok,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            temperature=TEMPERATURE,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
    generated_txts = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
    result.extend(generated_txts)
    pred_ans = [parse_answer_strategyqa(o) for o in result]
    acc = calc_acc(pred_ans, test_answers[:len(result)])
    print(f"{datetime.datetime.now().strftime('%y-%m-%d %H:%M')} | samples evaluated: {len(result)} | accuracy: {acc:.3f}")

# Optionally print a few predictions
for i in range(min(5, len(test_answers))):
    print(f"Q: {test_prompts[i]}")
    print(f"Model: {result[i]}")
    print(f"Gold: {test_answers[i]} | Pred: {pred_ans[i]}")
    print("---")
