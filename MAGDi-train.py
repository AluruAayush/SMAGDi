import torch
import json
import pickle
import numpy as np
from torch_geometric.data import Batch
from transformers import AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from accelerate import dispatch_model, infer_auto_device_map
from MAGDi-model import MAGDi, MAGDiTrainer  # Ensure your MAGDi model and trainer are accessible

# 1. Load Dataset
with open("mag_dataset.json", "r") as f:
    mag_dataset = json.load(f)

# 2. Prepare Contrastive Samples (as in original MAGDi)
def prepare_contrastive_samples(samples, labels):
    if len(samples) != len(labels):
        raise ValueError("Samples and labels must be of the same length.")
    positive_samples = [sample for sample, label in zip(samples, labels) if label == 1]
    negative_samples = [sample for sample, label in zip(samples, labels) if label == 0]
    if len(negative_samples) == 0:
        negative_samples = ["NA"]
    if len(positive_samples) == 0:
        return None
    if len(positive_samples) > len(negative_samples):
        negative_samples = (negative_samples * ((len(positive_samples) // len(negative_samples)) + 1))[:len(positive_samples)]
    elif len(negative_samples) > len(positive_samples):
        positive_samples = (positive_samples * ((len(negative_samples) // len(positive_samples)) + 1))[:len(negative_samples)]
    return positive_samples, negative_samples

# 3. Preprocessing Function
def preprocess_mag_dataset(mag_dataset, tokenizer):
    processed = []
    for item in mag_dataset:
        # Assume options is a list of dicts: {"text": ..., "label": ...}
        samples = [opt["text"] for opt in item["options"]]
        labels = [opt["label"] for opt in item["options"]]
        pairs = prepare_contrastive_samples(samples, labels)
        if not pairs:
            continue
        positive_samples, negative_samples = pairs
        pos_enc = tokenizer(positive_samples, padding="longest", truncation=True)
        neg_enc = tokenizer(negative_samples, padding="longest", truncation=True)
        processed.append({
            "pos_input_ids": pos_enc["input_ids"][0],
            "pos_attention_mask": pos_enc["attention_mask"][0],
            "pos_labels": pos_enc["input_ids"][0],
            "neg_input_ids": neg_enc["input_ids"][0],
            "neg_attention_mask": neg_enc["attention_mask"][0],
            "neg_labels": neg_enc["input_ids"][0],
            "graph": item["pyg_data"]
        })
    return processed

# 4. Custom Data Collator
class MAGDiDataCollator:
    def __init__(self, tokenizer, label_pad_token_id=-100):
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features):
        def pad_sequences(seqs, pad_value):
            max_len = max(len(s) for s in seqs)
            return [list(s) + [pad_value] * (max_len - len(s)) for s in seqs]
        pos_input_ids = pad_sequences([f["pos_input_ids"] for f in features], self.tokenizer.pad_token_id)
        pos_attention_mask = pad_sequences([f["pos_attention_mask"] for f in features], 0)
        pos_labels = pad_sequences([f["pos_labels"] for f in features], self.label_pad_token_id)
        neg_input_ids = pad_sequences([f["neg_input_ids"] for f in features], self.tokenizer.pad_token_id)
        neg_attention_mask = pad_sequences([f["neg_attention_mask"] for f in features], 0)
        neg_labels = pad_sequences([f["neg_labels"] for f in features], self.label_pad_token_id)
        graphs = [f["graph"] for f in features]
        batch = {
            "pos_input_ids": torch.tensor(pos_input_ids, dtype=torch.long),
            "pos_attention_mask": torch.tensor(pos_attention_mask, dtype=torch.long),
            "pos_labels": torch.tensor(pos_labels, dtype=torch.long),
            "neg_input_ids": torch.tensor(neg_input_ids, dtype=torch.long),
            "neg_attention_mask": torch.tensor(neg_attention_mask, dtype=torch.long),
            "neg_labels": torch.tensor(neg_labels, dtype=torch.long),
            "graph": Batch.from_data_list(graphs)
        }
        return batch

# 5. Tokenizer and Model Setup (Llama 3.1/3B-Instruct)
model_name = "meta-llama/Llama-3.2-3B
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", add_eos_token=True)
tokenizer.pad_token_id = tokenizer.eos_token_id

# 6. Preprocess Dataset
processed_dataset = preprocess_mag_dataset(mag_dataset, tokenizer)

# 7. Model Initialization
model = MAGDi(
    model_name=model_name,
    gcn_in_channels=4096,
    gcn_hidden_channels=512,
    gcn_out_channels=3,
    alpha=1.0,
    beta=1.0,
    gamma=0.1
)

# 8. Device Map and Model Dispatch (for multi-GPU)
from accelerate.utils import get_balanced_memory
max_memory = get_balanced_memory(
    model,
    max_memory=None,
    no_split_module_classes=["GCN", "LlamaDecoderLayer"],
    dtype="float16",
    low_zero=False,
)
device_map = infer_auto_device_map(
    model,
    max_memory=max_memory,
    no_split_module_classes=["GCN", "LlamaDecoderLayer"],
    dtype="float16"
)
model = dispatch_model(model, device_map=device_map)

# 9. PEFT Configuration
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model.decoder.gradient_checkpointing_enable()
model.decoder.enable_input_require_grads()
model.decoder = get_peft_model(model.decoder, config)

# 10. Training Arguments
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    num_train_epochs=10,
    learning_rate=5e-4,
    fp16=True,
    logging_steps=10,
    output_dir="outputs",
    remove_unused_columns=False,
    save_strategy="no"
)

# 11. Trainer Setup and Training
trainer = MAGDiTrainer(
    model=model,
    train_dataset=processed_dataset,
    args=training_args,
    data_collator=MAGDiDataCollator(tokenizer)
)
trainer.train()

# 12. Save Final Model
model.decoder.save_pretrained("MAGDi_Llama3_Distilled")
