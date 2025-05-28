# Load dataset (CommonsenseQA) 
!pip install -U datasets

import shutil
import os

# Clean cache to avoid LocalFileSystem error
shutil.rmtree(os.path.expanduser("~/.cache/huggingface/datasets"), ignore_errors=True)

from datasets import load_dataset

# High-level architecture implementing:
# - MLP moderator
# - Persona LLMs (small models returning softmax over MC options)
# - Grouping by domain (sorted by profession as hidden layers)
# - Optional self-reflective collaboration
# - Fine-tuning each persona for its domain-specific role
# - Per-group validation and checkpoint saving

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from datasets import load_dataset
from torch.utils.data import DataLoader

# -----------------------
# CONFIGURATION
# -----------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MC_OPTIONS = ['A', 'B', 'C', 'D', 'E']
NUM_CHOICES = len(MC_OPTIONS)
BATCH_SIZE = 4
EPOCHS = 3
CHECKPOINT_DIR = './checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Define persona metadata
persona_groups = {
    'science': ['doctor', 'engineer'],
    'humanities': ['artist', 'historian'],
    'law': ['lawyer', 'politician']
}
persona_models = {
    'doctor': 'google/bert_uncased_L-4_H-256_A-4',
    'engineer': 'google/bert_uncased_L-4_H-256_A-4',
    'artist': 'google/bert_uncased_L-4_H-256_A-4',
    'historian': 'google/bert_uncased_L-4_H-256_A-4',
    'lawyer': 'google/bert_uncased_L-4_H-256_A-4',
    'politician': 'google/bert_uncased_L-4_H-256_A-4'
}

# -----------------------
# TOKENIZER (shared)
# -----------------------
tokenizer = AutoTokenizer.from_pretrained('google/bert_uncased_L-4_H-256_A-4')

# -----------------------
# Persona LLM Wrapper (with fine-tuning support)
# -----------------------
class PersonaLLM(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.model = AutoModelForMultipleChoice.from_pretrained(persona_models[name])

    def forward(self, question, choices, context=None):
        # Add context if provided
        if context:
            question = f"{question}\nPeer suggestions: {context}"

        # Prepare question-choice pairs
        input_pairs = [f"{question} {choice}" for choice in choices]

        tokenized_inputs = tokenizer(
            input_pairs,
            padding = 'max_length',
            truncation = True,
            max_length = 128,
            return_tensors = "pt"
        )

        #tokenized_inputs = {k: v.view(1, 5, 128).to(DEVICE) for k, v in tokenized_inputs.items()}
        tokenized_inputs = {k: v.unsqueeze(0).to(DEVICE) for k, v in tokenized_inputs.items()}


        # Debug: Check shape of tokenized inputs
        print(f"Tokenized inputs shape: {tokenized_inputs['input_ids'].shape}")

        # Ensure the tensor has the correct shape: [1, 5, 128]
        assert tokenized_inputs['input_ids'].shape[0] == 1, f"Expected batch size of 1, got {tokenized_inputs['input_ids'].shape[0]}"
        assert tokenized_inputs['input_ids'].shape[1] == len(choices), f"Expected {len(choices)} choices, got {tokenized_inputs['input_ids'].shape[1]}"
        assert tokenized_inputs['input_ids'].shape[2] == 128, f"Expected sequence length of 128, got {tokenized_inputs['input_ids'].shape[2]}"

        # Pass tokenized inputs through the model
        logits = self.model(**tokenized_inputs).logits

        # Apply softmax over the logits to get probabilities for each choice
        probs = F.softmax(logits, dim=-1)
        return probs.squeeze()  # Return the probabilities for each choice

# -----------------------
# Moderator MLP
# -----------------------
class ModeratorMLP(nn.Module):
    def __init__(self, num_groups, hidden_size=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_groups * NUM_CHOICES, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, NUM_CHOICES)
        )

    def forward(self, group_logits):
        x = torch.cat(group_logits, dim=-1)
        return self.mlp(x)

# -----------------------
# MAS Inference Pipeline
# -----------------------
class MASSystem(nn.Module):
    def __init__(self):
        super().__init__()
        self.personas = nn.ModuleDict({name: PersonaLLM(name) for name in persona_models})
        self.groupings = persona_groups
        self.moderator = ModeratorMLP(num_groups=len(persona_groups) * NUM_CHOICES).to(DEVICE)  # Update input size

    def forward(self, question, choices):
        group_logits = []
        for group, persona_names in self.groupings.items():
            persona_outputs = []
            for name in persona_names:
                peer_context = ", ".join([f"{n}: ?" for n in persona_names if n != name])
                output = self.personas[name].forward(question, choices, context=peer_context)
                persona_outputs.append(output)
            group_output = torch.stack(persona_outputs, dim=0).mean(dim=0)  # Average outputs from personas in a group
            group_logits.append(group_output)
        return self.moderator(group_logits)

# -----------------------
# Load Dataset
# -----------------------
print("Loading dataset...")
dataset = load_dataset("commonsense_qa", download_mode="force_redownload")
dataset = dataset["train"].train_test_split(test_size=0.4, seed=42)
val_test = dataset["test"].train_test_split(test_size=0.75, seed=42)
dataset["val"] = val_test["train"]
dataset["test"] = val_test["test"]

# Preprocessing
class QADataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        question = ex["question"]
        labels = ex["choices"]["label"]
        texts = ex["choices"]["text"]
        choices = [f"{label}. {text}" for label, text in zip(labels, texts)]
        while len(choices) < NUM_CHOICES:
            choices.append("None of the above")
        choices = choices[:NUM_CHOICES]
        answer = ex.get("answerKey", "A")
        label = MC_OPTIONS.index(answer) if answer in MC_OPTIONS else 0
        return question, choices, label

def collate_fn(batch):
    questions, choices_list, labels = zip(*batch)
    return list(questions), list(choices_list), torch.tensor(labels)

train_loader = DataLoader(QADataset(dataset["train"]), batch_size=1, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(QADataset(dataset["val"]), batch_size=1, collate_fn=collate_fn)

# -----------------------
# Training Loop with Checkpoints & Per-Group Validation
# -----------------------
model = MASSystem().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()
best_val_loss = float('inf')

print("Starting training...")
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for questions, choices_list, answer_idxs in train_loader:
        question = questions[0]
        choices = choices_list[0]
        answer_idx = answer_idxs[0].item()
        logits = model.forward(question, choices)
        loss = loss_fn(logits.unsqueeze(0), torch.tensor([answer_idx], device=DEVICE))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for questions, choices_list, answer_idxs in val_loader:
            question = questions[0]
            choices = choices_list[0]
            answer_idx = answer_idxs[0].item()
            logits = model.forward(question, choices)
            val_loss += loss_fn(logits.unsqueeze(0), torch.tensor([answer_idx], device=DEVICE)).item()
            pred = torch.argmax(logits).item()
            correct += int(pred == answer_idx)
            total += 1
    val_loss /= len(val_loader)
    val_acc = correct / total

    print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save best checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'best_model.pt'))
    model.train()

# -----------------------
# Inference Example
# -----------------------
if __name__ == '__main__':
    model.eval()
    question = "What is the capital of France?"
    choices = ["A. Berlin", "B. Madrid", "C. Paris", "D. Rome", "E. Lisbon"]
    with torch.no_grad():
        logits = model.forward(question, choices)
        probs = F.softmax(logits, dim=-1)
    print("\nFinal Prediction Probabilities:")
    for i, choice in enumerate(choices):
        print(f"{choice}: {probs[i].item():.3f}")
    print(f"\nPredicted Answer: {choices[probs.argmax().item()]}")
