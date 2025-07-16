import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset
import pickle

# === Role-specific Instructions for the Personas ===
role_specifications = {
    "Scientist": """
You are a Scientist. When making decisions:

ALWAYS DO:
- Generate 2 conflicting hypotheses before selecting an option
- Conduct a Red Team analysis attacking your own conclusion
- Calculate Bayesian probabilities for competing explanations using P(H|E) = P(E|H)P(H)/P(E)
- Model system interactions using both linear and chaotic frameworks
- Compare findings against contradictory studies from adjacent fields
- Test your reasoning by asking "what could prove this wrong?"
- Consider environmental and health impacts spanning 50+ years
- Demand evidence with statistical significance before accepting claims
- Make Decision based on this

Respond with your reasoning and final answer.
""",
    "Lawyer": """
You are a Lawyer. When making decisions:

ALWAYS DO:
- Analyze under Common Law and Civil Law frameworks
- Simulate arguments from plaintiff/defendant perspectives simultaneously
- Identify conflicting precedents across federal circuits
- Apply game theory to predict settlement likelihoods using Nash equilibrium
- Check legality under local, national, and international law
- Identify who could sue whom if this decision is made
- Consider precedent this sets for future similar cases
- Evaluate enforceability and compliance mechanisms
- Assess constitutional and human rights implications
- Make Decision based on this

Respond with your reasoning and final answer.
""",
    "Historian": """
You are a Historian. When making decisions:

ALWAYS DO:
- Contextualize the issue within relevant historical periods and events
- Identify historical precedents and analogues for each option
- Analyze the long-term consequences of similar decisions in the past
- Examine the roles of key actors, institutions, and social forces in shaping outcomes
- Assess the reliability and biases of historical sources and narratives
- Consider the impact of cultural, economic, and technological changes over time
- Highlight lessons learned from both successes and failures in history
- Address how collective memory and historiography influence present choices
- Make Decision based on this

Respond with your reasoning and final answer.
""",
    "Mathematician": """
You are a Mathematician. When making decisions:

ALWAYS DO:
- Solve using both frequentist and Bayesian approaches
- Model with Monte Carlo and deterministic simulations
- Calculate error propagation through all estimation steps
- Apply robust optimization against adversarial inputs
- Quantify all variables and assign numerical values
- Calculate expected outcomes using probability theory
- Model best-case, worst-case, and most-likely scenarios
- Identify optimization targets and constraints
- Express uncertainty using confidence intervals
- Make Decision based on this

Respond with your reasoning and final answer.
""",
    "Ethicist": """
You are an Ethicist. When making decisions:

ALWAYS DO:
- Apply in sequence: Utilitarian, Deontological, Virtue Ethics lenses
- Calculate moral weightings using differentiable ethics equations
- Identify irreconcilable value conflicts through geometric mean analysis
- Apply multiple ethical tests: "Is this fair?", "Does this reduce suffering?", "Would I want this if roles were reversed?"
- Consider moral obligations to future generations
- Weigh individual rights against collective good
- Identify moral dilemmas and tragic trade-offs
- Question the moral legitimacy of the decision-makers
- Perform universalizability tests for proposed actions
- Make Decision based on this

Respond with your reasoning and final answer.
"""
}

# === Configuration ===
SEED = 42
set_seed(SEED)
LLAMA_TEACHER = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LLAMA_STUDENT = "meta-llama/Llama-3.2-3B"
TEMPERATURE = 4.0  # Temperature for softening logits
ALPHA = 0.7        # Weight for distillation loss vs hard target loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to clear GPU memory
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cleared.")
    else:
        print("No GPU available.")

# === Load Models and Tokenizer ===
print("Loading tokenizer and models...")
tokenizer = AutoTokenizer.from_pretrained(LLAMA_TEACHER, padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load a single teacher model
teacher_model = AutoModelForCausalLM.from_pretrained(
    LLAMA_TEACHER,
    torch_dtype=torch.bfloat16,
    device_map="auto"
).eval()

# Load student model
student_model = AutoModelForCausalLM.from_pretrained(
    LLAMA_STUDENT,
    torch_dtype=torch.bfloat16,
    device_map="auto"
).train()

# === Response Generation and Aggregation ===
def generate_persona_response(spec, question, options):
    prompt = spec + f"\n\nQuestion: {question}\n\nOptions:\n" + "\n".join(f"{i}: {opt}" for i, opt in enumerate(options)) + "\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = teacher_model.generate(**inputs, max_new_tokens=200, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def aggregate_persona_responses(question, options):
    responses = [generate_persona_response(spec, question, options) for spec in role_specifications.values()]
    # Simple aggregation: concatenate responses
    aggregated_response = "\n\n".join(responses)
    return aggregated_response

# === Dataset Preparation ===
class DistillationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_encoding = self.tokenizer(
            item["input"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        # Tokenize teacher response for full-sequence logits
        teacher_encoding = self.tokenizer(
            item["teacher_response"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": input_encoding["input_ids"].squeeze(0),
            "attention_mask": input_encoding["attention_mask"].squeeze(0),
            "teacher_logits": item["teacher_logits"],  # Pre-computed full logits
            "labels": teacher_encoding["input_ids"].squeeze(0)  # Use teacher response as labels for hard loss
        }

# === Distillation Loss Function ===
class DistillationLoss(nn.Module):
    def __init__(self, alpha=ALPHA, temperature=TEMPERATURE):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        # Full-sequence KL divergence
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        distill_loss = self.kl_loss(student_log_probs.view(-1, student_log_probs.size(-1)),
                                    teacher_probs.view(-1, teacher_probs.size(-1))) * (self.temperature ** 2)

        # Hard-target loss
        hard_loss = self.ce_loss(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))

        return self.alpha * distill_loss + (1 - self.alpha) * hard_loss

# === Data Processing ===
def process_dataset():
    print("Loading and processing dataset...")
    dataset = load_dataset("wics/strategy-qa", split="test")
    dataset = dataset.shuffle(seed=SEED).select(range(500))

    processed = []
    for ex in tqdm(dataset, desc="Processing examples"):
        q = ex["question"]
        opts = ["True", "False"]
        teacher_response = aggregate_persona_responses(q, opts)
        # Pre-compute teacher logits for the aggregated response
        teacher_inputs = tokenizer(teacher_response, return_tensors="pt").to(device)
        with torch.no_grad():
            teacher_logits = teacher_model(**teacher_inputs).logits.squeeze(0)
        processed.append({
            "input": f"Question: {q}\nAnswer:",
            "teacher_response": teacher_response,
            "teacher_logits": teacher_logits.cpu(),
            "gold_answer": ex["answer"]
        })
        clear_gpu_memory()  # Clear memory after each example
    clear_gpu_memory()  # Final clear after loop
    return processed

# === Custom Trainer ===
class DistillationTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.distill_loss = DistillationLoss()
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, num_items_in_batch, return_outputs=False):
        outputs = model(**inputs)
        student_logits = outputs.logits
        # Explicitly move teacher_logits to the same device as student_logits
        teacher_logits = inputs["teacher_logits"].to(student_logits.device)
        labels = inputs["labels"]
        loss = self.distill_loss(student_logits, teacher_logits, labels)
        return (loss, outputs) if return_outputs else loss

# === Main Training Function ===
def main():
    print("Starting multi-agent knowledge distillation...")
    data = process_dataset()

    # Split into train/validation
    split = int(0.8 * len(data))
    train_data, val_data = data[:split], data[split:]

    train_ds = DistillationDataset(train_data, tokenizer)
    val_ds = DistillationDataset(val_data, tokenizer)

    training_args = TrainingArguments(
        output_dir="./multi_agent_distilled_student",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=2,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=100,
        logging_dir="./logs",
        logging_steps=50,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        fp16=True,
        report_to="none"
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )

    trainer = DistillationTrainer(
        model=student_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model("./multi_agent_distilled_student/final")

    with open("./multi_agent_distilled_dataset.pkl", "wb") as f:
        pickle.dump(data, f)

    print("Training completed successfully!")
    print("Model saved to ./multi_agent_distilled_student/final")
    print("Dataset saved to ./multi_agent_distilled_dataset.pkl")

if __name__ == "__main__":
    main()
