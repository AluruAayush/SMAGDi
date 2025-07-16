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
LLAMA_STUDENT = "meta-llama/Meta-Llama-3.2-8B-Instruct"
TEMPERATURE = 4.0  # Temperature for softening logits[57]
ALPHA = 0.7  # Weight for distillation loss vs hard target loss[62]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Models and Tokenizer ===
print("Loading tokenizer and models...")
tokenizer = AutoTokenizer.from_pretrained(LLAMA_TEACHER, padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load teacher models for each persona
teacher_models = {}
for role in role_specifications.keys():
    print(f"Loading teacher model for {role}...")
    teacher_models[role] = AutoModelForCausalLM.from_pretrained(
        LLAMA_TEACHER, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    ).eval()

# Load student model
print("Loading student model...")
student_model = AutoModelForCausalLM.from_pretrained(
    LLAMA_STUDENT,
    torch_dtype=torch.bfloat16,
    device_map="auto"
).train()

# === Logits Extraction Function ===
def extract_logits_from_persona(role, question, options):
    """Extract logits from a specific persona for given input"""
    # Format prompt for the persona
    prompt = f"{role_specifications[role]}\n\nQuestion: {question}\n\nOptions:\n"
    for i, option in enumerate(options):
        prompt += f"{i}: {option}\n"
    prompt += "\nAnswer:"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(teacher_models[role].device) for k, v in inputs.items()}
    
    # Forward pass to get logits
    with torch.no_grad():
        outputs = teacher_models[role](**inputs)
        logits = outputs.logits[0, -1, :]  # Get logits for last token
    
    # Get logits for answer tokens (0, 1, 2, 3, etc.)
    answer_tokens = []
    for i in range(len(options)):
        token_id = tokenizer.encode(str(i), add_special_tokens=False)[0]
        answer_tokens.append(token_id)
    
    # Extract logits for answer options
    answer_logits = logits[answer_tokens].cpu().numpy()
    return answer_logits

def aggregate_persona_logits(question, options):
    """Aggregate logits from all personas"""
    all_logits = []
    
    for role in role_specifications.keys():
        try:
            logits = extract_logits_from_persona(role, question, options)
            all_logits.append(logits)
        except Exception as e:
            print(f"Error extracting logits for {role}: {e}")
            # Use zeros if extraction fails
            all_logits.append(np.zeros(len(options)))
    
    if not all_logits:
        return np.zeros(len(options))
    
    # Average the logits across all personas[1]
    aggregated_logits = np.mean(np.stack(all_logits), axis=0)
    return aggregated_logits

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
        
        # Tokenize input
        encoding = self.tokenizer(
            item["input"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "teacher_logits": torch.tensor(item["teacher_logits"], dtype=torch.float32),
            "labels": encoding["input_ids"].squeeze(0)  # For language modeling
        }

# === Knowledge Distillation Loss Function ===
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.7, temperature=4.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_logits, teacher_logits, labels):
        # Apply temperature scaling to both teacher and student logits[57]
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # Distillation loss (KL divergence)[58]
        distillation_loss = self.kl_loss(student_log_probs, teacher_probs) * (self.temperature ** 2)
        
        # Hard target loss (standard cross-entropy)
        hard_loss = self.ce_loss(student_logits, labels)
        
        # Combined loss[62]
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * hard_loss
        
        return total_loss

# === Data Processing ===
def process_dataset():
    print("Loading and processing dataset...")
    dataset = load_dataset("wics/strategy-qa", split="test")
    dataset = dataset.shuffle(seed=SEED).select(range(500))  # Limit for demonstration
    
    processed_data = []
    
    for example in tqdm(dataset, desc="Processing examples"):
        question = example["question"]
        options = ["True", "False"]
        
        # Get aggregated logits from all personas
        teacher_logits = aggregate_persona_logits(question, options)
        
        # Format input for student
        input_text = f"Question: {question}\nAnswer:"
        
        processed_data.append({
            "input": input_text,
            "teacher_logits": teacher_logits,
            "gold_answer": example["answer"]
        })
    
    return processed_data

# === Custom Trainer for Distillation ===
class DistillationTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.distillation_loss = DistillationLoss(alpha=ALPHA, temperature=TEMPERATURE)
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # Get student model outputs
        outputs = model(**inputs)
        student_logits = outputs.logits
        
        # Get teacher logits from inputs
        teacher_logits = inputs["teacher_logits"]
        
        # Compute distillation loss
        loss = self.distillation_loss(
            student_logits.view(-1, student_logits.size(-1)),
            teacher_logits.view(-1, teacher_logits.size(-1)),
            inputs["labels"].view(-1)
        )
        
        return (loss, outputs) if return_outputs else loss

# === Main Training Function ===
def main():
    print("Starting multi-agent knowledge distillation...")
    
    # Process dataset
    processed_data = process_dataset()
    
    # Split data (80% train, 20% validation)
    train_size = int(0.8 * len(processed_data))
    train_data = processed_data[:train_size]
    val_data = processed_data[train_size:]
    
    # Create datasets
    train_dataset = DistillationDataset(train_data, tokenizer)
    val_dataset = DistillationDataset(val_data, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./multi_agent_distilled_student",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
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
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # Initialize trainer
    trainer = DistillationTrainer(
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    trainer.save_model("./multi_agent_distilled_student/final")
    
    # Save the processed dataset
    with open("./multi_agent_distilled_dataset.pkl", "wb") as f:
        pickle.dump(processed_data, f)
    
    print("Training completed successfully!")
    print(f"Model saved to: ./multi_agent_distilled_student/final")
    print(f"Dataset saved to: ./multi_agent_distilled_dataset.pkl")

if __name__ == "__main__":
    main()
