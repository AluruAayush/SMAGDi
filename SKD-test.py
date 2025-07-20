import os
import torch
import re
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# === Configuration ===
MODEL_PATH = "./multi_agent_distilled_student/final"
DATASET_NAME = "wics/strategy-qa"
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_cot_prompt():
    """
    Returns the Chain-of-Thought prompting strategy
    """
    return """You are an expert reasoning assistant. Your task is to answer True/False questions with careful analysis.

Question: {question}

Instructions:
1. Let's think step by step about this question
2. Break down the key components and requirements
3. Consider what knowledge is needed to answer this
4. Apply logical reasoning to reach a conclusion
5. State your final answer as either "True" or "False"

Analysis and Answer:"""

def prepare_test_data():
    """Loads and prepares test data"""
    print("Preparing the test dataset...")
    full_dataset = load_dataset(DATASET_NAME, split="test")
    subset = full_dataset.train_test_split(test_size=0.2, seed=42)
    test_data = subset["test"]
    print(f"Test set created with {len(test_data)} samples.")
    return test_data

def parse_final_answer(generated_text: str) -> str | None:
    """Enhanced answer parsing that handles multiple formats"""
    # Look for structured formats first
    patterns = [
        r'ANSWER:\s*(True|False)',
        r'CONCLUSION:\s*(True|False)', 
        r'Answer:\s*(True|False)',
        r'Final answer:\s*(True|False)',
        r'\b(True|False)\s*$',  # Last True/False in text
        r'\b(True|False)\b'     # Any True/False
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, generated_text, re.IGNORECASE)
        if matches:
            return matches[-1].capitalize()
    
    return None

def evaluate_with_cot(model, tokenizer, test_dataset, num_samples=50):
    """Evaluate model using Chain-of-Thought prompting"""
    print("\n--- Evaluating with Chain-of-Thought Prompting ---")
    
    cot_prompt = get_cot_prompt()
    correct_matches = 0
    total_samples = 0
    
    # Limit to specified number of samples or dataset size, whichever is smaller
    eval_size = min(num_samples, len(test_dataset))
    
    for example in tqdm(test_dataset.select(range(eval_size)), desc="Evaluating CoT"):
        question = example["question"]
        gold_answer = str(example["answer"])
        
        # Format prompt with Chain-of-Thought strategy
        prompt = cot_prompt.format(question=question)
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        # Generate response with adjusted parameters for better reliability
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.3,  # Lower temperature for more focused responses
                top_p=0.8,        # Nucleus sampling for better quality
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        model_answer = parse_final_answer(generated_text)
        
        if model_answer == gold_answer:
            correct_matches += 1
        total_samples += 1
    accuracy = (correct_matches / total_samples) * 100 if total_samples > 0 else 0
    
    print(f"\nChain-of-Thought Results:")
    print(f"Correct: {correct_matches}/{total_samples}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    return accuracy, correct_matches, total_samples

def main():
    """Main evaluation function using only Chain-of-Thought"""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model path not found at '{MODEL_PATH}'")
        return

    # Load model and tokenizer
    print(f"Loading distilled model from '{MODEL_PATH}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()

    # Load test data
    test_dataset = prepare_test_data()
    
    # Evaluate using Chain-of-Thought only
    accuracy, correct, total = evaluate_with_cot(
        model, tokenizer, test_dataset, num_samples=50
    )
    
    # Final summary
    print("\n" + "="*50)
    print("CHAIN-OF-THOUGHT EVALUATION SUMMARY")
    print("="*50)
    print(f"Strategy: Chain-of-Thought Prompting")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Correct Answers: {correct}/{total}")
    print(f"Model Path: {MODEL_PATH}")

if __name__ == "__main__":
    main()
