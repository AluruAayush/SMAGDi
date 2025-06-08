import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
import time

# Import the unified model
#from smodel import SocraticMAGDi

def extract_answer(dataset_type, text):
    """Extract answer from generated text based on dataset format."""
    import re

    # For multiple choice questions
    if dataset_type in ["multiple_choice", "classification"]:
        # Look for answer pattern like "the answer is A" or "Option: B"
        answer_pattern = r"(?:answer|option)(?:\s+is\s+|\s*:\s*)([A-E])"
        match = re.search(answer_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Look for answer in the format {{A}}
        answer_pattern = r"\{\{([A-E])\}\}"
        match = re.search(answer_pattern, text)
        if match:
            return match.group(1).upper()

        # Check if any option letter appears at the end of the text
        last_lines = text.strip().split("\n")[-3:]
        last_text = " ".join(last_lines)
        options = re.findall(r"\b([A-E])\b", last_text)
        if options:
            return options[-1].upper()

    # For boolean answers (StrategyQA)
    elif dataset_type in ["boolean", "strategy-qa"]:
        # Look for true/false patterns
        answer_pattern = r"(?:answer|result|solution)(?:\s+is\s+|\s*:\s*)(true|false)"
        match = re.search(answer_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).lower()

        # Look for answer in the format {{true}}
        answer_pattern = r"\{\{(true|false)\}\}"
        match = re.search(answer_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).lower()

    # For numerical answers
    elif dataset_type in ["numerical", "math"]:
        # Look for answer pattern like "the answer is 42" or "= 42"
        answer_pattern = r"(?:answer|result|solution)(?:\s+is\s+|\s*=\s*)([+-]?\d+(?:\.\d+)?)"
        match = re.search(answer_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)

        # Look for answer in the format {{42}}
        answer_pattern = r"\{\{([+-]?\d+(?:\.\d+)?)\}\}"
        match = re.search(answer_pattern, text)
        if match:
            return match.group(1)

        # Look for the last number in the text
        numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", text)
        if numbers:
            return numbers[-1]

    # For free-form text answers, just return the generated text
    return text


def evaluate_accuracy(dataset_type, predictions, references):
    """Evaluate accuracy based on dataset type."""
    correct = 0
    total = len(predictions)

    for pred, ref in zip(predictions, references):
        if dataset_type in ["boolean", "strategy-qa"]:
            # Convert to consistent boolean format
            pred_bool = str(pred).lower().strip() == "true"
            ref_bool = bool(ref) if isinstance(ref, bool) else str(ref).lower().strip() == "true"
            if pred_bool == ref_bool:
                correct += 1
        elif dataset_type in ["numerical", "math"]:
            # Convert to float for numerical comparison
            try:
                pred_val = float(pred)
                ref_val = float(ref)
                if abs(pred_val - ref_val) < 1e-6:
                    correct += 1
            except ValueError:
                pass
        else:
            # Direct string comparison for multiple choice or text
            if str(pred).strip().upper() == str(ref).strip().upper():
                correct += 1

    return correct / total if total > 0 else 0


def load_model_from_config(model_path):
    """Load SocraticMAGDi model using saved configuration"""
    # Read saved configuration
    config_path = os.path.join(model_path, "config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, "r") as f:
        model_config = json.load(f)

    print(f"Loading model with config: {model_config}")

    # Create model using saved configuration
    model = SocraticMAGDi(
        decomposer_name=model_config["decomposer_name"],
        solver_name=model_config["solver_name"],
        gcn_in_channels=model_config["gcn_in_channels"],
        gcn_hidden_channels=model_config["gcn_hidden_channels"],
        gcn_out_channels=model_config["gcn_out_channels"],
        alpha=model_config["alpha"],
        beta=model_config["beta"],
        gamma=model_config["gamma"],
        delta=model_config["delta"]
    )

    # Load the saved state dict
    model_state_path = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(model_state_path):
        # Regular model loading (no LoRA)
        model.load_state_dict(torch.load(model_state_path, map_location="cpu"))
        print("Loaded model weights from saved checkpoint")
    else:
        print("Warning: No saved weights found, using initialized model")

    return model, model_config


def socratic_inference_unified(question, model, tokenizer, max_new_tokens, temperature, num_beams, do_sample, dataset_type, example_num=None):
    """Perform inference using the unified SocraticMAGDi model with detailed printing."""
    device = next(model.parameters()).device
    
    # Print the original question
    print(f"\n{'='*80}")
    if example_num:
        print(f"🔍 EXAMPLE {example_num}")
    print(f"📝 ORIGINAL QUESTION:")
    print(f"   {question}")
    print(f"{'='*80}")

    # Step 1: Use decomposer to break down the question
    print(f"🧩 STEP 1: DECOMPOSER - Breaking down the question...")
    decomposer_prompt = f"Question: {question}\nBreak this down into sub-questions:"
    print(f"📤 Decomposer Prompt: {decomposer_prompt}")

    decomposer_inputs = tokenizer(
        decomposer_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    # Generate using the decomposer component
    with torch.no_grad():
        decomposer_outputs = model.decomposer.model.generate(
            **decomposer_inputs,
            max_new_tokens=max_new_tokens // 2,
            temperature=temperature,
            num_beams=num_beams,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id
        )

    sub_questions_text = tokenizer.decode(
        decomposer_outputs[0][decomposer_inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    
    print(f"📥 Decomposer Raw Output:")
    print(f"   {sub_questions_text}")

    # Extract sub-questions
    sub_questions = []
    for line in sub_questions_text.split('\n'):
        line = line.strip()
        if line.startswith('-') or line.startswith('*'):
            sub_questions.append(line[1:].strip())
        elif "?" in line:
            sub_questions.append(line)

    # If no sub-questions were extracted, use a default approach
    if not sub_questions:
        sub_questions = [
            "What are the key components of this problem?",
            "What approach should I use to solve this problem?"
        ]

    print(f"🔗 EXTRACTED SUB-QUESTIONS ({len(sub_questions)}):")
    for i, sq in enumerate(sub_questions, 1):
        print(f"   {i}. {sq}")

    # Step 2: Use solver to answer each sub-question
    print(f"\n🔧 STEP 2: SOLVER - Answering each sub-question...")
    sub_answers = []
    for i, sub_q in enumerate(sub_questions, 1):
        print(f"\n--- Sub-question {i}/{len(sub_questions)} ---")
        print(f"❓ Question: {sub_q}")
        
        solver_prompt = f"Question: {sub_q}\nContext: {question}\nAnswer:"
        print(f"📤 Solver Prompt: {solver_prompt}")

        solver_inputs = tokenizer(
            solver_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            solver_outputs = model.solver.model.generate(
                **solver_inputs,
                max_new_tokens=max_new_tokens // len(sub_questions),
                temperature=temperature,
                num_beams=num_beams,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id
            )

        sub_answer = tokenizer.decode(
            solver_outputs[0][solver_inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        print(f"💡 Answer: {sub_answer}")
        sub_answers.append(sub_answer)

    # Step 3: Use solver to combine sub-answers for final answer
    print(f"\n🎯 STEP 3: FINAL SYNTHESIS - Combining sub-answers...")
    combined_prompt = f"Question: {question}\n\n"
    for i, (sub_q, sub_a) in enumerate(zip(sub_questions, sub_answers)):
        combined_prompt += f"Sub-question {i + 1}: {sub_q}\nAnswer: {sub_a}\n\n"
    combined_prompt += "Based on the above sub-questions and answers, what is the final answer to the original question?"

    print(f"📤 Final Synthesis Prompt:")
    print(f"{combined_prompt}")

    solver_inputs = tokenizer(
        combined_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    ).to(device)

    with torch.no_grad():
        solver_outputs = model.solver.model.generate(
            **solver_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_beams=num_beams,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id
        )

    final_answer = tokenizer.decode(
        solver_outputs[0][solver_inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    extracted_answer = extract_answer(dataset_type, final_answer)
    
    print(f"📥 Final Raw Answer:")
    print(f"   {final_answer}")
    print(f"🎯 EXTRACTED FINAL ANSWER: {extracted_answer}")
    print(f"{'='*80}\n")

    return {
        "sub_questions": sub_questions,
        "sub_answers": sub_answers,
        "final_answer": final_answer,
        "extracted_answer": extracted_answer
    }


def main():
    # Configuration Variables (no argparse - direct variables)
    MODEL_PATH = "outputs/final"  # Path to your trained model
    DATASET_TYPE = "boolean"  # Type of dataset for answer extraction
    TEST_FILE = None  # Path to test file (optional)
    OUTPUT_FILE = "results.json"  # Path to output file
    BATCH_SIZE = 4  # Batch size for inference
    MAX_NEW_TOKENS = 512  # Maximum number of new tokens to generate
    TEMPERATURE = 0.7  # Temperature for generation
    NUM_BEAMS = 1  # Number of beams for beam search
    DO_SAMPLE = True  # Whether to use sampling for generation
    MAX_EXAMPLES = 10  # Maximum number of examples to process (set to None for all)

    print(f"Starting inference with model from: {MODEL_PATH}")

    # Load test data
    test_data = load_dataset("wics/strategy-qa", split="test")
    test_data = test_data.train_test_split(test_size=0.20, seed=42)
    test_data = test_data["test"]
    
    # Limit examples if specified
    if MAX_EXAMPLES:
        test_data = test_data.select(range(min(MAX_EXAMPLES, len(test_data))))
    
    print(f"Processing {len(test_data)} examples")

    # Load the unified trained model using saved configuration
    print(f"Loading SocraticMAGDi model from {MODEL_PATH}")

    try:
        model, model_config = load_model_from_config(MODEL_PATH)
        print("Successfully loaded model with saved configuration")

        # Print loaded configuration for verification
        print(f"Model configuration:")
        print(f"  Decomposer: {model_config['decomposer_name']}")
        print(f"  Solver: {model_config['solver_name']}")
        print(f"  GCN dimensions: {model_config['gcn_in_channels']} -> {model_config['gcn_hidden_channels']} -> {model_config['gcn_out_channels']}")
        print(f"  Loss weights: α={model_config['alpha']}, β={model_config['beta']}, γ={model_config['gamma']}, δ={model_config['delta']}")

    except Exception as e:
        print(f"Error: Failed to load model from config: {e}")
        print("Falling back to default configuration")

        # Fallback to default configuration (matching your training setup)
        model = SocraticMAGDi(
            decomposer_name="Qwen/Qwen2-1.5B",
            solver_name="Qwen/Qwen2-1.5B",
            gcn_in_channels=768,
            gcn_hidden_channels=256,
            gcn_out_channels=4,
            alpha=1.0,
            beta=1.0,
            gamma=0.1,
            delta=0.5
        )

        # Try to load weights anyway
        model_state_path = os.path.join(MODEL_PATH, "pytorch_model.bin")
        if os.path.exists(model_state_path):
            model.load_state_dict(torch.load(model_state_path, map_location="cpu"), strict=False)
            print("Loaded model weights with default configuration")
        
        model_config = {
            "decomposer_name": "Qwen/Qwen2-1.5B",
            "solver_name": "Qwen/Qwen2-1.5B",
            "gcn_in_channels": 768,
            "gcn_hidden_channels": 256,
            "gcn_out_channels": 4,
            "alpha": 1.0,
            "beta": 1.0,
            "gamma": 0.1,
            "delta": 0.5
        }

    # Load tokenizer from the model path (saved by training script)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    except:
        # Fallback to loading from model name if path doesn't work
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Process test examples
    print("Processing test examples with detailed output...")
    results = []
    predictions = []
    references = []
    per_example_times = []

    for idx, example in enumerate(test_data):
        question = example["question"]
        reference = example["answer"]  # StrategyQA uses "answer" field

        # Perform Socratic inference using unified model with detailed printing
        start = time.time()
        result = socratic_inference_unified(
            question,
            model,
            tokenizer,
            MAX_NEW_TOKENS,
            TEMPERATURE,
            NUM_BEAMS,
            DO_SAMPLE,
            DATASET_TYPE,
            example_num=idx + 1
        )
        elapsed = time.time() - start
        per_example_times.append(elapsed)

        result["time_taken"] = elapsed

        # Print comparison with ground truth
        is_correct = "✅ CORRECT" if str(result["extracted_answer"]).lower().strip() == str(reference).lower().strip() else "❌ WRONG"
        print(f"🏆 GROUND TRUTH: {reference}")
        print(f"🎯 MODEL PREDICTION: {result['extracted_answer']}")
        print(f"📊 RESULT: {is_correct}")
        print(f"⏱️  TIME TAKEN: {elapsed:.2f}s")

        results.append({
            "question": question,
            "reference": reference,
            "prediction": result["extracted_answer"],
            "sub_questions": result["sub_questions"],
            "sub_answers": result["sub_answers"],
            "final_answer": result["final_answer"],
            "time_taken": elapsed,
            "is_correct": is_correct == "✅ CORRECT"
        })

        predictions.append(result["extracted_answer"])
        references.append(reference)

    total_time = sum(per_example_times)
    avg_time = total_time / len(per_example_times)

    # Calculate accuracy
    accuracy = evaluate_accuracy(DATASET_TYPE, predictions, references)
    
    print(f"\n{'='*50}")
    print(f"🎊 FINAL RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"📊 Overall accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"⏱️  Avg time per example: {avg_time:.3f}s")
    print(f"🚀 Efficiency (acc/sec): {accuracy/avg_time:.4f}")
    print(f"✅ Correct predictions: {sum(predictions[i] == references[i] for i in range(len(predictions)))}/{len(predictions)}")

    eff = accuracy / avg_time

    # Save results
    output = {
        "results": results,
        "accuracy": accuracy,
        "avg_time": avg_time,
        "efficiency": eff,
        "total_examples": len(results),
        "metadata": {
            "model_path": MODEL_PATH,
            "dataset_type": DATASET_TYPE,
            "temperature": TEMPERATURE,
            "num_beams": NUM_BEAMS,
            "do_sample": DO_SAMPLE,
            "model_config": model_config
        }
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"💾 Results saved to {OUTPUT_FILE}")

    # Print summary of correct vs incorrect predictions
    print(f"\n📈 PREDICTION BREAKDOWN:")
    correct_count = sum(1 for r in results if r["is_correct"])
    print(f"✅ Correct: {correct_count}")
    print(f"❌ Incorrect: {len(results) - correct_count}")


if __name__ == "__main__":
    main()
