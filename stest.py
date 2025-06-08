import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
import time
import re

# Import the unified model
#from smodel import SocraticMAGDi

def extract_answer_robust(dataset_type, text):
    """Extract answer with robust boolean handling - always returns true/false."""
    import re
    
    if dataset_type in ["boolean", "strategy-qa"]:
        # Convert text to lowercase for better matching
        text_lower = text.lower().strip()
        
        # Priority 1: Look for explicit true/false patterns
        true_patterns = [
            r"(?:answer|result|solution|conclusion)(?:\s+is\s+|\s*:\s*|\s+)(?:yes|true)",
            r"\b(?:yes|true)\b(?:\s*[.,!]?\s*$)",
            r"{{(?:yes|true)}}",
            r"the\s+answer\s+is\s+(?:yes|true)",
            r"(?:definitely|certainly|absolutely)?\s*(?:yes|true)"
        ]
        
        false_patterns = [
            r"(?:answer|result|solution|conclusion)(?:\s+is\s+|\s*:\s*|\s+)(?:no|false)",
            r"\b(?:no|false)\b(?:\s*[.,!]?\s*$)",
            r"{{(?:no|false)}}",
            r"the\s+answer\s+is\s+(?:no|false)",
            r"(?:definitely|certainly|absolutely)?\s*(?:no|false)"
        ]
        
        # Check for true patterns
        for pattern in true_patterns:
            if re.search(pattern, text_lower):
                return "true"
        
        # Check for false patterns  
        for pattern in false_patterns:
            if re.search(pattern, text_lower):
                return "false"
        
        # Priority 2: Look for affirmative vs negative language
        affirmative_words = ["yes", "correct", "right", "accurate", "valid", "possible", "likely", "can", "will", "does", "is"]
        negative_words = ["no", "incorrect", "wrong", "inaccurate", "invalid", "impossible", "unlikely", "cannot", "won't", "doesn't", "isn't"]
        
        # Count affirmative vs negative sentiment
        affirmative_count = sum(1 for word in affirmative_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if affirmative_count > negative_count:
            return "true"
        elif negative_count > affirmative_count:
            return "false"
        
        # Priority 3: Default based on text length and complexity
        # Longer, more detailed answers tend to be affirmative
        if len(text.split()) > 20:
            return "true"
        else:
            return "false"
    
    # For other dataset types, use original logic
    return extract_answer_original(dataset_type, text)

def extract_answer_original(dataset_type, text):
    """Original answer extraction for non-boolean datasets."""
    import re

    # For multiple choice questions
    if dataset_type in ["multiple_choice", "classification"]:
        answer_pattern = r"(?:answer|option)(?:\s+is\s+|\s*:\s*)([A-E])"
        match = re.search(answer_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        answer_pattern = r"\{\{([A-E])\}\}"
        match = re.search(answer_pattern, text)
        if match:
            return match.group(1).upper()

        last_lines = text.strip().split("\n")[-3:]
        last_text = " ".join(last_lines)
        options = re.findall(r"\b([A-E])\b", last_text)
        if options:
            return options[-1].upper()

    # For numerical answers
    elif dataset_type in ["numerical", "math"]:
        answer_pattern = r"(?:answer|result|solution)(?:\s+is\s+|\s*=\s*)([+-]?\d+(?:\.\d+)?)"
        match = re.search(answer_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)

        answer_pattern = r"\{\{([+-]?\d+(?:\.\d+)?)\}\}"
        match = re.search(answer_pattern, text)
        if match:
            return match.group(1)

        numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", text)
        if numbers:
            return numbers[-1]

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
            try:
                pred_val = float(pred)
                ref_val = float(ref)
                if abs(pred_val - ref_val) < 1e-6:
                    correct += 1
            except ValueError:
                pass
        else:
            if str(pred).strip().upper() == str(ref).strip().upper():
                correct += 1

    return correct / total if total > 0 else 0

def load_model_from_config(model_path):
    """Load SocraticMAGDi model using saved configuration"""
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

    model_state_path = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(model_state_path):
        model.load_state_dict(torch.load(model_state_path, map_location="cpu"))
        print("Loaded model weights from saved checkpoint")
    else:
        print("Warning: No saved weights found, using initialized model")

    return model, model_config

def socratic_inference_unified(question, model, tokenizer, max_new_tokens, temperature, num_beams, do_sample, dataset_type, example_num=None):
    """Perform inference with forced boolean output."""
    device = next(model.parameters()).device
    
    print(f"\n{'='*80}")
    if example_num:
        print(f"🔍 EXAMPLE {example_num}")
    print(f"📝 ORIGINAL QUESTION:")
    print(f"   {question}")
    print(f"{'='*80}")

    # Step 1: Enhanced decomposer prompt for boolean questions
    print(f"🧩 STEP 1: DECOMPOSER - Breaking down the question...")
    if dataset_type in ["boolean", "strategy-qa"]:
        decomposer_prompt = f"Question: {question}\nThis is a yes/no question. Break this down into key sub-questions that will help determine if the answer is TRUE or FALSE:"
    else:
        decomposer_prompt = f"Question: {question}\nBreak this down into sub-questions:"
    
    print(f"📤 Decomposer Prompt: {decomposer_prompt}")

    decomposer_inputs = tokenizer(
        decomposer_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

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

    if not sub_questions:
        if dataset_type in ["boolean", "strategy-qa"]:
            sub_questions = [
                "What evidence supports a TRUE answer?",
                "What evidence supports a FALSE answer?",
                "What is the most logical conclusion?"
            ]
        else:
            sub_questions = [
                "What are the key components of this problem?",
                "What approach should I use to solve this problem?"
            ]

    print(f"🔗 EXTRACTED SUB-QUESTIONS ({len(sub_questions)}):")
    for i, sq in enumerate(sub_questions, 1):
        print(f"   {i}. {sq}")

    # Step 2: Enhanced solver prompts for boolean questions
    print(f"\n🔧 STEP 2: SOLVER - Answering each sub-question...")
    sub_answers = []
    for i, sub_q in enumerate(sub_questions, 1):
        print(f"\n--- Sub-question {i}/{len(sub_questions)} ---")
        print(f"❓ Question: {sub_q}")
        
        if dataset_type in ["boolean", "strategy-qa"]:
            solver_prompt = f"Question: {sub_q}\nContext: {question}\nProvide a clear answer that helps determine if the main question is TRUE or FALSE. Answer:"
        else:
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

    # Step 3: Enhanced final synthesis for boolean questions
    print(f"\n🎯 STEP 3: FINAL SYNTHESIS - Combining sub-answers...")
    
    if dataset_type in ["boolean", "strategy-qa"]:
        combined_prompt = f"Question: {question}\n\n"
        for i, (sub_q, sub_a) in enumerate(zip(sub_questions, sub_answers)):
            combined_prompt += f"Sub-question {i + 1}: {sub_q}\nAnswer: {sub_a}\n\n"
        combined_prompt += "Based on the analysis above, is the answer to the original question TRUE or FALSE? Provide your final answer as either 'TRUE' or 'FALSE' and explain your reasoning:"
    else:
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

    # Use robust extraction for boolean answers
    extracted_answer = extract_answer_robust(dataset_type, final_answer)
    
    print(f"📥 Final Raw Answer:")
    print(f"   {final_answer}")
    print(f"🎯 EXTRACTED FINAL ANSWER: {extracted_answer}")
    
    # Validate boolean answer
    if dataset_type in ["boolean", "strategy-qa"]:
        if extracted_answer not in ["true", "false"]:
            print(f"⚠️ WARNING: Non-boolean answer detected! Forcing to boolean...")
            # Fallback: analyze sentiment of final answer
            if any(word in final_answer.lower() for word in ["yes", "true", "correct", "right", "possible"]):
                extracted_answer = "true"
            else:
                extracted_answer = "false"
            print(f"🔧 CORRECTED ANSWER: {extracted_answer}")
    
    print(f"{'='*80}\n")

    return {
        "sub_questions": sub_questions,
        "sub_answers": sub_answers,
        "final_answer": final_answer,
        "extracted_answer": extracted_answer
    }

def main():
    # Configuration Variables
    MODEL_PATH = "outputs/final"
    DATASET_TYPE = "boolean"  # This ensures boolean handling
    OUTPUT_FILE = "results.json"
    MAX_NEW_TOKENS = 512
    TEMPERATURE = 0.3  # Lower temperature for more consistent boolean answers
    NUM_BEAMS = 1
    DO_SAMPLE = True
    MAX_EXAMPLES = 10

    print(f"Starting BOOLEAN inference with model from: {MODEL_PATH}")

    # Load test data
    test_data = load_dataset("wics/strategy-qa", split="test")
    test_data = test_data.train_test_split(test_size=0.20, seed=42)
    test_data = test_data["test"]
    
    if MAX_EXAMPLES:
        test_data = test_data.select(range(min(MAX_EXAMPLES, len(test_data))))
    
    print(f"Processing {len(test_data)} examples for TRUE/FALSE classification")

    # Load model (keeping your existing loading logic)
    print(f"Loading SocraticMAGDi model from {MODEL_PATH}")

    try:
        model, model_config = load_model_from_config(MODEL_PATH)
        print("Successfully loaded model with saved configuration")
    except Exception as e:
        print(f"Error: Failed to load model from config: {e}")
        print("Falling back to default configuration")

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

        model_state_path = os.path.join(MODEL_PATH, "pytorch_model.bin")
        if os.path.exists(model_state_path):
            model.load_state_dict(torch.load(model_state_path, map_location="cpu"), strict=False)
            print("Loaded model weights with default configuration")
        
        model_config = {
            "decomposer_name": "Qwen/Qwen2-1.5B",
            "solver_name": "Qwen/Qwen2-1.5B"
        }

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    except:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Process test examples
    print("Processing test examples with FORCED TRUE/FALSE output...")
    results = []
    predictions = []
    references = []
    per_example_times = []

    for idx, example in enumerate(test_data):
        question = example["question"]
        reference = example["answer"]

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

        # Ensure boolean output
        extracted = result["extracted_answer"]
        if extracted not in ["true", "false"]:
            print(f"🚨 CRITICAL: Non-boolean output detected: {extracted}")
            extracted = "false"  # Default fallback

        # Print comparison with ground truth
        ref_str = "true" if reference else "false"
        is_correct = "✅ CORRECT" if extracted == ref_str else "❌ WRONG"
        
        print(f"🏆 GROUND TRUTH: {ref_str}")
        print(f"🎯 MODEL PREDICTION: {extracted}")
        print(f"📊 RESULT: {is_correct}")
        print(f"⏱️  TIME TAKEN: {elapsed:.2f}s")

        results.append({
            "question": question,
            "reference": ref_str,
            "prediction": extracted,
            "sub_questions": result["sub_questions"],
            "sub_answers": result["sub_answers"],
            "final_answer": result["final_answer"],
            "time_taken": elapsed,
            "is_correct": is_correct == "✅ CORRECT"
        })

        predictions.append(extracted)
        references.append(ref_str)

    # Calculate final metrics
    total_time = sum(per_example_times)
    avg_time = total_time / len(per_example_times)
    accuracy = evaluate_accuracy(DATASET_TYPE, predictions, references)
    
    print(f"\n{'='*50}")
    print(f"🎊 BOOLEAN CLASSIFICATION RESULTS")
    print(f"{'='*50}")
    print(f"📊 Overall accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"⏱️  Avg time per example: {avg_time:.3f}s")
    print(f"✅ Correct TRUE/FALSE predictions: {sum(p == r for p, r in zip(predictions, references))}/{len(predictions)}")
    
    # Boolean-specific metrics
    true_predictions = sum(1 for p in predictions if p == "true")
    false_predictions = sum(1 for p in predictions if p == "false")
    print(f"🔄 Prediction distribution: TRUE={true_predictions}, FALSE={false_predictions}")

    # Save results
    output = {
        "results": results,
        "accuracy": accuracy,
        "avg_time": avg_time,
        "total_examples": len(results),
        "prediction_distribution": {
            "true": true_predictions,
            "false": false_predictions
        },
        "metadata": {
            "model_path": MODEL_PATH,
            "dataset_type": DATASET_TYPE,
            "forced_boolean": True,
            "temperature": TEMPERATURE,
            "model_config": model_config
        }
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"💾 Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
