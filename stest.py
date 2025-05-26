import os
import json
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
import time

# Import the unified model
from smodel import SocraticMAGDi

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

def socratic_inference_unified(question, model, tokenizer, args):
    """Perform inference using the unified SocraticMAGDi model."""
    device = next(model.parameters()).device
    
    # Step 1: Use decomposer to break down the question
    decomposer_prompt = f"Question: {question}\nBreak this down into sub-questions:"
    
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
            max_new_tokens=args.max_new_tokens // 2,
            temperature=args.temperature,
            num_beams=args.num_beams,
            do_sample=args.do_sample,
            pad_token_id=tokenizer.eos_token_id
        )
    
    sub_questions_text = tokenizer.decode(
        decomposer_outputs[0][decomposer_inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    
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
    
    # Step 2: Use solver to answer each sub-question
    sub_answers = []
    for sub_q in sub_questions:
        solver_prompt = f"Question: {sub_q}\nContext: {question}\nAnswer:"
        
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
                max_new_tokens=args.max_new_tokens // len(sub_questions),
                temperature=args.temperature,
                num_beams=args.num_beams,
                do_sample=args.do_sample,
                pad_token_id=tokenizer.eos_token_id
            )
        
        sub_answer = tokenizer.decode(
            solver_outputs[0][solver_inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        sub_answers.append(sub_answer)
    
    # Step 3: Use solver to combine sub-answers for final answer
    combined_prompt = f"Question: {question}\n\n"
    for i, (sub_q, sub_a) in enumerate(zip(sub_questions, sub_answers)):
        combined_prompt += f"Sub-question {i+1}: {sub_q}\nAnswer: {sub_a}\n\n"
    combined_prompt += "Based on the above sub-questions and answers, what is the final answer to the original question?"
    
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
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            num_beams=args.num_beams,
            do_sample=args.do_sample,
            pad_token_id=tokenizer.eos_token_id
        )
    
    final_answer = tokenizer.decode(
        solver_outputs[0][solver_inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    
    return {
        "sub_questions": sub_questions,
        "sub_answers": sub_answers,
        "final_answer": final_answer,
        "extracted_answer": extract_answer(args.dataset_type, final_answer)
    }

def main():
    parser = argparse.ArgumentParser(description="Test a trained SocraticMAGDi model")
    
    # Model configuration - now expects unified model path
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to trained SocraticMAGDi model directory")
    
    # Testing configuration
    parser.add_argument("--dataset_type", type=str, default="boolean", 
                        choices=["multiple_choice", "numerical", "math", "classification", "text", "boolean", "strategy-qa"],
                        help="Type of dataset for answer extraction")
    parser.add_argument("--test_file", type=str, default=None, 
                        help="Path to test file (optional)")
    parser.add_argument("--output_file", type=str, default="results.json", 
                        help="Path to output file")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="Batch size for inference")
    parser.add_argument("--max_new_tokens", type=int, default=512, 
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="Temperature for generation")
    parser.add_argument("--num_beams", type=int, default=1, 
                        help="Number of beams for beam search")
    parser.add_argument("--do_sample", action="store_true", 
                        help="Whether to use sampling for generation")
    
    args = parser.parse_args()
    
    # Load test data
    test_data = load_dataset("wics/strategy-qa", split="test")
    test_data = test_data.train_test_split(test_size=0.20, seed=42)
    test_data = test_data["test"]
    
    # Load the unified trained model
    logger.info(f"Loading SocraticMAGDi model from {args.model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the unified model
    # We need to reconstruct the model since it's a custom class
    model = SocraticMAGDi(
        decomposer_name=args.model_path,  # Will load from saved checkpoint
        solver_name=args.model_path,      # Will load from saved checkpoint
        gcn_in_channels=768,
        gcn_hidden_channels=256,
        gcn_out_channels=4,
        alpha=1.0,
        beta=1.0,
        gamma=0.1,
        delta=0.5
    )
    
    # Load the saved state dict
    model_state_path = os.path.join(args.model_path, "pytorch_model.bin")
    if os.path.exists(model_state_path):
        model.load_state_dict(torch.load(model_state_path, map_location="cpu"))
        logger.info("Loaded model weights from saved checkpoint")
    else:
        logger.warning("No saved weights found, using initialized model")
    
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # Process test examples
    logger.info("Processing test examples")
    results = []
    predictions = []
    references = []
    per_example_times = []
    
    for example in tqdm(test_data, desc="Inferencing"):
        question = example["question"]
        reference = example["answer"]  # StrategyQA uses "answer" field
        
        # Perform Socratic inference using unified model
        start = time.time()
        result = socratic_inference_unified(
            question, 
            model, 
            tokenizer, 
            args
        )
        elapsed = time.time() - start
        per_example_times.append(elapsed)

        result["time_taken"] = elapsed
        
        results.append({
            "question": question,
            "reference": reference,
            "prediction": result["extracted_answer"],
            "sub_questions": result["sub_questions"],
            "sub_answers": result["sub_answers"],
            "final_answer": result["final_answer"],
            "time_taken": elapsed
        })
        
        predictions.append(result["extracted_answer"])
        references.append(reference)

    total_time = sum(per_example_times)
    avg_time = total_time / len(per_example_times)
    
    # Calculate accuracy
    accuracy = evaluate_accuracy(args.dataset_type, predictions, references)
    logger.info(f"Overall accuracy: {accuracy:.4f}")
    logger.info(f"Avg time per example: {avg_time:.3f}s")

    eff = accuracy / avg_time
    logger.info(f"Efficiency (acc/sec): {eff:.4f}")
    
    # Save results
    output = {
        "results": results,
        "accuracy": accuracy,
        "avg_time": avg_time,
        "efficiency": eff,
        "metadata": {
            "model_path": args.model_path,
            "dataset_type": args.dataset_type,
            "temperature": args.temperature,
            "num_beams": args.num_beams,
            "do_sample": args.do_sample
        }
    }
    
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Results saved to {args.output_file}")
    
    # Print examples of predictions
    logger.info("\nExample predictions:")
    for i in range(min(5, len(results))):
        logger.info(f"Question: {results[i]['question']}")
        logger.info(f"Sub-questions: {results[i]['sub_questions']}")
        logger.info(f"Final answer: {results[i]['final_answer']}")
        logger.info(f"Prediction: {results[i]['prediction']}")
        logger.info(f"Reference: {results[i]['reference']}")
        logger.info("---")

if __name__ == "__main__":
    main()
