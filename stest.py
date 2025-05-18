import os
import json
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from dataset import load_dataset
import time

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
        if dataset_type in ["numerical", "math"]:
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
            if pred.strip().upper() == ref.strip().upper():
                correct += 1
    
    return correct / total if total > 0 else 0

def socratic_inference(question, decomposer, solver, tokenizer_decomposer, tokenizer_solver, args):
    """Perform inference using the Socratic model approach."""
    # Step 1: Decompose the question into sub-questions
    decomposer_prompt = f"Question: {question}\nBreak this down into sub-questions:"
    
    decomposer_inputs = tokenizer_decomposer(
        decomposer_prompt, 
        return_tensors="pt", 
        padding=True
    ).to(decomposer.device)
    
    decomposer_outputs = decomposer.generate(
        **decomposer_inputs,
        max_new_tokens=args.max_new_tokens // 2,
        temperature=args.temperature,
        num_beams=args.num_beams,
        do_sample=args.do_sample
    )
    
    sub_questions_text = tokenizer_decomposer.decode(
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
    
    # Step 2: Solve each sub-question
    sub_answers = []
    for sub_q in sub_questions:
        solver_prompt = f"Question: {sub_q}\nContext: {question}"
        
        solver_inputs = tokenizer_solver(
            solver_prompt, 
            return_tensors="pt", 
            padding=True
        ).to(solver.device)
        
        solver_outputs = solver.generate(
            **solver_inputs,
            max_new_tokens=args.max_new_tokens // len(sub_questions),
            temperature=args.temperature,
            num_beams=args.num_beams,
            do_sample=args.do_sample
        )
        
        sub_answer = tokenizer_solver.decode(
            solver_outputs[0][solver_inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        sub_answers.append(sub_answer)
    
    # Step 3: Combine sub-answers to solve the original question
    combined_prompt = f"Question: {question}\n\n"
    for i, (sub_q, sub_a) in enumerate(zip(sub_questions, sub_answers)):
        combined_prompt += f"Sub-question {i+1}: {sub_q}\nAnswer: {sub_a}\n\n"
    combined_prompt += "Based on the above sub-questions and answers, what is the final answer to the original question?"
    
    solver_inputs = tokenizer_solver(
        combined_prompt, 
        return_tensors="pt", 
        padding=True
    ).to(solver.device)
    
    solver_outputs = solver.generate(
        **solver_inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        num_beams=args.num_beams,
        do_sample=args.do_sample
    )
    
    final_answer = tokenizer_solver.decode(
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
    
    # Model configuration
    parser.add_argument("--decomposer_model", type=str, required=True, 
                        help="Path to trained decomposer model")
    parser.add_argument("--solver_model", type=str, required=True, 
                        help="Path to trained solver model")
    
    # Testing configuration
    parser.add_argument("--dataset_type", type=str, default="multiple_choice", 
                        choices=["multiple_choice", "numerical", "math", "classification", "text"],
                        help="Type of dataset for answer extraction")
    parser.add_argument("--test_file", type=str, required=True, 
                        help="Path to test file")
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
    
    # Load test data here
    test_data = load_dataset("commonsense_qa")
    test_data = test_data["test"].train_test_split(test_size=0.2, seed=42)
    
    # Load models and tokenizers
    logger.info("Loading decomposer model and tokenizer")
    tokenizer_decomposer = AutoTokenizer.from_pretrained(args.decomposer_model)
    decomposer = AutoModelForCausalLM.from_pretrained(args.decomposer_model).to("cuda")
    
    logger.info("Loading solver model and tokenizer")
    tokenizer_solver = AutoTokenizer.from_pretrained(args.solver_model)
    solver = AutoModelForCausalLM.from_pretrained(args.solver_model).to("cuda")
    
    # Ensure padding tokens are set
    if tokenizer_decomposer.pad_token is None:
        tokenizer_decomposer.pad_token = tokenizer_decomposer.eos_token
    if tokenizer_solver.pad_token is None:
        tokenizer_solver.pad_token = tokenizer_solver.eos_token
    
    # Process test examples
    logger.info("Processing test examples")
    results = []
    predictions = []
    references = []
    per_example_times = []
    
    for example in tqdm(test_data, desc="Inferencing"):
        question = example["question"]
        reference = example["answer"]
        
        # Perform Socratic inference
        start = time.time()
        result = socratic_inference(
            question, 
            decomposer, 
            solver, 
            tokenizer_decomposer, 
            tokenizer_solver, 
            args
        )
        elapsed = time.time()-start
        per_example_times.append(elapsed)

        result["time_taken"] = elapsed
        
        results.append({
            "question": question,
            "reference": reference,
            "prediction": result["extracted_answer"],
            "sub_questions": result["sub_questions"],
            "sub_answers": result["sub_answers"],
            "final_answer": result["final_answer"]
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

    logger.ingo(f"Efficiency (acc/sec): {eff:.4f}")
    
    # Save results
    output = {
        "results": results,
        "accuracy": accuracy,
        "avg_time": avg_time,
        "efficiency": eff,
        "metadata": {
            "decomposer_model": args.decomposer_model,
            "solver_model": args.solver_model,
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
