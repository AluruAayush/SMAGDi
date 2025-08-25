import os
import json
import torch
import argparse
import numpy as np
import networkx as nx
import pickle
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed
)

#from smodel import SocraticMAGDi, SocraticMAGDiDataCollator
from datasets import load_dataset
import random
from collections import Counter
import re
import sys
import tensorboardX
import sklearn
from transformers import EarlyStoppingCallback
from transformers import IntervalStrategy

class MAGDiDataset(Dataset):
    """
    Dataset for training SocraticMAGDi with variable-length lists for each example.
    """
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            "decomposer": [
                {
                    "prompt_input_ids": d["prompt_input_ids"],
                    "prompt_attention_mask": d["prompt_attention_mask"],
                    "completion_input_ids": d["completion_input_ids"],
                    "completion_attention_mask": d.get("completion_attention_mask", None)
                }
                for d in example["decomposer"]
            ],
            "solver": [
                {
                    "prompt_input_ids": s["prompt_input_ids"],
                    "prompt_attention_mask": s["prompt_attention_mask"],
                    "completion_input_ids": s["completion_input_ids"],
                    "completion_attention_mask": s.get("completion_attention_mask", None)
                }
                for s in example["solver"]
            ],
            "pos": [
                {
                    "input_ids": p["input_ids"],
                    "attention_mask": p["attention_mask"],
                    "labels": p["labels"]
                }
                for p in example["pos"]
            ],
            "neg": [
                {
                    "input_ids": n["input_ids"],
                    "attention_mask": n["attention_mask"],
                    "labels": n["labels"]
                }
                for n in example["neg"]
            ],
            "graph": example["graph"]
        }
class SocraticMAGDiTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(
            decomposer=inputs["decomposer"],
            solver=inputs["solver"],
            pos=inputs["pos"],
            neg=inputs["neg"],
            graph=inputs["graph"]
        )
        
        # Extract the total loss
        total_loss = outputs["loss"]
        
        if return_outputs:
            # Ensure outputs contain 'loss' for evaluation
            outputs_dict = {
                "loss": total_loss,
                "lm_loss": outputs.get("lm_loss", total_loss),
                "node_loss": outputs.get("node_loss", 0),
                "mr_loss": outputs.get("mr_loss", 0),
                "alignment_loss": outputs.get("alignment_loss", 0)
            }
            return total_loss, outputs_dict
        return total_loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Custom prediction step to ensure loss is computed during evaluation.
        """
        model.eval()
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            
        if prediction_loss_only:
            return (loss, None, None)
            
        # Return loss, logits (if available), and labels (if available)
        return (loss, None, None)


def prepare_training_examples(dataset, tokenizer, max_length=512):
    """
    Prepares decomposer, solver, positive, and negative examples for MAGDi-style multi-agent reasoning.
    - Positive: Correct reasoning chains (final answer == gold).
    - Negative: Incorrect reasoning chains (final answer != gold), with synthetic negatives if needed.
    - Decomposer: Always synthetic, model-inspired decomposition steps per question.
    - Reasoning chains are built as sequences of nodes connected by influence, and treated as positive/negative.
    - Dummy values are inserted for missing positive/negative examples.

    Returns:
        tuple: (decomposer_examples, solver_examples, positive_examples, negative_examples)
               Each is a list of lists, outer length = len(dataset), inner = examples for that item.
    """
    decomposer_examples = []
    solver_examples = []
    positive_examples = []
    negative_examples = []

    dummy_tensor = torch.zeros(max_length, dtype=torch.long)

    for item in tqdm(dataset, desc="Preparing Training Examples"):
        item_decomposer = []
        item_solver = []
        item_pos = []
        item_neg = []

        question = item.get('question')
        gold_answer = str(item.get('gold_answer', '')).strip().lower()
        nx_graph = item.get('graph')
        pyg_data = item.get('pyg_data')

        # Handle missing data with dummy examples
        if not all([question, gold_answer, nx_graph, hasattr(nx_graph, 'nodes'), pyg_data]):
            dummy_example = {
                "input_ids": dummy_tensor,
                "attention_mask": dummy_tensor,
                "labels": dummy_tensor.clone(),
                "subgraph_x": dummy_tensor,
                "subgraph_edge_index": torch.zeros((2,0), dtype=torch.long)
            }
            dummy_decomposer = {
                "prompt_input_ids": dummy_tensor,
                "prompt_attention_mask": dummy_tensor,
                "completion_input_ids": dummy_tensor,
                "completion_attention_mask": dummy_tensor,
            }
            dummy_solver = {
                "prompt_input_ids": dummy_tensor,
                "prompt_attention_mask": dummy_tensor,
                "completion_input_ids": dummy_tensor,
                "completion_attention_mask": dummy_tensor,
            }
            decomposer_examples.append([dummy_decomposer])
            solver_examples.append([dummy_solver])
            positive_examples.append([dummy_example])
            negative_examples.append([dummy_example])
            continue

        # --- Decomposer: Always generate a synthetic decomposition ---
        decomposer_prompt = (
            f"Decompose the following question into a sequence of simpler sub-questions that, when answered, "
            f"would help solve the main question:\n\nQuestion: {question}"
        )
        decomposer_completion = (
            "1. What are the key facts or entities in the question?\n"
            "2. What is being asked or claimed?\n"
            "3. What evidence or reasoning links the facts to the answer?\n"
            "4. What is the final answer based on the above?"
        )
        prompt_tok = tokenizer(decomposer_prompt, max_length=max_length, truncation=True, return_tensors="pt")
        comp_tok = tokenizer(decomposer_completion, max_length=max_length, truncation=True, return_tensors="pt")
        item_decomposer.append({
            "prompt_input_ids": prompt_tok.input_ids.squeeze(0),
            "prompt_attention_mask": prompt_tok.attention_mask.squeeze(0),
            "completion_input_ids": comp_tok.input_ids.squeeze(0),
            "completion_attention_mask": comp_tok.attention_mask.squeeze(0)
        })
        solver_prompt = (
           "Answer the decompositions similar to the agent's responses"
        )
        solver_completion = (
            "Answer thoroughly"
        )
        prompt_tok = tokenizer(solver_prompt, max_length=max_length, truncation=True, return_tensors="pt")
        comp_tok = tokenizer(solver_completion, max_length=max_length, truncation=True, return_tensors="pt")
        item_decomposer.append({
            "prompt_input_ids": prompt_tok.input_ids.squeeze(0),
            "prompt_attention_mask": prompt_tok.attention_mask.squeeze(0),
            "completion_input_ids": comp_tok.input_ids.squeeze(0),
            "completion_attention_mask": comp_tok.attention_mask.squeeze(0)
        })

        # --- Build reasoning chains and process solver examples ---
        def build_chains(node_id, current_chain=[]):
            chain = current_chain + [node_id]
            influencers = nx_graph.nodes[node_id].get('influenced_by', [])
            if not influencers:
                return [chain]
            return [c for inf in influencers for c in build_chains(inf, chain.copy())]

        for node_id, node_data in nx_graph.nodes(data=True):
            if node_data.get("type") not in ["initial_response", "response"]:
                continue

            # Create solver example for this individual node
            analysis = node_data.get("content", "")
            role = node_data.get("role", "Agent")
            round_num = node_data.get("round", 0)
            influencers = node_data.get("influenced_by", [])
            influence_str = f"\nAfter considering input from: {', '.join(influencers)}" if influencers else ""

            solver_prompt = (
                f"Question: {question}\n\nRole: {role}\nRound: {round_num}{influence_str}\n\n"
                f"Provide your detailed analysis and final decision."
            )
            prompt_tok = tokenizer(solver_prompt, max_length=max_length, truncation=True, return_tensors="pt")
            comp_tok = tokenizer(analysis, max_length=max_length, truncation=True, return_tensors="pt")
            item_solver.append({
                "prompt_input_ids": prompt_tok.input_ids.squeeze(0),
                "prompt_attention_mask": prompt_tok.attention_mask.squeeze(0),
                "completion_input_ids": comp_tok.input_ids.squeeze(0),
                "completion_attention_mask": comp_tok.attention_mask.squeeze(0)
            })

            # Process reasoning chains for pos/neg examples
            chains = build_chains(node_id)
            for chain in chains:
                chain_is_positive = True
                chain_text = []
                chain_nodes = []

                for n in chain:
                    n_data = nx_graph.nodes[n]
                    decision = (n_data.get("decision") or "").strip().lower()
                    if decision and decision in ["true", "false"] and decision != gold_answer:
                        chain_is_positive = False
                    chain_text.append(n_data.get("content", ""))
                    chain_nodes.append(n)

                full_chain = "\n".join(chain_text)
                tokenized_chain = tokenizer(full_chain, max_length=max_length, truncation=True, return_tensors="pt")
                example_dict = {
                    "input_ids": tokenized_chain.input_ids.squeeze(0),
                    "attention_mask": tokenized_chain.attention_mask.squeeze(0),
                    "labels": tokenized_chain.input_ids.squeeze(0).clone(),
                    "chain_nodes": chain_nodes,
                    "subgraph_x": pyg_data.x,
                    "subgraph_edge_index": pyg_data.edge_index,
                }

                if chain_is_positive:
                    item_pos.append(example_dict)
                else:
                    item_neg.append(example_dict)
        if not item_pos:
            item_pos.append({
                "input_ids": dummy_tensor,
                "attention_mask": dummy_tensor,
                "labels": dummy_tensor.clone(),
                "chain_nodes": [],
                "subgraph_x": pyg_data.x,
                "subgraph_edge_index": pyg_data.edge_index,
            })
        if not item_neg:
            item_neg.append({
                "input_ids": dummy_tensor,
                "attention_mask": dummy_tensor,
                "labels": dummy_tensor.clone(),
                "chain_nodes": [],
                "subgraph_x": pyg_data.x,
                "subgraph_edge_index": pyg_data.edge_index,
            })

        decomposer_examples.append(item_decomposer)
        solver_examples.append(item_solver)
        positive_examples.append(item_pos)
        negative_examples.append(item_neg)

    return decomposer_examples, solver_examples, positive_examples, negative_examples

def main():
    # Configuration Variables (Replace all args with direct variables)
    torch.cuda.empty_cache()
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    SEED = 42
    MODEL_SIZE = "small"  # choices: "small", "medium", "large"
    DECOMPOSER_MODEL = "meta-llama/Llama-3.2-3B"
    SOLVER_MODEL = "meta-llama/Llama-3.2-3B"
    OUTPUT_DIR = "outputs"
    
    # GCN Configuration
    GCN_IN_CHANNELS = 768
    GCN_HIDDEN_CHANNELS = 256
    GCN_OUT_CHANNELS = 4
    
    # Loss weights
    ALPHA = 1.0  # Weight for language modeling loss
    BETA = 1.0   # Weight for node classification loss
    GAMMA = 0.1  # Weight for contrastive loss
    DELTA = 0.5  # Weight for decomposer-solver alignment loss
    
    # Training configuration
    NUM_EPOCHS = 50
    BATCH_SIZE = 1
    LEARNING_RATE = 5e-5
    
    # OpenAI API Key    
    set_seed(SEED)
    # Load the saved MAG dataset from the pickle file
    with open('data/mag_dataset_refined.pkl', 'rb') as f:
        mag_dataset = pickle.load(f)
        
    for i in range(len(mag_dataset)):
        if mag_dataset[i]["is_correct"] == None:
            mag_dataset.drop(i)
        

    # Split MAG dataset into train/val
    from sklearn.model_selection import train_test_split
    mag_train, mag_val = train_test_split(mag_dataset, test_size=0.1, random_state=42)

    # Initialize tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(DECOMPOSER_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare training and validation examples
    decomposer_examples_train, solver_examples_train, pos_examples_train, neg_examples_train = prepare_training_examples(mag_train, tokenizer, max_length=512)
    decomposer_examples_val, solver_examples_val, pos_examples_val, neg_examples_val = prepare_training_examples(mag_val, tokenizer, max_length=512)

    pyg_graphs_train = [item["pyg_data"] for item in mag_train]
    pyg_graphs_val = [item["pyg_data"] for item in mag_val]

    # Build train and val datasets
    examples_train = []
    for i in range(len(mag_train)):
        examples_train.append({
            "decomposer": decomposer_examples_train[i],
            "solver": solver_examples_train[i],
            "pos": pos_examples_train[i],
            "neg": neg_examples_train[i],
            "graph": pyg_graphs_train[i]
        })
    train_dataset = MAGDiDataset(examples_train)

    examples_val = []
    for i in range(len(mag_val)):
        examples_val.append({
            "decomposer": decomposer_examples_val[i],
            "solver": solver_examples_val[i],
            "pos": pos_examples_val[i],
            "neg": neg_examples_val[i],
            "graph": pyg_graphs_val[i]
        })
    val_dataset = MAGDiDataset(examples_val)

    print(f"Final train dataset size: {len(train_dataset)}")
    print(f"Final validation dataset size: {len(val_dataset)}")

    # Initialize model
    print("Initializing SocraticMAGDi model")
    model = SocraticMAGDi(
        decomposer_name=DECOMPOSER_MODEL,
        solver_name=SOLVER_MODEL,
        gcn_in_channels=GCN_IN_CHANNELS,
        gcn_hidden_channels=GCN_HIDDEN_CHANNELS,
        gcn_out_channels=GCN_OUT_CHANNELS,
        alpha=ALPHA,
        beta=BETA,
        gamma=GAMMA,
        delta=DELTA
    )
    # TrainingArguments with validation enabled
    from transformers import TrainingArguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=8,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=32,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=1,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        save_safetensors=False,
        dataloader_pin_memory=False,
        fp16=False,
        gradient_checkpointing=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        per_device_eval_batch_size=1, # <-- For validation
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        eval_strategy=IntervalStrategy.EPOCH,  # <-- Use this in new versions
        save_strategy=IntervalStrategy.EPOCH,
        save_total_limit=1,          # Keep only 1 checkpoint
    )

    # Trainer with validation set
    trainer = SocraticMAGDiTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # <-- Validation set here
        data_collator=SocraticMAGDiDataCollator(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Train model
    print("Training SocraticMAGDi model")
    trainer.train()

    # Enhanced model saving
    print(f"Saving complete model to {OUTPUT_DIR}/final")
    final_dir = f"{OUTPUT_DIR}/final"
    os.makedirs(final_dir, exist_ok=True)

    # Save the complete model state dict
    model_state_dict = model.state_dict()
    torch.save(model_state_dict, os.path.join(final_dir, "pytorch_model.bin"))

    # Save model configuration
    model_config = {
        "decomposer_name": DECOMPOSER_MODEL,
        "solver_name": SOLVER_MODEL,
        "gcn_in_channels": GCN_IN_CHANNELS,
        "gcn_hidden_channels": GCN_HIDDEN_CHANNELS,
        "gcn_out_channels": GCN_OUT_CHANNELS,
        "alpha": ALPHA,
        "beta": BETA,
        "gamma": GAMMA,
        "delta": DELTA,
        "model_type": "SocraticMAGDi",
        "torch_dtype": "float32",
        "transformers_version": "4.36.0"
    }

    with open(os.path.join(final_dir, "config.json"), "w") as f:
        json.dump(model_config, f, indent=2)

    # Save tokenizer
    tokenizer.save_pretrained(final_dir)

    # Save individual component states
    components_dir = os.path.join(final_dir, "components")
    os.makedirs(components_dir, exist_ok=True)

    # Save decomposer component
    decomposer_state = {
        "model_state_dict": model.decomposer.model.state_dict(),
        "config": model.decomposer.model.config.to_dict() if hasattr(model.decomposer.model, 'config') else {}
    }
    torch.save(decomposer_state, os.path.join(components_dir, "decomposer.bin"))

    # Save solver component
    solver_state = {
        "model_state_dict": model.solver.model.state_dict(),
        "config": model.solver.model.config.to_dict() if hasattr(model.solver.model, 'config') else {}
    }
    torch.save(solver_state, os.path.join(components_dir, "solver.bin"))

    # Save GCN component
    gcn_state = {
        "model_state_dict": model.gcn.state_dict(),
        "in_channels": GCN_IN_CHANNELS,
        "hidden_channels": GCN_HIDDEN_CHANNELS,
        "out_channels": GCN_OUT_CHANNELS
    }
    torch.save(gcn_state, os.path.join(components_dir, "gcn.bin"))

    print("Model saved successfully with all components!")
    print(f"Main model: {os.path.join(final_dir, 'pytorch_model.bin')}")
    print(f"Config: {os.path.join(final_dir, 'config.json')}")
    print(f"Tokenizer: {final_dir}")
    print(f"Components: {components_dir}")
    print("Training complete!")

    return model, tokenizer, final_dir

# Run the training
if __name__ == "__main__":
    model, tokenizer, model_path = main()
    print(f"Final model saved at: {model_path}")
