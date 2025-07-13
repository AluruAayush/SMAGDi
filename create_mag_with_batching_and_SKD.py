import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
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
    pipeline,
    set_seed
)
from datasets import load_dataset
import random
from collections import Counter
import re
import sys
import tensorboardX
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Agent role specifications (unchanged)
role_specifications = {
    "Scientist": {
        "temperature": 0.3,
        "instructions": """
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

RESPONSE FORMAT(DON'T PUT '''json before this):
{
  "decision": "<option>",
  "analysis": {
  "competing_hypotheses": ["<hyp1>", "<hyp2>"],
  "bayesian_analysis": "<P(H|E) calculations>",
  "red_team_critique": "<weaknesses in conclusion>",
  "confidence_level": "<percentage>",
  "evidence_quality": "<assessment>",
  "long_term_consequences": "<50+ year impact analysis>"
  }
}

Deviation from this format will exclude you from consensus.
"""
    },

    "Lawyer": {
        "temperature": 0.4,
        "instructions": """
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

RESPONSE FORMAT(DON'T PUT '''json before this):
{
  "decision": "<option>",
  "analysis": {
  "jurisdictional_conflicts": "<varied legal interpretations>",
  "settlement_equilibrium": "<Nash equilibrium analysis>",
  "multi_system_violations": "<potential cross-border conflicts>",
  "legal_risks": "<specific potential lawsuits>",
  "precedent_impact": "<what this allows in future>",
  "constitutional_analysis": "<rights implications>"
  }
}

Deviation from this format will exclude you from consensus.
"""
    },

    "Historian": {
        "temperature": 0.8,
        "instructions": """
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

RESPONSE FORMAT (DON'T PUT '''json before this):
{
  "decision": "<option>",
  "analysis": {
    "historical_context": "<relevant eras, events, or trends>",
    "precedents": "<similar decisions and their outcomes>",
    "key_actors": "<influential individuals, groups, or institutions>",
    "source_critique": "<assessment of historical evidence reliability>",
    "long_term_consequences": "<impacts observed over decades/centuries>",
    "historiographical_issues": "<how history is remembered/interpreted>"
  }
}

Deviation from this format will exclude you from consensus.
"""
    },

    "Mathematician": {
        "temperature": 0.25,
        "instructions": """
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

RESPONSE FORMAT(DON'T PUT '''json before this):
{
  "decision": "<option>",
  "analysis": {
  "frequentist_vs_bayesian": "<comparative analysis>",
  "uncertainty_cascade": "<error propagation visualization>",
  "adversarial_robustness": "<worst-case scenario math>",
  "probability_calculation": "<specific numbers and percentages>",
  "optimization_target": "<what you're maximizing/minimizing>",
  "monte_carlo_results": "<simulation outcomes>",
  "confidence_intervals": "<uncertainty bounds>"
  }
}

Deviation from this format will exclude you from consensus.
"""
    },

    "Ethicist": {
        "temperature": 0.5,
        "instructions": """
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

RESPONSE FORMAT(DON'T PUT '''json before this):
{
  "decision": "<option>",
  "analysis": {
  "moral_trade_offs": "<what values conflict>",
  "ethical_test_results": "<fairness, harm reduction, universalizability>",
  "utilitarian_analysis": "<greatest good calculation>",
  "deontological_analysis": "<duty-based assessment>",
  "virtue_ethics_analysis": "<character-based evaluation>",
  "future_obligations": "<intergenerational ethics>",
  "legitimacy_assessment": "<decision-maker authority evaluation>"
  }
}

Deviation from this format will exclude you from consensus.
"""
    }
}

# === Llama 3.1 8B Instruct Setup with Batch Processing ===
LLAMA_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

print("Loading Llama 3.1 8B Instruct model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_ID)
llama_pipeline = pipeline(
    "text-generation",
    model=LLAMA_MODEL_ID,
    tokenizer=tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto"
)

# Thread lock for model access
model_lock = threading.Lock()

def llama_generate(messages, temperature=0.7, max_tokens=600):
    """Single message generation - thread-safe"""
    with model_lock:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        )
        inputs = {k: v.to(llama_pipeline.model.device) for k, v in inputs.items()}
        outputs = llama_pipeline.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        )
        response = outputs[0][inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(response, skip_special_tokens=True)
def llama_generate_true_batch(messages_list, temperatures=None, max_tokens=600):
    """
    True batch inference - processes multiple prompts simultaneously
    This version handles the complexity of batching chat templates
    """
    if not messages_list:
        return []
    
    if temperatures is None:
        temperatures = [0.7] * len(messages_list)
    
    with model_lock:
        # Prepare batch inputs
        batch_inputs = []
        for messages in messages_list:
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True
            )
            batch_inputs.append(inputs)
        
        # Find max length for padding
        max_length = max(inp["input_ids"].shape[1] for inp in batch_inputs)
        
        # Pad all inputs to same length
        padded_input_ids = []
        padded_attention_masks = []
        
        for inputs in batch_inputs:
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            
            # Pad to max_length
            pad_length = max_length - input_ids.shape[1]
            if pad_length > 0:
                pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                input_ids = torch.cat([
                    torch.full((1, pad_length), pad_token_id, dtype=input_ids.dtype),
                    input_ids
                ], dim=1)
                attention_mask = torch.cat([
                    torch.zeros((1, pad_length), dtype=attention_mask.dtype),
                    attention_mask
                ], dim=1)
            
            padded_input_ids.append(input_ids)
            padded_attention_masks.append(attention_mask)
        
        # Stack into batch tensors
        batch_input_ids = torch.cat(padded_input_ids, dim=0).to(llama_pipeline.model.device)
        batch_attention_mask = torch.cat(padded_attention_masks, dim=0).to(llama_pipeline.model.device)
        
        # Generate for entire batch
        # Note: Using average temperature for simplicity - could be enhanced for per-sample temperature
        avg_temperature = sum(temperatures) / len(temperatures)
        
        outputs = llama_pipeline.model.generate(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=avg_temperature,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        )
        
        # Decode responses
        responses = []
        for i, output in enumerate(outputs):
            # Extract only the generated part (after input)
            generated_tokens = output[batch_input_ids[i].shape[0]:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append(response.strip())
        
        return responses

# === Dataset class (unchanged) ===
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

# === Batch-optimized agent analysis ===
def generate_analysis_batch(agents, prompts, debate_round=0):
    """
    Generate analysis for multiple agents in batch
    Args:
        agents: List of agent dictionaries
        prompts: List of prompts corresponding to each agent
        debate_round: Current debate round number
    Returns:
        List of responses corresponding to each agent
    """
    if len(agents) != len(prompts):
        raise ValueError("Number of agents must match number of prompts")
    
    # Prepare messages and temperatures for batch processing
    messages_list = []
    temperatures = []
    
    for agent, prompt in zip(agents, prompts):
        base_temp = agent.get('temperature', 0.7)
        temp = base_temp * (1 + 0.1 * debate_round)
        temperatures.append(temp)
        
        messages = [
            {"role": "system", "content": agent['instructions']},
            {"role": "user", "content": prompt}
        ]
        messages_list.append(messages)
    
    # Use batch generation
    responses = llama_generate_true_batch(messages_list, temperatures, max_tokens=600)
    return [response.strip() for response in responses]

def generate_analysis(agent, prompt, debate_round=0):
    """Single agent analysis - kept for compatibility"""
    base_temp = agent.get('temperature', 0.7)
    temp = base_temp * (1 + 0.1 * debate_round)
    messages = [
        {"role": "system", "content": agent['instructions']},
        {"role": "user", "content": prompt}
    ]
    response = llama_generate(messages, temperature=temp, max_tokens=600)
    return response.strip()

def parse_json_response(response):
    try:
        return json.loads(response)
    except Exception:
        return {"analysis": response, "decision": None, "influenced_by": []}

def train_agent_weights(agents, training_data):
    print("Training agent weights")
    for agent in agents:
        agent['accuracy'] = 0.0
        agent['correct_count'] = 0
        agent['total_count'] = 0
    
    # Process training data in batches for efficiency
    batch_size = len(agents)  # Process all agents per question simultaneously
    
    for item in tqdm(training_data, desc="Training agents"):
        question = item['question']
        correct_answer = str(item['answerKey'])
        options = item.get('options', [])
        
        # Prepare prompts for all agents
        prompts = []
        for agent in agents:
            prompt = (
                f"""You are a {agent['role']}. Answer this question using your specialized expertise.

                Question: {question}

                Instructions: 
                - Apply your {agent['role']}'s unique methodology and analytical framework
                - Use the specific reasoning approaches defined in your role instructions
                - Provide your final answer as either 0: {options[0]}, 1: {options[1]}, 2: {options[2]}, 3: {options[3]}
                - Follow your role's JSON response format

                {agent['instructions']}

                Response:"""
            )
            prompts.append(prompt)
        
        # Generate responses in batch
        responses = generate_analysis_batch(agents, prompts)
        
        # Process responses
        for agent, response in zip(agents, responses):
            print(f"{agent['role']}: {response}")
            parsed = parse_json_response(response)
            decision = parsed.get("decision", "")
            print(f"Decision: {decision}, Correct: {correct_answer}")
            
            agent['total_count'] += 1
            if decision == correct_answer:
                agent['correct_count'] += 1
    
    # Calculate accuracies and weights
    total_accuracy = 0
    for agent in agents:
        if agent['total_count'] > 0:
            agent['accuracy'] = agent['correct_count'] / agent['total_count']
        else:
            agent['accuracy'] = 0
        agent['weight'] = max(0.1, agent['accuracy'])
        total_accuracy += agent['accuracy']
    
    if total_accuracy > 0:
        weight_sum = sum(agent['weight'] for agent in agents)
        for agent in agents:
            agent['weight'] = agent['weight'] / weight_sum
    else:
        for agent in agents:
            agent['weight'] = 1.0 / len(agents)
    
    print("Agent Training Results:")
    for agent in agents:
        print(f"{agent['role']}: Accuracy = {agent['accuracy']:.4f}, Weight = {agent['weight']:.4f}")
    
    return agents

def track_influence(agents):
    influence_counts = {agent['id']: 0 for agent in agents}
    for agent in agents:
        if len(agent['analysis']) < 2:
            continue
        try:
            parsed = parse_json_response(agent['analysis'][-1])
            influenced_by = parsed.get("influenced_by", "")
            if isinstance(influenced_by, str):
                for other_agent in agents:
                    if other_agent['role'].lower() in influenced_by.lower():
                        influence_counts[other_agent['id']] += 1
            elif isinstance(influenced_by, list):
                for influencer in influenced_by:
                    for other_agent in agents:
                        if other_agent['role'].lower() in influencer.lower():
                            influence_counts[other_agent['id']] += 1
        except Exception:
            continue
    
    total_influences = sum(influence_counts.values()) or 1
    for agent in agents:
        agent['influence_score'] = influence_counts[agent['id']] / total_influences
    return influence_counts

def create_debate_graph(agents, question, gold_answer=None, decision=None, is_correct=None):
    G = nx.DiGraph()
    G.add_node("question", content=question, type="question", round=-1)
    if gold_answer is not None:
        G.add_node("ground_truth", content=gold_answer, type="ground_truth")
        G.add_edge("question", "ground_truth", type="has_ground_truth")
    
    max_rounds = max(len(a['analysis']) for a in agents)
    for agent in agents:
        role = agent['role']
        for round_num, response in enumerate(agent['analysis']):
            node_id = f"{role}_{round_num}"
            parsed = parse_json_response(response)
            decision = (parsed.get("decision") or "").strip().lower()
            is_correct = (decision == gold_answer.strip().lower()) if gold_answer else None
            
            G.add_node(
                node_id,
                content=response,
                type="initial_response" if round_num == 0 else "response",
                round=round_num,
                role=role,
                decision=decision,
                is_correct=is_correct,
                gold_answer=gold_answer
            )
            
            if round_num == 0:
                G.add_edge("question", node_id, type="responds_to")
            else:
                prev_node = f"{role}_{round_num - 1}"
                G.add_edge(prev_node, node_id, type="continues")
            
            influencers = parsed.get("influenced_by", [])
            if isinstance(influencers, str):
                influencers = [influencers]
            for infl in influencers:
                for other in agents:
                    if other['role'].lower() in infl.lower():
                        other_prev = f"{other['role']}_{round_num - 1}"
                        if other_prev in G:
                            G.add_edge(
                                other_prev,
                                node_id,
                                type="influences",
                                weight=other["weight"]
                            )
    
    edges_to_add = []
    for src, dst, data in G.edges(data=True):
        if data.get('type') in ['continues', 'influences']:
            edges_to_add.append((dst, src, data))
    for dst, src, data in edges_to_add:
        G.add_edge(dst, src, **data)
    
    return G

def has_consensus(agents):
    decisions = []
    for ag in agents:
        raw = ag['analysis'][-1]
        try:
            parsed = parse_json_response(raw)
            decision = parsed.get("decision", "")
        except Exception:
            decision = ""
        if not decision:
            m = re.search(r'"decision"\s*:\s*"([^"]+)"', raw)
            decision = m.group(1) if m else ""
        norm = decision.strip().lower()
        if not norm:
            return False, None
        decisions.append(norm)
    
    counts = Counter(decisions)
    if len(counts) == 1:
        return True, decisions[0]
    return False, None

def layered_consensus_process(question, agents, base_options, gold_answer):
    """
    Optimized consensus process with batch LLM inference
    """
    discussion_history = []
    discussion_history.append(f"GOLD_ANSWER: {gold_answer}")
    
    print("=== INITIAL ROUND (BATCHED) ===")
    for agent in agents:
        agent['analysis'] = []
    
    # Prepare initial prompts for all agents
    initial_prompts = []
    for agent in agents:
        prompt = (
            f"""You are a {agent['role']}. Answer this question using your specialized expertise.

            Question: {question}

            Instructions: 
            - Apply your {agent['role']}'s unique methodology and analytical framework
            - Use the specific reasoning approaches defined in your role instructions
            - Provide your final answer as either True or False
            - Follow your role's JSON response format

            {agent['instructions']}

            Response:"""
        )
        initial_prompts.append(prompt)
    
    # Generate initial responses in batch
    initial_responses = generate_analysis_batch(agents, initial_prompts)
    
    # Store responses and log
    for agent, response in zip(agents, initial_responses):
        agent['analysis'] = [response]
        discussion_history.append(f"INITIAL - {agent['role']}:\n{response}")
        print(f"[{agent['role']} INITIAL]\n{response}")
    
    # Check for consensus after initial round
    consensus, decision = has_consensus(agents)
    if consensus:
        print(f"Consensus reached on '{decision}' in initial round")
        print(gold_answer)
        is_correct = (decision == gold_answer.strip().lower()) if gold_answer else None
        print(f"Is correct: {is_correct}")
        discussion_history.append(f"CONSENSUS_CORRECT: {is_correct}")
        return discussion_history, create_debate_graph(agents, question, gold_answer, decision, is_correct)
    
    # Debate rounds with batch processing
    for round_num in range(3):
        print(f"=== DEBATE ROUND {round_num + 1} (BATCHED) ===")
        
        # Prepare debate prompts for all agents
        debate_prompts = []
        for i, agent in enumerate(agents):
            others = "\n".join([
                f"{other['role']} (Weight: {other['weight']:.3f}): {parse_json_response(other['analysis'][-1]).get('analysis', 'No analysis available')}"
                for j, other in enumerate(agents) if j != i
            ])
            
            prompt = (
            f"""You are a {agent['role']}. Answer this question using your specialized expertise, and considering the other analyses {others}.

            Question: {question}

            Instructions: 
            - Apply your {agent['role']}'s unique methodology and analytical framework
            - Use the specific reasoning approaches defined in your role instructions
            - Provide your final answer as either True or False
            - Follow your role's JSON response format

            {agent['instructions']}

            Response:"""
        )


            debate_prompts.append(prompt)
        
        # Generate debate responses in batch
        debate_responses = generate_analysis_batch(agents, debate_prompts, debate_round=round_num + 1)
        
        # Store responses and log
        for agent, response in zip(agents, debate_responses):
            agent['analysis'].append(response)
            discussion_history.append(f"ROUND {round_num + 1} - {agent['role']}:\n{response}")
            print(f"[{agent['role']} REFINED]\n{response}")
        
        # Track influence and check consensus
        influence_counts = track_influence(agents)
        print(f"Influence counts after round {round_num + 1}: {influence_counts}")
        
        consensus, decision = has_consensus(agents)
        if consensus:
            print(f"Consensus reached on '{decision}' at round {round_num + 1}")
            decision = str(decision).lower()
            print(decision)
            gold_answer = str(gold_answer).lower()
            print(gold_answer)
            is_correct = (decision == gold_answer) if gold_answer else None
            print(f"Is correct: {is_correct}")
            discussion_history.append(f"CONSENSUS_CORRECT: {is_correct}")
            return discussion_history, create_debate_graph(agents, question, gold_answer, decision, is_correct)
    
    # Final weighted vote if no consensus
    vote_totals = defaultdict(float)
    for ag in agents:
        parsed = parse_json_response(ag['analysis'][-1])
        dec = parsed.get("decision")
        if dec:
            vote_totals[dec.strip().lower()] += ag['weight']
    
    if vote_totals:
        final_decision, total = max(vote_totals.items(), key=lambda x: x[1])
        print(f"Weighted vote selects '{final_decision}' ({total:.2f} total weight)")
        discussion_history.append(f"WEIGHTED_VOTE: {final_decision}")
        is_correct = (final_decision == gold_answer.strip().lower()) if gold_answer else None
        discussion_history.append(f"WEIGHTED_VOTE_CORRECT: {is_correct}")
    else:
        final_decision = None
        is_correct = None
    
    return discussion_history, create_debate_graph(agents, question, gold_answer, final_decision, is_correct)

def extract_node_embeddings(graph, model_name="sentence-transformers/all-mpnet-base-v2"):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        embeddings = {}
        for node_id in graph.nodes():
            content = graph.nodes[node_id].get('content', '')
            if isinstance(content, str) and content:
                embeddings[node_id] = model.encode(content)
            else:
                embeddings[node_id] = np.zeros(model.get_sentence_embedding_dimension())
        return embeddings
    except ImportError:
        print("Warning: sentence-transformers not installed. Using random embeddings.")
        dim = 768
        embeddings = {}
        for node_id in graph.nodes():
            embeddings[node_id] = np.random.normal(0, 0.1, dim)
        return embeddings

def convert_to_pyg_data(graph, embeddings):
    node_list = list(graph.nodes())
    node_list = [n for n in node_list if graph.nodes[n].get('type') != 'ground_truth']
    
    x = []
    for node in node_list:
        emb = embeddings[node] if isinstance(embeddings, dict) else embeddings[node_list.index(node)]
        x.append(emb)
    x = torch.tensor(np.stack(x), dtype=torch.float)
    
    edge_index = []
    edge_attr = []
    for src, dst, data in graph.edges(data=True):
        if src in node_list and dst in node_list:
            edge_type = data.get('type', '')
            weight = data.get('weight', 0.0) if edge_type == 'influences' else 0.0
            edge_index.append([node_list.index(src), node_list.index(dst)])
            edge_attr.append([weight])
            edge_index.append([node_list.index(dst), node_list.index(src)])
            edge_attr.append([weight])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float) if edge_attr else None
    
    y = []
    for node in node_list:
        node_data = graph.nodes[node]
        y.append(1 if node_data.get('is_correct', False) else 0)
    y = torch.tensor(y, dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

def create_mag_dataset(training_data, agents, batch_size=1):
    """
    Create MAG dataset with batch-optimized LLM inference
    """
    mag_dataset = []
    
    # Process questions with batch-optimized agent responses
    for i in tqdm(range(0, len(training_data), batch_size), desc="Creating MAG dataset (batch-optimized)"):
        batch = training_data[i:i+batch_size]
        for item in batch:
            question = item['question']
            options = item.get('options', [])
            gold_answer = str(item['gold_answer'])
            
            # The layered_consensus_process now uses batch inference internally
            _, debate_graph = layered_consensus_process(question, agents, options, gold_answer)
            embeddings = extract_node_embeddings(debate_graph)
            pyg_data = convert_to_pyg_data(debate_graph, embeddings)
            
            mag_dataset.append({
                "question": question,
                "options": options,
                "graph": debate_graph,
                "pyg_data": pyg_data
            })
    
    return mag_dataset

def main():
    torch.cuda.empty_cache()
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    
    # Configuration
    SEED = 42
    MODEL_SIZE = "small"
    OUTPUT_DIR = "outputs"
    GCN_IN_CHANNELS = 768
    GCN_HIDDEN_CHANNELS = 256
    GCN_OUT_CHANNELS = 4
    ALPHA = 1.0
    BETA = 1.0
    GAMMA = 0.1
    DELTA = 0.5
    NUM_EPOCHS = 5
    BATCH_SIZE = 2
    LEARNING_RATE = 1e-6
    
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize agents
    agents = [
        {
            "id": i,
            "role": role,
            "instructions": role_specifications[role]["instructions"],
            "analysis": [],
            "previous_positions": [],
            "base_temp": 0.8 + random.uniform(-0.7, 0.7),
            "weight": 1.0,
            "accuracy": 0.0,
            "influence_score": 0.0
        } for i, role in enumerate(role_specifications)
    ]
    
    # Load dataset
    full_train = load_dataset("wics/strategy-qa", split="test")
    subsplits = full_train.train_test_split(test_size=.2, seed=42)
    mag_creation_data = subsplits["test"]
    
    # Set pre-trained agent weights (from your previous training)
    for agent in agents:
        if agent["role"] == "Scientist":
            agent["weight"] = 0.1903
        elif agent["role"] == "Lawyer":
            agent["weight"] = 0.1957
        elif agent["role"] == "Historian":
            agent["weight"] = 0.2208
        elif agent["role"] == "Mathematician":
            agent["weight"] = 0.1957
        else:
            agent["weight"] = 0.1975
    
    # Prepare training data
    training_data = [
        {"question": item["question"],
         "gold_answer": item["answer"],
         "options": ["True", "False"]}
        for item in mag_creation_data
    ]
    mag_dataset = create_mag_dataset(training_data, agents)
    
    print("Creating MAG dataset with batch processing...")
    
    # Save dataset
    os.makedirs("data", exist_ok=True)
    with open("data/mag_dataset.pkl", "wb") as f:
        pickle.dump(mag_dataset, f)
    
    print(f"MAG dataset created with {len(mag_dataset)} examples")
    print("Dataset saved to data/mag_dataset.pkl")

class StudentModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(StudentModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x): 
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

class TeacherModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TeacherModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

def train_with_all_losses(
    teacher, student, train_loader, epochs, learning_rate,
    T, ce_loss_weight, kl_loss_weight, cosine_loss_weight, device
):
    ce_loss = nn.CrossEntropyLoss()
    cosine_similarity = nn.CosineSimilarity(dim=-1)
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.eval()
    student.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits = teacher(inputs)

            student_logits = student(inputs)

            soft_targets = F.softmax(teacher_logits / T, dim=-1)
            student_log_probs = F.log_softmax(student_logits / T, dim=-1)
            kl_loss = F.kl_div(student_log_probs, soft_targets, reduction="batchmean") * (T**2)

            ce = ce_loss(student_logits, labels)

            cosine_sim = cosine_similarity(student_logits, teacher_logits)
            cosine_loss = 1 - cosine_sim.mean()

            loss = (
                ce_loss_weight * ce +
                kl_loss_weight * kl_loss +
                cosine_loss_weight * cosine_loss
            )

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

input_dim = 768
output_dim = 4
batch_size = 32
num_samples = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train = torch.randn(num_samples, input_dim)
y_train = torch.randint(0, output_dim, (num_samples,))
X_test = torch.randn(int(num_samples * 0.2), input_dim)
y_test = torch.randint(0, output_dim, (int(num_samples * 0.2),))

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

nn_deep = TeacherModel(input_dim, output_dim).to(device)
new_nn_light = StudentModel(input_dim, output_dim).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(new_nn_light.parameters(), lr=0.001)
new_nn_light.train()
for epoch in range(5):
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = new_nn_light(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

test_accuracy_light_ce = test(new_nn_light, test_loader, device)

train_with_all_losses(
    teacher=nn_deep,
    student=new_nn_light,
    train_loader=train_loader,
    epochs=10,
    learning_rate=0.001,
    T=2,
    ce_loss_weight=0.5,
    kl_loss_weight=0.4,
    cosine_loss_weight=0.1,
    device=device
)

test_accuracy_deep = test(nn_deep, test_loader, device)
test_accuracy_light_all = test(new_nn_light, test_loader, device)

print(f"Teacher accuracy: {test_accuracy_deep:.2f}%")
print(f"Student accuracy (CE only): {test_accuracy_light_ce:.2f}%")
print(f"Student accuracy (CE + KL + Cosine): {test_accuracy_light_all:.2f}%")

if __name__ == "__main__":
    main()
