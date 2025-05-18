import os
import json
import torch
import logging
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
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
import openai

from smodel import SocraticMAGDi, SocraticMAGDiDataCollator
from datasets import load_dataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Agent role specifications
role_specifications = {
    "Scientist": {
        "instructions": """
You are a Scientist.
- For any decision-making scenario, select exactly one of the provided options as your decision.
- Justify your decision using rigorous scientific reasoning, evidence, and a global perspective.
- Consider both immediate and long-term impacts on humanity, scientific progress, public health, and the environment.
- Reflect on the value of knowledge, innovation, and the risks of losing unique expertise or setting scientific precedent.
- Weigh risks, probabilities, and cascading effects for society and the world.
- Respond strictly in JSON with two fields:
{
  "decision": "<selected_option>",
  "analysis": "<detailed scientific reasoning, global implications, trade-offs, and confidence.>"
}
If you do not follow this format exactly, your response will be disregarded for consensus.
"""
    },
    "Lawyer": {
        "instructions": """
You are a Lawyer.
- For any decision-making scenario, select exactly one of the provided options as your decision.
- Justify your decision using legal frameworks, compliance, and precedent at both local and international levels.
- Consider legal and ethical responsibilities, justice, rights, and the broader implications for governance and the rule of law.
- Reflect on the importance of fairness, equal treatment, and the consequences for future legal systems and social contracts.
- Discuss conflicts between different legal systems, humanitarian law, and moral obligations.
- Respond strictly in JSON with two fields:
{
  "decision": "<selected_option>",
  "analysis": "<detailed legal reasoning, societal implications, trade-offs, and confidence.>"
}
If you do not follow this format exactly, your response will be disregarded for consensus.
"""
    },
    "Historian": {
        "instructions": """
You are a Historian.
- For any decision-making scenario, select exactly one of the provided options as your decision.
- Justify your decision using historical precedent, context, and lessons from past events.
- Consider how similar dilemmas have shaped societies, and the long-term impact of such decisions on culture, innovation, and social cohesion.
- Reflect on the legacy and narrative your decision would create for future generations.
- Explicitly weigh risks and benefits for both the immediate group and the world at large.
- Consider how similar situations in the past have happened
- Respond strictly in JSON with two fields:
{
  "decision": "<selected_option>",
  "analysis": "<detailed historical reasoning, global implications, trade-offs, and confidence.>"
}
If you do not follow this format exactly, your response will be disregarded for consensus.
"""
    },
    "Mathematician": {
        "instructions": """
You are a Mathematician.
- For any decision-making scenario, select exactly one of the provided options as your decision.
- Justify your decision using quantitative reasoning, probability, risk analysis, and modeling.
- Consider both immediate outcomes and expected value for society or humanity, factoring in potential for future discoveries, aid, or loss.
- Explicitly model outcomes, discuss trade-offs between maximizing survival, minimizing risk, and optimizing for long-term benefit.
- Respond strictly in JSON with two fields:
{
  "decision": "<selected_option>",
  "analysis": "<detailed quantitative reasoning, global implications, trade-offs, and confidence.>"
}
If you do not follow this format exactly, your response will be disregarded for consensus.
"""
    },
    "Economist": {
        "instructions": """
You are an Economist.
- For any decision-making scenario, select exactly one of the provided options as your decision.
- Justify your decision using economic impact, cost-benefit analysis, and externalities at both local and global scales.
- Consider the economic consequences of each option, including innovation, productivity, costs, high taxes, and societal stability.
- Reflect on the value of human capital, incentives, and broader economic ripple effects.
- Explicitly discuss trade-offs, uncertainties, and long-term implications for economic growth and well-being.
- Respond strictly in JSON with two fields:
{
  "decision": "<selected_option>",
  "analysis": "<detailed economic reasoning, global implications, trade-offs, and confidence.>"
}
If you do not follow this format exactly, your response will be disregarded for consensus.
"""
    },
    "Ethicist": {
        "instructions": """
You are an Ethicist.
- For any decision-making scenario, select exactly one of the provided options as your decision.
- Justify your decision using ethical frameworks (utilitarianism, deontology, virtue ethics, etc.) and principles, considering both local and global consequences.
- Reflect on the value of each life, fairness, and the potential for harm or benefit to humanity.
- Explicitly discuss moral dilemmas, conflicts between duties, and trade-offs between individual and collective good.
- Respond strictly in JSON with two fields:
{
  "decision": "<selected_option>",
  "analysis": "<detailed ethical reasoning, global implications, trade-offs, and confidence.>"
}
If you do not follow this format exactly, your response will be disregarded for consensus.
"""
    }
}

class MAGDiDataset(Dataset):
    """Dataset for training the SocraticMAGDi model."""
    
    def __init__(self, decomposer_examples, solver_examples, pos_examples, neg_examples, graphs):
        assert len(decomposer_examples) == len(solver_examples) == len(pos_examples) == len(neg_examples) == len(graphs)
        self.decomposer_examples = decomposer_examples
        self.solver_examples = solver_examples
        self.pos_examples = pos_examples
        self.neg_examples = neg_examples
        self.graphs = graphs
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return {
            "decomposer_input_ids": self.decomposer_examples[idx]["prompt_input_ids"],
            "decomposer_attention_mask": self.decomposer_examples[idx]["prompt_attention_mask"],
            "decomposer_labels": self.decomposer_examples[idx]["completion_input_ids"],
            "solver_input_ids": self.solver_examples[idx]["prompt_input_ids"],
            "solver_attention_mask": self.solver_examples[idx]["prompt_attention_mask"],
            "solver_labels": self.solver_examples[idx]["completion_input_ids"],
            "pos_input_ids": self.pos_examples[idx]["input_ids"],
            "pos_attention_mask": self.pos_examples[idx]["attention_mask"],
            "pos_labels": self.pos_examples[idx]["labels"],
            "neg_input_ids": self.neg_examples[idx]["input_ids"],
            "neg_attention_mask": self.neg_examples[idx]["attention_mask"],
            "neg_labels": self.neg_examples[idx]["labels"],
            "graph": self.graphs[idx]
        }

class SocraticMAGDiTrainer(Trainer):
    """Custom trainer for the SocraticMAGDi model."""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the combined loss for the SocraticMAGDi model.
        """
        outputs = model(
            decomposer_input_ids=inputs["decomposer_input_ids"],
            decomposer_attention_mask=inputs["decomposer_attention_mask"],
            decomposer_labels=inputs["decomposer_labels"],
            solver_input_ids=inputs["solver_input_ids"],
            solver_attention_mask=inputs["solver_attention_mask"],
            solver_labels=inputs["solver_labels"],
            pos_input_ids=inputs["pos_input_ids"],
            pos_attention_mask=inputs["pos_attention_mask"],
            pos_labels=inputs["pos_labels"],
            neg_input_ids=inputs["neg_input_ids"],
            neg_attention_mask=inputs["neg_attention_mask"],
            neg_labels=inputs["neg_labels"],
            graph=inputs["graph"]
        )
        
        lm_loss, node_loss, mr_loss, alignment_loss = outputs
        total_loss = lm_loss + node_loss + mr_loss + alignment_loss
        
        if return_outputs:
            return total_loss, outputs
        
        return total_loss

def generate_analysis(agent, prompt, client, debate_round=0):
    """Generate analysis from an agent using OpenAI API"""
    # Dynamic temperature scheduling
    temp = agent['base_temp'] * (1 + 0.1 * debate_round)  # Increase temp in later rounds
    temp = min(max(temp, 0.5), 1.5)  # Clamp between 0.5-1.5

    messages = [
        {"role": "system", "content": agent['instructions']},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=temp,
        max_tokens=400
    )
    return response.choices[0].message.content.strip()

def parse_json_response(response):
    """Parse JSON response, handling errors gracefully"""
    try:
        return json.loads(response)
    except Exception:
        return {"analysis": response, "decision": None, "influenced_by": []}

def train_agent_weights(agents, training_data, client):
    """Train agent weights based on performance on training data"""
    logger.info("Training agent weights")
    
    # Reset metrics
    for agent in agents:
        agent['accuracy'] = 0.0
        agent['correct_count'] = 0
        agent['total_count'] = 0
    
    # Evaluate each agent on training data
    for item in tqdm(training_data, desc="Training agents"):
        question = item['question']
        correct_answer = item['answerKey'].strip().lower()
        options = item.get('options', [])
        
        base_options = ""
        if options:
            base_options = f"\nOptions: {', '.join(options)}"
        
        for agent in agents:
            prompt = (
                f"{question}{base_options}\n\n"
                "Read the question thoroughly, understand the entire context, make 0 assumptions. "
                "You must respond to the question aptly, selecting exactly one of the provided options. "
                """Respond strictly in JSON format with your decision and analysis.
                {
                  "decision": "<selected_option>",
                  "analysis": "<reasoning>"
                }"""
            )
            
            response = generate_analysis(agent, prompt, client)
            parsed = parse_json_response(response)
            decision = parsed.get("decision", "").strip().lower()
            
            # Update accuracy metrics
            agent['total_count'] += 1
            if decision == correct_answer:
                agent['correct_count'] += 1
    
    # Calculate final accuracy and weights
    total_accuracy = 0
    for agent in agents:
        if agent['total_count'] > 0:
            agent['accuracy'] = agent['correct_count'] / agent['total_count']
        else:
            agent['accuracy'] = 0
        
        # Ensure minimum weight of 0.1 to prevent complete exclusion
        agent['weight'] = max(0.1, agent['accuracy'])
        total_accuracy += agent['accuracy']
    
    # Normalize weights if we have any accuracy
    if total_accuracy > 0:
        weight_sum = sum(agent['weight'] for agent in agents)
        for agent in agents:
            agent['weight'] = agent['weight'] / weight_sum
    else:
        # Equal weights if no accuracy data
        for agent in agents:
            agent['weight'] = 1.0 / len(agents)
    
    # Print results
    logger.info("Agent Training Results:")
    for agent in agents:
        logger.info(f"{agent['role']}: Accuracy = {agent['accuracy']:.4f}, Weight = {agent['weight']:.4f}")
    
    return agents

def track_influence(agents):
    """Track which agents influenced others based on the 'influenced_by' field"""
    influence_counts = {agent['id']: 0 for agent in agents}
    
    for agent in agents:
        if len(agent['analysis']) < 2:
            continue
        
        try:
            parsed = parse_json_response(agent['analysis'][-1])
            influenced_by = parsed.get("influenced_by", "")
            
            # Parse influenced_by field
            if isinstance(influenced_by, str):
                # Look for role names in the string
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
    
    # Update influence scores
    total_influences = sum(influence_counts.values()) or 1  # Avoid division by zero
    for agent in agents:
        agent['influence_score'] = influence_counts[agent['id']] / total_influences
    
    return influence_counts

def create_debate_graph(agents, question):
    """Create a graph representation of the debate for MAG creation"""
    G = nx.DiGraph()
    
    # Add question node
    G.add_node("question", content=question, type="question", round=-1)
    
    # Add initial response nodes and connect to question
    for agent in agents:
        if not agent['analysis']:
            continue
            
        initial_node_id = f"{agent['role']}_0"
        G.add_node(
            initial_node_id,
            content=agent['analysis'][0],
            type="initial_response",
            role=agent['role'],
            round=0
        )
        G.add_edge("question", initial_node_id, type="responds_to")
    
    # Add subsequent rounds and track influences
    for round_num in range(1, len(agents[0]['analysis']) if agents[0]['analysis'] else 0):
        for agent in agents:
            if len(agent['analysis']) <= round_num:
                continue
                
            node_id = f"{agent['role']}_{round_num}"
            prev_node_id = f"{agent['role']}_{round_num-1}"
            
            G.add_node(
                node_id,
                content=agent['analysis'][round_num],
                type="response",
                role=agent['role'],
                round=round_num
            )
            
            # Connect to previous response from same agent
            if prev_node_id in G:
                G.add_edge(prev_node_id, node_id, type="continues")
            
            # Add influence edges based on the "influenced_by" field
            try:
                parsed = parse_json_response(agent['analysis'][round_num])
                influenced_by = parsed.get("influenced_by", "")
                
                if isinstance(influenced_by, str):
                    # Look for role names in the string
                    for other_agent in agents:
                        if other_agent['role'].lower() in influenced_by.lower():
                            other_prev_node = f"{other_agent['role']}_{round_num-1}"
                            if other_prev_node in G:
                                G.add_edge(other_prev_node, node_id, type="influences")
                elif isinstance(influenced_by, list):
                    for influencer in influenced_by:
                        for other_agent in agents:
                            if other_agent['role'].lower() in influencer.lower():
                                other_prev_node = f"{other_agent['role']}_{round_num-1}"
                                if other_prev_node in G:
                                    G.add_edge(other_prev_node, node_id, type="influences")
            except Exception:
                continue
    
    return G

def layered_consensus_process(question, agents, client, base_options="", max_debate_rounds=2):
    """Run the multi-agent debate process with weighted consensus"""
    discussion_history = []
    logger.info("=== INITIAL ROUND ===")
    
    # Reset agent analysis for new question
    for agent in agents:
        agent['analysis'] = []
    
    # Initial responses
    for agent in agents:
        prompt = (
            f"{question}{base_options}\n\n"
            "Read the question thoroughly, understand the entire context, make 0 assumptions. "
            "You must respond to the question aptly, selecting exactly one of the provided options. "
            """Respond strictly in JSON format with your decision and analysis.
            {
              "decision": "<selected_option>",
              "analysis": "<reasoning>"
            }"""
        )
        
        response = generate_analysis(agent, prompt, client)
        agent['analysis'] = [response]
        discussion_history.append(f"INITIAL - {agent['role']}:\n{response}")
        logger.info(f"[{agent['role']} INITIAL]\n{response}")
    
    # Check initial consensus
    consensus = False
    decision = None
    
    # Debate rounds with temperature escalation
    for round_num in range(max_debate_rounds):
        logger.info(f"=== DEBATE ROUND {round_num+1} ===")
        for i, agent in enumerate(agents):
            others = "\n".join([
                f"{other['role']}: {parse_json_response(other['analysis'][-1])['analysis']}"
                for j, other in enumerate(agents) if j != i
            ])
            
            prompt = (
                f"Peer Analyses:\n{others}\n\n"
                "Reconsider the initial question and your peers' decisions to come to a decision. "
                "Don't simply go with the majority as they may be trying to trick you! "
                "You may revise your decision considering these perspectives. "
                "Explain why your chosen option is more valid (make sure it is factually correct). "
                "Maintain your professional viewpoint while acknowledging valid arguments. "
                f"Only include the given options in your decision{base_options}! "
                """Respond strictly in JSON Format!!!
                {
                  "decision": "<updated_choice>",
                  "analysis": "<revised_reasoning>",
                  "influenced_by": "<key influencing agents (as few as possible, only those that really helped you)>"
                }"""
            )
            
            response = generate_analysis(agent, prompt, client, debate_round=round_num+1)
            agent['analysis'].append(response)
            discussion_history.append(f"ROUND {round_num+1} - {agent['role']}:\n{response}")
            logger.info(f"[{agent['role']} REFINED]\n{response}")
        
        # Track influence after each round
        influence_counts = track_influence(agents)
        logger.info(f"Influence counts after round {round_num+1}: {influence_counts}")
    
    return discussion_history, create_debate_graph(agents, question)

def extract_node_embeddings(graph, model_name="sentence-transformers/all-mpnet-base-v2"):
    """Extract embeddings for graph nodes using a pretrained model"""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        
        embeddings = {}
        for node_id in graph.nodes():
            content = graph.nodes[node_id].get('content', '')
            if isinstance(content, str) and content:
                embeddings[node_id] = model.encode(content)
            else:
                # Default embedding for nodes without content
                embeddings[node_id] = np.zeros(model.get_sentence_embedding_dimension())
        
        return embeddings
    except ImportError:
        logger.warning("sentence-transformers not installed. Using random embeddings.")
        # Fallback to random embeddings
        dim = 768  # Default embedding dimension
        embeddings = {}
        for node_id in graph.nodes():
            embeddings[node_id] = np.random.normal(0, 0.1, dim)
        return embeddings

def convert_to_pyg_data(graph, embeddings):
    """Convert networkx graph to PyTorch Geometric Data object"""
    # Map node IDs to indices
    node_map = {node: i for i, node in enumerate(graph.nodes())}
    
    # Prepare node features
    x = []
    for node in graph.nodes():
        x.append(embeddings[node])
    x = torch.tensor(np.array(x), dtype=torch.float)
    
    # Prepare edge indices
    edge_index = []
    for u, v in graph.edges():
        edge_index.append([node_map[u], node_map[v]])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Assign node labels (0: question, 1: incorrect, 2: partially correct, 3: correct)
    y = []
    for node in graph.nodes():
        node_data = graph.nodes[node]
        node_type = node_data.get('type', '')
        
        if node_type == 'question':
            y.append(0)
        else:
            # For response nodes, use a simple heuristic based on round number
            round_num = node_data.get('round', 0)
            if round_num == 0:
                y.append(1)  # Initial responses are considered less refined
            elif round_num == 1:
                y.append(2)  # Middle round responses are partially correct
            else:
                y.append(3)  # Final round responses are considered most correct
    
    y = torch.tensor(y, dtype=torch.long)
    
    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, y=y)
    return data

def create_mag_dataset(questions, agents, client, options=None):
    """Create a MAG dataset from multiple questions"""
    mag_dataset = []
    
    for i, question in enumerate(questions):
        logger.info(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
        
        base_options = ""
        if options and i < len(options):
            base_options = f"\nOptions: {', '.join(options[i])}"
        
        _, debate_graph = layered_consensus_process(question, agents, client, base_options)
        
        # Extract embeddings
        embeddings = extract_node_embeddings(debate_graph)
        
        # Convert to PyG Data
        pyg_data = convert_to_pyg_data(debate_graph, embeddings)
        
        # Add to dataset
        mag_dataset.append({
            "question": question,
            "options": options[i] if options and i < len(options) else [],
            "graph": debate_graph,
            "pyg_data": pyg_data
        })
    
    return mag_dataset

def prepare_socratic_examples(mag_dataset, tokenizer):
    """Prepare examples for training the Socratic model components"""
    decomposer_examples = []
    solver_examples = []
    
    for item in mag_dataset:
        question = item["question"]
        graph = item["graph"]
        
        # Extract sub-questions (nodes with highest influence)
        sub_questions = []
        for node, data in graph.nodes(data=True):
            if data.get('type') == 'response' and graph.in_degree(node) > 1:
                # Nodes with multiple incoming edges are influential
                sub_questions.append(data.get('content', ''))
        
        if sub_questions:
            # Create decomposer example
            decomposer_prompt = f"Question: {question}\nBreak this down into sub-questions:"
            decomposer_completion = "\n".join([f"- {sq[:100]}..." for sq in sub_questions[:3]])
            
            decomposer_examples.append({
                "prompt": decomposer_prompt,
                "completion": decomposer_completion
            })
        
        # Find sub-question answers (solver examples)
        for node, data in graph.nodes(data=True):
            if data.get('type') == 'response' and data.get('round', 0) > 0:
                # Find the nodes that influenced this response
                influencers = []
                for pred in graph.predecessors(node):
                    if graph.nodes[pred].get('type') == 'response':
                        influencers.append(pred)
                
                if influencers:
                    # Create solver example
                    influencer_content = graph.nodes[influencers[0]].get('content', '')
                    solver_prompt = f"Question: {influencer_content[:100]}..."
                    solver_completion = data.get('content', '')
                    
                    solver_examples.append({
                        "prompt": solver_prompt,
                        "completion": solver_completion
                    })
    
    # Tokenize examples
    tokenized_decomposer = []
    for ex in decomposer_examples:
        prompt_tokens = tokenizer(
            ex["prompt"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        
        completion_tokens = tokenizer(
            ex["completion"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        
        tokenized_decomposer.append({
            "prompt_input_ids": prompt_tokens.input_ids[0],
            "prompt_attention_mask": prompt_tokens.attention_mask[0],
            "completion_input_ids": completion_tokens.input_ids[0],
            "completion_attention_mask": completion_tokens.attention_mask[0]
        })
    
    tokenized_solver = []
    for ex in solver_examples:
        prompt_tokens = tokenizer(
            ex["prompt"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        
        completion_tokens = tokenizer(
            ex["completion"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        
        tokenized_solver.append({
            "prompt_input_ids": prompt_tokens.input_ids[0],
            "prompt_attention_mask": prompt_tokens.attention_mask[0],
            "completion_input_ids": completion_tokens.input_ids[0],
            "completion_attention_mask": completion_tokens.attention_mask[0]
        })
    
    return tokenized_decomposer, tokenized_solver

def prepare_pos_neg_examples(mag_dataset, tokenizer):
    """Prepare positive and negative examples for contrastive learning"""
    pos_examples = []
    neg_examples = []
    
    for item in mag_dataset:
        question = item["question"]
        graph = item["graph"]
        
        # Find nodes with high and low influence
        node_influence = {}
        for node in graph.nodes():
            # Measure influence by out-degree (how many nodes it influences)
            node_influence[node] = graph.out_degree(node)
        
        # Sort nodes by influence
        sorted_nodes = sorted(node_influence.items(), key=lambda x: x[1], reverse=True)
        
        # Get positive examples (high influence) and negative examples (low influence)
        pos_nodes = [node for node, _ in sorted_nodes[:len(sorted_nodes)//3]]
        neg_nodes = [node for node, _ in sorted_nodes[-len(sorted_nodes)//3:]]
        
        # Create examples
        for pos_node in pos_nodes:
            if 'content' in graph.nodes[pos_node]:
                pos_prompt = f"Question: {question}"
                pos_completion = graph.nodes[pos_node]['content']
                
                pos_tokens = tokenizer(
                    pos_prompt,
                    pos_completion,
                    truncation=True,
                    max_length=512,
                    padding="max_length",
                    return_tensors="pt"
                )
                
                pos_examples.append({
                    "input_ids": pos_tokens.input_ids[0],
                    "attention_mask": pos_tokens.attention_mask[0],
                    "labels": pos_tokens.input_ids[0].clone()
                })
        
        for neg_node in neg_nodes:
            if 'content' in graph.nodes[neg_node]:
                neg_prompt = f"Question: {question}"
                neg_completion = graph.nodes[neg_node]['content']
                
                neg_tokens = tokenizer(
                    neg_prompt,
                    neg_completion,
                    truncation=True,
                    max_length=512,
                    padding="max_length",
                    return_tensors="pt"
                )
                
                neg_examples.append({
                    "input_ids": neg_tokens.input_ids[0],
                    "attention_mask": neg_tokens.attention_mask[0],
                    "labels": neg_tokens.input_ids[0].clone()
                })
    
    return pos_examples, neg_examples

def main():
    parser = argparse.ArgumentParser(description="Train a SocraticMAGDi model")
    
    # Model configuration
    parser.add_argument("--decomposer_model", type=str, default="gpt2", 
                        help="Model name for the decomposer")
    parser.add_argument("--solver_model", type=str, default="gpt2", 
                        help="Model name for the solver")
    parser.add_argument("--gcn_in_channels", type=int, default=768, 
                        help="Input dimension for GCN")
    parser.add_argument("--gcn_hidden_channels", type=int, default=256, 
                        help="Hidden dimension for GCN")
    parser.add_argument("--gcn_out_channels", type=int, default=4, 
                        help="Output dimension for GCN (number of node classes)")
    
    # Loss weights
    parser.add_argument("--alpha", type=float, default=1.0, 
                        help="Weight for language modeling loss")
    parser.add_argument("--beta", type=float, default=1.0, 
                        help="Weight for node classification loss")
    parser.add_argument("--gamma", type=float, default=0.1, 
                        help="Weight for contrastive loss")
    parser.add_argument("--delta", type=float, default=0.5, 
                        help="Weight for decomposer-solver alignment loss")
    
    # Training configuration
    parser.add_argument("--dataset_path", type=str, default="data/dataset.json", 
                        help="Path to dataset file")
    parser.add_argument("--output_dir", type=str, default="outputs", 
                        help="Output directory for model checkpoints")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=5, 
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, 
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    parser.add_argument("--use_lora", action="store_true", 
                        help="Whether to use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--train_ratio", type=float, default=0.5, 
                    help="Ratio of data to use for training (default: 0.5)")
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Initialize agents
    agents = [
        {
            "id": i,
            "role": role,
            "instructions": role_specifications[role]["instructions"],
            "analysis": [],
            "previous_positions": [],
            "base_temp": 0.8 + random.uniform(-0.7, 0.7),  # Unique base temp per agent
            "weight": 1.0,  # Default weight, will be updated during training
            "accuracy": 0.0,  # Will track accuracy during training
            "influence_score": 0.0  # Will track how often this agent influences others
        } for i, role in enumerate(role_specifications)
    ]
    
    # Load dataset Put in specific dataset here
    dataset = load_dataset("commonsense_qa")
    dataset = dataset["train"].train_test_split(train_size=0.8, seed=42)

    # Split dataset into training and testing sets
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(
        dataset, 
        test_size=1-args.train_ratio, 
        random_state=args.seed
    )
    
    logger.info(f"Split dataset: {len(train_data)} training examples, {len(test_data)} testing examples")
    
    # Save test data for later evaluation
    with open(os.path.join(args.output_dir, "test_data.json"), "w") as f:
        json.dump(test_data, f)
    logger.info(f"Saved test data to {os.path.join(args.output_dir, 'test_data.json')}")
    
    # Extract questions and answers for agent training
    training_examples = []
    for item in train_data:
        training_examples.append({
            "question": item["question"],
            "answer": item["answer"],
            "options": item.get("options", [])
        })
    
    # Train agent weights using only the training data
    logger.info("Training agent weights using training data")
    agents = train_agent_weights(agents, training_examples, client)
    
    # Create MAG dataset from the training data
    test_questions = [item["question"] for item in test_data]
    test_options = [item.get("options", []) for item in test_data]
    
    logger.info("Creating MAG dataset from training data")
    mag_dataset = create_mag_dataset(test_questions, agents, client, test_options)
    
    # Save MAG dataset
    os.makedirs("data", exist_ok=True)
    with open("data/mag_dataset.pkl", "wb") as f:
        pickle.dump(mag_dataset, f)
    
    logger.info(f"Created MAG dataset with {len(mag_dataset)} examples")
    
    # Continue with the rest of the training process...
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.decomposer_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare examples for Socratic model components
    logger.info("Preparing examples for Socratic model components")
    decomposer_examples, solver_examples = prepare_socratic_examples(mag_dataset, tokenizer)
    
    # Prepare positive and negative examples for contrastive learning
    logger.info("Preparing examples for contrastive learning")
    pos_examples, neg_examples = prepare_pos_neg_examples(mag_dataset, tokenizer)
    
    # Extract PyG graphs
    pyg_graphs = [item["pyg_data"] for item in mag_dataset]
    
    # Create dataset
    dataset = MAGDiDataset(
        decomposer_examples=decomposer_examples,
        solver_examples=solver_examples,
        pos_examples=pos_examples,
        neg_examples=neg_examples,
        graphs=pyg_graphs
    )
    
    # Initialize model
    logger.info("Initializing SocraticMAGDi model")
    model = SocraticMAGDi(
        decomposer_name=args.decomposer_model,
        solver_name=args.solver_model,
        gcn_in_channels=args.gcn_in_channels,
        gcn_hidden_channels=args.gcn_hidden_channels,
        gcn_out_channels=args.gcn_out_channels,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        delta=args.delta
    )
    
    # Apply LoRA if specified
    if args.use_lora:
        logger.info("Applying LoRA for parameter-efficient fine-tuning")
        
        # Prepare decomposer for LoRA
        model.decomposer.model = prepare_model_for_kbit_training(model.decomposer.model)
        decomposer_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["c_attn", "c_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model.decomposer.model = get_peft_model(model.decomposer.model, decomposer_config)
        
        # Prepare solver for LoRA
        model.solver.model = prepare_model_for_kbit_training(model.solver.model)
        solver_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["c_attn", "c_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model.solver.model = get_peft_model(model.solver.model, solver_config)
    
    # Initialize trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        fp16=True,
        report_to="tensorboard"
    )
    
    trainer = SocraticMAGDiTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=SocraticMAGDiDataCollator(tokenizer)
    )
    
    # Train model
    logger.info("Training SocraticMAGDi model")
    trainer.train()
    
    # Save final model
    logger.info(f"Saving model to {args.output_dir}/final")
    trainer.save_model(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()
