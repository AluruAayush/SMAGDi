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
#from smodel import SocraticMAGDi, SocraticMAGDiDataCollator
from datasets import load_dataset
import random
from collections import Counter
import re
import sys
import tensorboardX

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

    "SocialChangeAdvocate": {
        "temperature": 0.8,
        "instructions": """
You are a Social Change Advocate. When making decisions:

ALWAYS DO:
- Project impacts across 10/25/100 year horizons
- Model power shift dynamics using differential game theory
- Quantify intersectional disadvantage indices
- Identify which groups are helped vs. harmed by each option
- Challenge existing power structures and inequalities
- Push for solutions that redistribute resources to marginalized communities
- Question whose voices are missing from the decision-making process
- Demand accountability mechanisms for those in power
- Apply intersectionality framework to all analyses
- Make Decision based on this

RESPONSE FORMAT(DON'T PUT '''json before this):
{
  "decision": "<option>",
  "analysis": {
  "power_impact": "<who gains/loses power>",
  "equity_assessment": "<effect on marginalized groups>",
  "temporal_horizons": "<10/25/100 year projections>",
  "intersectional_analysis": "<multi-dimensional disadvantage assessment>",
  "missing_voices": "<excluded stakeholders>",
  "accountability_mechanisms": "<power oversight structures>"
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
# === Llama 3.1 8B Instruct Setup ===
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

def llama_generate(messages, temperature=0.7, max_tokens=600):
    # messages: list of {"role":..., "content":...}
    # Use chat template for prompt formatting
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )
    # Move all tensors to the model's device
    inputs = {k: v.to(llama_pipeline.model.device) for k, v in inputs.items()}
    outputs = llama_pipeline.model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,  # Explicitly set pad_token_id
        eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    )
    response = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

# === Dataset class (unchanged) ===
class MAGDiDataset(Dataset):
    """
    Dataset for training SocraticMAGDi with variable-length lists for each example.
    Each item is a dict with lists of decomposer, solver, pos, neg examples (each with input_ids, attention_mask, labels), and a graph.
    """

    def __init__(self, examples):
        """
        Args:
            examples (list): List of dicts, each dict has:
                - decomposer: list of dicts, each with 'prompt_input_ids', 'prompt_attention_mask', 'completion_input_ids', 'completion_attention_mask'
                - solver: list of dicts, each with 'prompt_input_ids', 'prompt_attention_mask', 'completion_input_ids', 'completion_attention_mask'
                - pos: list of dicts, each with 'input_ids', 'attention_mask', 'labels'
                - neg: list of dicts, each with 'input_ids', 'attention_mask', 'labels'
                - graph: PyG Data object
        """
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            # Each is a list of dicts (possibly variable length per example)
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
# === Llama-based agent analysis ===
def generate_analysis(agent, prompt, debate_round=0):
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
    for item in tqdm(training_data, desc="Training agents"):
        question = item['question']
        correct_answer = str(item['answerKey'])
        options = item.get('options', [])
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
            response = generate_analysis(agent, prompt)
            print(response)
            parsed = parse_json_response(response)
            decision = parsed.get("decision", "")
            print(decision)
            print(correct_answer)
            agent['total_count'] += 1
            if decision == correct_answer:
                agent['correct_count'] += 1
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

def layered_consensus_process(
        question,
        agents,
        base_options,
        gold_answer
):
    discussion_history = []
    discussion_history.append(f"GOLD_ANSWER: {gold_answer}")
    print("=== INITIAL ROUND ===")
    for agent in agents:
        agent['analysis'] = []
    for agent in agents:
        prompt = (
            f"<System Protocol>DO NOT INCLUDE REASONING IN OUTPUT</System>\n\n"
            f"{question}{base_options}\n\n"
            f"<Analysis Steps (INTERNAL USE ONLY)>\n"
            f"1. Surface assumptions → 2. Verify facts → 3. Generate alternatives → 4. Validate\n\n"
            f"<Output Requirements>\n"
            f"- Start with '{{' and end with '}}'\n"
            f"- No surrounding text or analysis steps\n\n"
            f"{agent['instructions']}\n\n"
            f"<BAD Example>DO NOT OUTPUT LIKE THIS:\n"
            f"'First, I considered...' {{\"decision\":...}}\n"
        )
        response = generate_analysis(agent, prompt)
        agent['analysis'] = [response]
        discussion_history.append(f"INITIAL - {agent['role']}:\n{response}")
        print(f"[{agent['role']} INITIAL]\n{response}")
    consensus, decision = has_consensus(agents)
    if consensus:
        print(f"Consensus reached on '{decision}' in initial round")
        is_correct = (decision == gold_answer.strip().lower()) if gold_answer else None
        discussion_history.append(f"CONSENSUS_CORRECT: {is_correct}")
        return discussion_history, create_debate_graph(agents, question, gold_answer, decision, is_correct)
    for round_num in range(2):
        print(f"=== DEBATE ROUND {round_num + 1} ===")
        for i, agent in enumerate(agents):
            others = "\n".join([
                f"{other['role']} (Weight: {other['weight']:.3f}): {parse_json_response(other['analysis'][-1]).get('analysis', 'No analysis available')}"
                for j, other in enumerate(agents) if j != i
            ])
            prompt = (
                f"Peer Analyses (with credibility weights):\n{others}\n\n"
                "Re-evaluate through your professional lens using:\n\n"
                "1. **Weighted Perspective Integration**\n"
                "   a. Calculate argument credibility using:\n"
                "      - Source weight * {role}_relevance_score\n"
                "      - Minimum 2 conflicting perspectives must be preserved\n"
                "   b. For high-weight peers (>0.6):\n"
                "      - Apply {role}_counteranalysis protocol\n"
                "      - Require 3x evidence verification\n\n"
                "2. **Persona-Centric Revision**\n"
                "   a. If changing decision:\n"
                "      - Must pass {core_principles}_checklist\n"
                "      - Show direct alignment with 2+ {role}_decision_factors\n"
                "   b. If keeping decision:\n"
                "      - Incorporate 1+ valid peer insight\n"
                "      - Strengthen with {role}_specific_evidence\n\n"
                "3. **Influence Documentation**\n"
                "   a. **Mandatory Field** - You MUST credit at least 2 peers using:\n"
                "      - Roles that passed {role}_verification_threshold\n"
                "      - Roles providing novel {domain}_insights\n"
                "   b. **Validation Protocol**\n"
                "      - Responses WITHOUT 'influenced_by' will be considered invalid\n"
                "      - Cite 1-3 roles using exact role names from peer analyses\n"
                "   c. **Example Enforcement**:\n"
                "      GOOD: \"influenced_by\": [\"Mathematician\", \"Ethicist\"]\n"
                "      BAD: \"influenced_by\": [] or missing field → REJECTED\n\n"
                "Answer As Follows:"
                "Response Requirements:\n"
                "- Use YOUR SPECIFIED RESPONSE FORMAT\n"
                "- ADD 'influenced_by' field at END\n"
                "- Maintain original JSON structure\n"
                f"- Only use specified options: {base_options}\n\n"
                "Begin professional analysis:"
            )
            response = generate_analysis(agent, prompt, debate_round=round_num + 1)
            agent['analysis'].append(response)
            discussion_history.append(f"ROUND {round_num + 1} - {agent['role']}:\n{response}")
            print(f"[{agent['role']} REFINED]\n{response}")
        influence_counts = track_influence(agents)
        print(f"Influence counts after round {round_num + 1}: {influence_counts}")
        consensus, decision = has_consensus(agents)
        if consensus:
            print(f"Consensus reached on '{decision}' at round {round_num + 1}")
            is_correct = (decision == gold_answer) if gold_answer else None
            discussion_history.append(f"CONSENSUS_CORRECT: {is_correct}")
            return discussion_history, create_debate_graph(agents, question, gold_answer, decision, is_correct)
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

def create_mag_dataset(training_data, agents):
    mag_dataset = []
    for item in tqdm(training_data, desc="Creating MAG dataset"):
        question = item['question']
        options = item.get('options', [])
        gold_answer = str(item['gold_answer'])
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
    full_train = load_dataset("cais/mmlu", "all", split = "auxiliary_train")
    random_indices = random.sample(range(len(full_train)), 200)
    agent_data= full_train.select(random_indices)
    print(f"Agent weight set: {len(agent_data)} examples")
    training_examples = [
        {"question": item["question"],
         "answerKey": item["answer"],
         "options": item["choices"]}
        for item in agent_data
    ]
    agents = train_agent_weights(agents, training_examples)
    sys.exit()
    training_data = [
         {"question": item["question"],
        "gold_answer": item["answer"],
          "options": item["choices"]}
        for item in mag_creation_data
     ]
    mag_dataset = create_mag_dataset(training_data, agents)
    os.makedirs("data", exist_ok=True)
    with open("data/mag_dataset.pkl", "wb") as f:
        pickle.dump(mag_dataset, f)

if __name__ == "__main__":
    main()
