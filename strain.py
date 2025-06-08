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
import openai

#from smodel import SocraticMAGDi, SocraticMAGDiDataCollator
from datasets import load_dataset
import random
from collections import Counter
import re
import sys
import tensorboardX

# Agent role specifications
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


def get_recommended_model(model_size="small"):
    """Get recommended models based on size requirements"""
    models = {
        "small": {
            "decomposer": "Qwen/Qwen2-1.5B",
            "solver": "Qwen/Qwen2-1.5B"
        },
        "medium": {
            "decomposer": "Qwen/Qwen2-7B",
            "solver": "Qwen/Qwen2-7B"
        },
        "large": {
            "decomposer": "meta-llama/Llama-3.2-3B",
            "solver": "meta-llama/Llama-3.2-3B"
        }
    }
    return models.get(model_size, models["small"])


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

class SocraticMAGDiTrainer(Trainer):
    """Custom trainer for the SocraticMAGDi model."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch = None):
        """
        Compute the combined loss for the SocraticMAGDi model.
        """
        outputs = model(
            decomposer=inputs["decomposer"],
            solver=inputs["solver"],
            pos=inputs["pos"],
            neg=inputs["neg"],
            graph=inputs["graph"]
        )

        lm_loss, node_loss, mr_loss, alignment_loss = outputs
        total_loss = lm_loss + node_loss + mr_loss + alignment_loss

        if return_outputs:
            return total_loss, outputs

        return total_loss


def generate_analysis(agent, prompt, client, debate_round=0):
    """Generate analysis from an agent using OpenAI API"""
    base_temp = agent.get('temperature', 0.7)  # Default to 0.7 if not specified

    # Optional: Still apply debate round scaling if desired
    temp = base_temp * (1 + 0.1 * debate_round)  # Increase temp in later rounds
    messages = [
        {"role": "system", "content": agent['instructions']},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=temp,
        max_tokens=600
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
    print("Training agent weights")

    # Reset metrics
    for agent in agents:
        agent['accuracy'] = 0.0
        agent['correct_count'] = 0
        agent['total_count'] = 0

    # Evaluate each agent on training data
    for item in tqdm(training_data, desc="Training agents"):
        print(item)
        question = item['question']
        print(question)
        correct_answer = str(item['answerKey'])
        options = item.get('options', [])
        print(options)

        for agent in agents:
            prompt = (
                f"<System Protocol>DO NOT INCLUDE REASONING IN OUTPUT</System>\n\n"
                f"{question}{options}\n\n"
                f"<Analysis Steps (INTERNAL USE ONLY)>\n"
                f"1. Surface assumptions → 2. Verify facts → 3. Generate alternatives → 4. Validate\n\n"
                f"<Output Requirements>\n"
                f"- Start with '{{' and end with '}}'\n"
                f"- No surrounding text or analysis steps\n\n"
                f"{agent['instructions']}\n\n"
                f"<BAD Example>DO NOT OUTPUT LIKE THIS:\n"
                f"'First, I considered...' {{\"decision\":...}}\n"
            )

            response = generate_analysis(agent, prompt, client)
            print(response)
            parsed = parse_json_response(response)
            decision = parsed.get("decision", "")

            # Update accuracy metrics
            agent['total_count'] += 1
            print(decision)
            print(correct_answer)
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
    print("Agent Training Results:")
    for agent in agents:
        print(f"{agent['role']}: Accuracy = {agent['accuracy']:.4f}, Weight = {agent['weight']:.4f}")

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


def create_debate_graph(agents, question, gold_answer=None, decision=None, is_correct=None):
    """
    Create a debate graph (MAG) with ground truth and correctness annotations.
    Each response node records if its decision matches the gold answer.
    """
    G = nx.DiGraph()

    # 1. Add question node
    G.add_node("question", content=question, type="question", round=-1)

    # 2. Add ground truth node and link it to the question
    if gold_answer is not None:
        G.add_node(
            "ground_truth",
            content=gold_answer,
            type="ground_truth"
        )
        G.add_edge("question", "ground_truth", type="has_ground_truth")

    # 3. Add each agent's responses and debate rounds
    max_rounds = max(len(a['analysis']) for a in agents)
    for agent in agents:
        role = agent['role']
        for round_num, response in enumerate(agent['analysis']):
            node_id = f"{role}_{round_num}"
            parsed = parse_json_response(response)
            decision = (parsed.get("decision") or "").strip().lower()

            # Determine correctness against gold answer
            is_correct = (decision == gold_answer.strip().lower()) if gold_answer else None

            # Create node with correctness annotation
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

            # Connect to question or previous round
            if round_num == 0:
                G.add_edge("question", node_id, type="responds_to")
            else:
                prev_node = f"{role}_{round_num - 1}"
                G.add_edge(prev_node, node_id, type="continues")

            # 4. Add weighted influences edges
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

    # Add the reverse edges
    for dst, src, data in edges_to_add:
        G.add_edge(dst, src, **data)
    return G


def has_consensus(agents):
    """
    Returns (True, decision) if every agent's last decision matches,
    else (False, None). Handles missing or malformed JSON.
    """
    decisions = []

    for ag in agents:
        raw = ag['analysis'][-1]
        # 1) Try JSON parsing
        try:
            parsed = parse_json_response(raw)
            decision = parsed.get("decision", "")
        except Exception:
            decision = ""
        # 2) Fallback regex if JSON failed or key absent
        if not decision:
            m = re.search(r'"decision"\s*:\s*"([^"]+)"', raw)
            decision = m.group(1) if m else ""
        norm = decision.strip().lower()
        if not norm:
            return False, None
        decisions.append(norm)

    # 3) Check unanimous agreement
    counts = Counter(decisions)
    if len(counts) == 1:
        return True, decisions[0]
    return False, None


def layered_consensus_process(
        question,
        agents,
        client,
        base_options,
        gold_answer
):
    """Run the multi-agent debate process with weighted consensus and ground truth comparison"""
    discussion_history = []
    discussion_history.append(f"GOLD_ANSWER: {gold_answer}")
    print("=== INITIAL ROUND ===")

    # Reset agent analysis for new question
    for agent in agents:
        agent['analysis'] = []

    # Initial responses
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

        response = generate_analysis(agent, prompt, client)
        agent['analysis'] = [response]
        discussion_history.append(f"INITIAL - {agent['role']}:\n{response}")
        print(f"[{agent['role']} INITIAL]\n{response}")

    # Check initial consensus
    consensus, decision = has_consensus(agents)
    if consensus:
        print(f"Consensus reached on '{decision}' in initial round")
        is_correct = (decision == gold_answer.strip().lower()) if gold_answer else None
        discussion_history.append(f"CONSENSUS_CORRECT: {is_correct}")
        return discussion_history, create_debate_graph(agents, question, gold_answer, decision, is_correct)

    # Debate rounds with temperature escalation
    for round_num in range(2):
        print(f"=== DEBATE ROUND {round_num + 1} ===")
        for i, agent in enumerate(agents):
            # Include precalculated agent weights with their analyses (excluding current agent)
            others = "\n".join([
                f"{other['role']} (Weight: {other['weight']:.3f}): {parse_json_response(other['analysis'][-1]).get('analysis', 'No analysis available')}"
                for j, other in enumerate(agents) if j != i  # Excludes current agent's own analysis
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
                "      - Format EXACTLY as: [\"Role1\", \"Role2\"]\n"
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

            response = generate_analysis(agent, prompt, client, debate_round=round_num + 1)
            agent['analysis'].append(response)
            discussion_history.append(f"ROUND {round_num + 1} - {agent['role']}:\n{response}")
            print(f"[{agent['role']} REFINED]\n{response}")

        # Track influence after each round
        influence_counts = track_influence(agents)
        print(f"Influence counts after round {round_num + 1}: {influence_counts}")
        consensus, decision = has_consensus(agents)
        if consensus:
            print(f"Consensus reached on '{decision}' at round {round_num + 1}")
            is_correct = (decision == gold_answer) if gold_answer else None
            discussion_history.append(f"CONSENSUS_CORRECT: {is_correct}")
            return discussion_history, create_debate_graph(agents, question, gold_answer, decision, is_correct)

    # --- Weighted-voting fallback if no unanimity reached ---
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
        print("Warning: sentence-transformers not installed. Using random embeddings.")
        # Fallback to random embeddings
        dim = 768  # Default embedding dimension
        embeddings = {}
        for node_id in graph.nodes():
            embeddings[node_id] = np.random.normal(0, 0.1, dim)
        return embeddings


def convert_to_pyg_data(graph, embeddings):
    """
    Convert a NetworkX debate graph to PyTorch Geometric Data object,
    using correctness as node labels (1=correct, 0=incorrect).
    Adds bidirectional edges for proper message passing.
    """
    # Sort nodes for consistent ordering
    node_list = list(graph.nodes())
    node_list = [n for n in node_list if graph.nodes[n].get('type') != 'ground_truth']

    # Build feature matrix X from provided embeddings
    x = []
    for node in node_list:
        emb = embeddings[node] if isinstance(embeddings, dict) else embeddings[node_list.index(node)]
        x.append(emb)
    x = torch.tensor(np.stack(x), dtype=torch.float)

    # Build edge index and edge attributes with bidirectional edges
    edge_index = []
    edge_attr = []
    for src, dst, data in graph.edges(data=True):
        if src in node_list and dst in node_list:
            edge_type = data.get('type', '')
            weight = data.get('weight', 0.0) if edge_type == 'influences' else 0.0

            # Original direction
            edge_index.append([node_list.index(src), node_list.index(dst)])
            edge_attr.append([weight])

            # Reverse direction
            edge_index.append([node_list.index(dst), node_list.index(src)])
            edge_attr.append([weight])
    # Convert to tensors
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0),
                                                                                                            dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float) if edge_attr else None

    # Labels: 1 if node's decision matches gold, else 0
    y = []
    for node in node_list:
        node_data = graph.nodes[node]
        y.append(1 if node_data.get('is_correct', False) else 0)
    y = torch.tensor(y, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def create_mag_dataset(training_data, agents, client):
    """Create a MAG dataset from multiple questions"""
    mag_dataset = []

    for item in tqdm(training_data, desc="Creating MAG dataset"):
        print(item)
        question = item['question']
        print(question)
        options = item.get('options', [])
        print(options)
        gold_answer = str(item['gold_answer'])
        _, debate_graph = layered_consensus_process(question, agents, client, options, gold_answer)

        # Extract embeddings
        embeddings = extract_node_embeddings(debate_graph)

        # Convert to PyG Data
        pyg_data = convert_to_pyg_data(debate_graph, embeddings)

        # Add to dataset
        mag_dataset.append({
            "question": question,
            "options": options,
            "graph": debate_graph,
            "pyg_data": pyg_data
        })

    return mag_dataset


def prepare_socratic_examples(mag_dataset, tokenizer):
    """Prepare examples for training the Socratic model components, grouped by MAG item"""
    decomposer_examples = []  # List of lists (one per MAG item)
    solver_examples = []  # List of lists (one per MAG item)

    for item in mag_dataset:
        question = item["question"]
        graph = item["graph"]

        # Per-item lists
        item_decomposer = []
        item_solver = []

        # Extract sub-questions (nodes with highest influence)
        sub_questions = []
        for node, data in graph.nodes(data=True):
            if data.get('type') == 'response' and graph.in_degree(node) > 1:
                sub_questions.append(data.get('content', ''))

        if sub_questions:
            # Create decomposer example for this item
            decomposer_prompt = f"Question: {question}\nBreak this down into sub-questions:"
            decomposer_completion = "\n".join([f"- {sq[:100]}..." for sq in sub_questions[:3]])

            # Tokenize immediately for this item
            prompt_tokens = tokenizer(
                decomposer_prompt,
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt"
            )
            completion_tokens = tokenizer(
                decomposer_completion,
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt"
            )
            item_decomposer.append({
                "prompt_input_ids": prompt_tokens.input_ids[0],
                "prompt_attention_mask": prompt_tokens.attention_mask[0],
                "completion_input_ids": completion_tokens.input_ids[0],
                "completion_attention_mask": completion_tokens.attention_mask[0]
            })

        # Find sub-question answers (solver examples) for this item
        for node, data in graph.nodes(data=True):
            if data.get('type') == 'response' and data.get('round', 0) > 0:
                influencers = []
                for pred in graph.predecessors(node):
                    if graph.nodes[pred].get('type') == 'response':
                        influencers.append(pred)

                if influencers:
                    influencer_content = graph.nodes[influencers[0]].get('content', '')
                    solver_prompt = f"Question: {influencer_content[:100]}..."
                    solver_completion = data.get('content', '')

                    # Tokenize immediately for this item
                    prompt_tokens = tokenizer(
                        solver_prompt,
                        truncation=True,
                        max_length=512,
                        padding="max_length",
                        return_tensors="pt"
                    )
                    completion_tokens = tokenizer(
                        solver_completion,
                        truncation=True,
                        max_length=512,
                        padding="max_length",
                        return_tensors="pt"
                    )
                    item_solver.append({
                        "prompt_input_ids": prompt_tokens.input_ids[0],
                        "prompt_attention_mask": prompt_tokens.attention_mask[0],
                        "completion_input_ids": completion_tokens.input_ids[0],
                        "completion_attention_mask": completion_tokens.attention_mask[0]
                    })

        # Add per-item lists to main lists
        decomposer_examples.append(item_decomposer)
        solver_examples.append(item_solver)

    return decomposer_examples, solver_examples


def prepare_pos_neg_examples(mag_dataset, tokenizer, max_path_length=4):
    """Prepare positive/negative examples with correct input-label pairs"""
    pos_examples = []
    neg_examples = []

    for item in mag_dataset:
        question = item["question"]
        graph = item["graph"]
        
        item_pos = []
        item_neg = []

        # Get response nodes
        response_nodes = [
            n for n, data in graph.nodes(data=True)
            if data.get('type') in ['response', 'initial_response']
        ]

        # Process positive examples (correct reasoning)
        correct_nodes = [
            n for n in response_nodes 
            if graph.nodes[n].get('is_correct', False)
        ]
        
        for node in correct_nodes[:3]:  # Limit to 3 examples per item
            content = graph.nodes[node].get('content', '')
            if not content or len(content.strip()) < 10:
                continue  # Skip empty or very short content
                
            # Create proper prompt-completion pair
            prompt = f"Question: {question}\nProvide valid reasoning:"
            
            # Tokenize prompt and completion separately
            prompt_tokens = tokenizer(
                prompt,
                truncation=True,
                max_length=256,  # Shorter to leave room for completion
                padding=False,
                return_tensors="pt"
            )
            
            completion_tokens = tokenizer(
                content,
                truncation=True,
                max_length=256,
                padding=False,
                return_tensors="pt"
            )
            
            # Create input_ids by concatenating prompt + completion
            input_ids = torch.cat([
                prompt_tokens.input_ids[0],
                completion_tokens.input_ids[0]
            ])
            
            # Create labels: -100 for prompt tokens, completion tokens for loss
            labels = torch.cat([
                torch.full((len(prompt_tokens.input_ids[0]),), -100),  # Ignore prompt in loss
                completion_tokens.input_ids[0]  # Predict completion
            ])
            
            # Pad to max_length
            max_len = 512
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
                labels = labels[:max_len]
            else:
                # Pad with tokenizer.pad_token_id
                pad_length = max_len - len(input_ids)
                input_ids = torch.cat([
                    input_ids,
                    torch.full((pad_length,), tokenizer.pad_token_id)
                ])
                labels = torch.cat([
                    labels,
                    torch.full((pad_length,), -100)  # Ignore padding in loss
                ])
            
            # Create attention mask
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
            
            item_pos.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            })

        # Process negative examples (incorrect reasoning)
        incorrect_nodes = [
            n for n in response_nodes 
            if not graph.nodes[n].get('is_correct', False)
        ]
        
        for node in incorrect_nodes[:3]:  # Limit to 3 examples per item
            content = graph.nodes[node].get('content', '')
            if not content or len(content.strip()) < 10:
                continue
                
            prompt = f"Question: {question}\nIdentify flawed reasoning:"
            
            # Same process as positive examples
            prompt_tokens = tokenizer(
                prompt,
                truncation=True,
                max_length=256,
                padding=False,
                return_tensors="pt"
            )
            
            completion_tokens = tokenizer(
                content,
                truncation=True,
                max_length=256,
                padding=False,
                return_tensors="pt"
            )
            
            input_ids = torch.cat([
                prompt_tokens.input_ids[0],
                completion_tokens.input_ids[0]
            ])
            
            labels = torch.cat([
                torch.full((len(prompt_tokens.input_ids[0]),), -100),
                completion_tokens.input_ids[0]
            ])
            
            # Pad to max_length
            max_len = 512
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
                labels = labels[:max_len]
            else:
                pad_length = max_len - len(input_ids)
                input_ids = torch.cat([
                    input_ids,
                    torch.full((pad_length,), tokenizer.pad_token_id)
                ])
                labels = torch.cat([
                    labels,
                    torch.full((pad_length,), -100)
                ])
            
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
            
            item_neg.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            })

        pos_examples.append(item_pos)
        neg_examples.append(item_neg)

    return pos_examples, neg_examples
def validate_and_fix_examples(pos_examples, neg_examples, tokenizer):
    """Fix malformed examples that cause NaN losses"""
    fixed_pos = []
    fixed_neg = []
    
    for i, (pos_list, neg_list) in enumerate(zip(pos_examples, neg_examples)):
        # Fix positive examples
        if not pos_list:  # Empty list
            # Create dummy positive example
            dummy_text = "This is a valid reasoning step."
            tokens = tokenizer(
                dummy_text,
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt"
            )
            pos_list = [{
                "input_ids": tokens.input_ids[0],
                "attention_mask": tokens.attention_mask[0],
                "labels": tokens.input_ids[0].clone()
            }]
        else:
            # Validate existing examples
            validated_pos = []
            for pos_item in pos_list:
                # Check for valid tensors
                if (torch.is_tensor(pos_item.get("input_ids")) and 
                    torch.is_tensor(pos_item.get("attention_mask")) and
                    torch.is_tensor(pos_item.get("labels"))):
                    
                    # Check for all-zero or invalid tokens
                    if pos_item["input_ids"].sum() > 0:
                        validated_pos.append(pos_item)
            
            if not validated_pos:  # All examples were invalid
                # Create dummy
                dummy_text = "This is a valid reasoning step."
                tokens = tokenizer(
                    dummy_text,
                    truncation=True,
                    max_length=512,
                    padding="max_length",
                    return_tensors="pt"
                )
                pos_list = [{
                    "input_ids": tokens.input_ids[0],
                    "attention_mask": tokens.attention_mask[0],
                    "labels": tokens.input_ids[0].clone()
                }]
            else:
                pos_list = validated_pos
        
        # Same validation for negative examples
        if not neg_list:
            dummy_text = "This is a flawed reasoning step."
            tokens = tokenizer(
                dummy_text,
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt"
            )
            neg_list = [{
                "input_ids": tokens.input_ids[0],
                "attention_mask": tokens.attention_mask[0],
                "labels": tokens.input_ids[0].clone()
            }]
        
        fixed_pos.append(pos_list)
        fixed_neg.append(neg_list)
    
    return fixed_pos, fixed_neg


# Cell: Main Training Function (Jupyter Compatible)
def main():
    # Configuration Variables (Replace all args with direct variables)
    torch.cuda.empty_cache()
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    SEED = 42
    MODEL_SIZE = "small"  # choices: "small", "medium", "large"
    DECOMPOSER_MODEL = "Qwen/Qwen2-1.5B"
    SOLVER_MODEL = "Qwen/Qwen2-1.5B"
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
    NUM_EPOCHS = 5
    BATCH_SIZE = 2
    LEARNING_RATE = 1e-6
    
    # OpenAI API Key
    OPENAI_API_KEY = ENV[‘AUTH_TOKEN’]
    
    set_seed(SEED)

    # Update model selection based on size
    recommended_models = get_recommended_model(MODEL_SIZE)
    if DECOMPOSER_MODEL == "Qwen/Qwen2-1.5B" and MODEL_SIZE != "small":
        DECOMPOSER_MODEL = recommended_models["decomposer"]
    if SOLVER_MODEL == "Qwen/Qwen2-1.5B" and MODEL_SIZE != "small":
        SOLVER_MODEL = recommended_models["solver"]

    print(f"Using decomposer model: {DECOMPOSER_MODEL}")
    print(f"Using solver model: {SOLVER_MODEL}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize OpenAI client
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
    print("Sample data:", full_train[1])
    full_train = full_train.train_test_split(test_size=0.20, seed=41)

    # Split the 80% into two 40% halves
    subsplits = full_train["train"].train_test_split(test_size=0.50)
    agent_data = subsplits["train"]
    mag_creation_data = subsplits["test"]
    
    print(f"Agent weight set: {len(agent_data)} examples")
    print(f"MAG creation set: {len(mag_creation_data)} examples")

    # Set pre-computed agent weights (faster than training)
    for agent in agents:
        if agent["role"] == "Scientist":
            agent["weight"] = 0.2013
        elif agent["role"] == "Lawyer":
            agent["weight"] = 0.1992
        elif agent["role"] == "SocialChangeAdvocate":
            agent["weight"] = 0.1971
        elif agent["role"] == "Mathematician":
            agent["weight"] = 0.2031
        else:
            agent["weight"] = 0.1992

    # Load the saved MAG dataset from the pickle file
    with open('mag_dataset.pkl', 'rb') as f:
        mag_dataset = pickle.load(f)
    
    # Process dataset items
    for item in mag_dataset:
        graph = item["graph"]
        gold_answer = None
        final_decision = None
        max_round = -1

        # Extract gold_answer from any response node
        for node_id, node_data in graph.nodes(data=True):
            if node_data.get('type') in ['response', 'initial_response']:
                if node_data.get('gold_answer'):
                    gold_answer = str(node_data['gold_answer']).strip().lower()
                    break

        # Find the final consensus decision (latest round)
        for node_id, node_data in graph.nodes(data=True):
            if node_data.get('type') in ['response', 'initial_response']:
                round_num = node_data.get('round', 0)
                if round_num > max_round:
                    max_round = round_num
                    final_decision = node_data.get('decision', '').strip().lower()

        # Add fields to item
        item['gold_answer'] = gold_answer
        item['final_decision'] = final_decision
        item['is_correct'] = (final_decision == gold_answer) if final_decision and gold_answer else False

    # Filter dataset to keep only consensus examples
    round_counts = {"initial": 0, "round1": 0, "round2": 0, "no_consensus": 0}
    filtered_mag_dataset = []

    for item in mag_dataset:
        graph = item["graph"]
        consensus_round = None
        round_label = {0: "initial", 1: "round1", 2: "round2"}
        
        agent_roles = set(
            node_data.get('role')
            for node_id, node_data in graph.nodes(data=True)
            if node_data.get('type') in ['response', 'initial_response']
        )
        
        max_round = max(
            (node_data.get('round', 0)
             for node_id, node_data in graph.nodes(data=True)
             if node_data.get('type') in ['response', 'initial_response']),
            default=0
        )
        
        for r in range(max_round + 1):
            decisions = [
                node_data.get('decision')
                for node_id, node_data in graph.nodes(data=True)
                if node_data.get('type') in ['response', 'initial_response'] and node_data.get('round', 0) == r
            ]
            
            if len(decisions) == len(agent_roles) and len(set(decisions)) == 1:
                consensus_round = r
                break
        
        if consensus_round is not None:
            filtered_mag_dataset.append(item)
            label = round_label.get(consensus_round, f"round{consensus_round}")
            round_counts[label] += 1
        else:
            round_counts["no_consensus"] += 1

    print("Round distribution:", round_counts)

    # Remove initial round consensus examples
    filtered_by_rounds = []
    for item in filtered_mag_dataset:
        graph = item["graph"]
        max_round = max(
            (node_data.get('round', 0)
             for node_id, node_data in graph.nodes(data=True)
             if node_data.get('type') in ['response', 'initial_response']),
            default=0
        )
        if max_round > 0:
            filtered_by_rounds.append(item)

    mag_dataset = filtered_by_rounds
    print(f"After filtering: {len(mag_dataset)} examples")

    # Calculate MAS Consensus Accuracy
    print("\n=== Multi-Agent System (MAS) Final Consensus Accuracy ===")
    consensus_results = []
    
    for item in mag_dataset:
        question = item["question"]
        graph = item["graph"]
        final_decision = None
        gold_answer = None

        # Look for consensus decision in graph nodes
        for node_id, node_data in graph.nodes(data=True):
            if node_data.get('type') in ['response', 'initial_response']:
                if node_data.get('decision') and node_data.get('gold_answer'):
                    gold_answer = str(node_data['gold_answer']).strip().lower()
                    break

        # Get the final consensus decision by finding the most recent decision
        max_round = -1
        for node_id, node_data in graph.nodes(data=True):
            if node_data.get('type') in ['response', 'initial_response']:
                round_num = node_data.get('round', 0)
                if round_num > max_round:
                    max_round = round_num
                    final_decision = node_data.get('decision', '').strip().lower()

        # Determine if final consensus was correct
        if final_decision and gold_answer:
            is_consensus_correct = (final_decision == gold_answer)
            consensus_results.append({
                'question': question[:50] + '...',
                'final_decision': final_decision,
                'gold_answer': gold_answer,
                'correct': is_consensus_correct
            })

    # Calculate final consensus accuracy
    total_questions = len(consensus_results)
    correct_consensus = sum(1 for result in consensus_results if result['correct'])
    consensus_accuracy = correct_consensus / total_questions if total_questions > 0 else 0

    print(f"Total questions processed: {total_questions}")
    print(f"Correct final consensus decisions: {correct_consensus}")
    print(f"Incorrect final consensus decisions: {total_questions - correct_consensus}")
    print(f"MAS Final Consensus Accuracy: {consensus_accuracy:.1%} ({correct_consensus}/{total_questions})")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(DECOMPOSER_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare examples for Socratic model components
    print("Preparing examples for Socratic model components")
    decomposer_examples, solver_examples = prepare_socratic_examples(mag_dataset, tokenizer)

    # Prepare positive and negative examples for contrastive learning
    print("Preparing examples for contrastive learning")
    pos_examples, neg_examples = prepare_pos_neg_examples(mag_dataset, tokenizer)
    # Add this to your main() function after preparing examples:
    print("🔧 Validating and fixing examples...")
    pos_examples, neg_examples = validate_and_fix_examples(pos_examples, neg_examples, tokenizer)


    # Extract PyG graphs
    pyg_graphs = [item["pyg_data"] for item in mag_dataset]
    print(f"Decomposer examples: {len(decomposer_examples)}")
    print(f"Solver examples: {len(solver_examples)}")
    print(f"Positive examples: {len(pos_examples)}")
    print(f"Negative examples: {len(neg_examples)}")

    # Create dataset
    examples = []
    num_graphs = len(mag_dataset)
    for i in range(num_graphs):
        examples.append({
            "decomposer": decomposer_examples[i],
            "solver": solver_examples[i],
            "pos": pos_examples[i],
            "neg": neg_examples[i],
            "graph": pyg_graphs[i]
        })
    dataset = MAGDiDataset(examples)
    print(f"Final dataset size: {len(dataset)}")

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

    # Initialize trainer
    # In your main() function, change training args to:
    training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,                    # Reduce epochs
    per_device_train_batch_size=1,         # Reduce to 1
    gradient_accumulation_steps=4,          # Simulate batch size 4
    learning_rate=1e-7,                    # Very conservative LR
    weight_decay=0.0,                      
    max_grad_norm=0.1,                     
    logging_steps=1,                       
    save_strategy="no",                    
    warmup_steps=0,                        
    lr_scheduler_type="constant",
    save_safetensors=False,
    dataloader_pin_memory=False,
    fp16=False,                            # DISABLE FP16
    gradient_checkpointing=False,           # Enable for memory
    dataloader_num_workers=0,              # Disable multiprocessing
    )


    trainer = SocraticMAGDiTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=SocraticMAGDiDataCollator(tokenizer)
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

    # Save training metadata
    training_metadata = {
        "training_args": {
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "use_lora": False
        },
        "dataset_info": {
            "num_examples": len(dataset),
            "num_mag_graphs": len(mag_dataset)
        },
        "agent_weights": {agent["role"]: agent["weight"] for agent in agents}
    }

    with open(os.path.join(final_dir, "training_info.json"), "w") as f:
        json.dump(training_metadata, f, indent=2)

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
