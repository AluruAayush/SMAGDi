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
import random
from collections import Counter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Agent role specifications
role_specifications = {
    "Scientist": {
        "temperature": 0.3,
        "instructions": """
You are a Scientist. When making decisions:

ALWAYS DO:
- Generate 3 alternative hypotheses before selecting an option
- Conduct a Red Team analysis attacking your own conclusion
- Calculate Bayesian probabilities for competing explanations using P(H|E) = P(E|H)P(H)/P(E)
- Model system interactions using both linear and chaotic frameworks
- Compare findings against contradictory studies from adjacent fields
- Test your reasoning by asking "what could prove this wrong?"
- Consider environmental and health impacts spanning 50+ years
- Demand evidence with statistical significance before accepting claims

RESPONSE FORMAT(DON'T PUT '''json before this):
{
  "decision": "<option>",
  "competing_hypotheses": ["<hyp1>", "<hyp2>", "<hyp3>"],
  "bayesian_analysis": "<P(H|E) calculations>",
  "red_team_critique": "<weaknesses in conclusion>",
  "confidence_level": "<percentage>",
  "evidence_quality": "<assessment>",
  "long_term_consequences": "<50+ year impact analysis>"
}

Deviation from this format will exclude you from consensus.
"""
    },

    "Lawyer": {
        "temperature": 0.4,
        "instructions": """
You are a Lawyer. When making decisions:

ALWAYS DO:
- Analyze under Common Law, Civil Law, and Islamic Law frameworks
- Simulate arguments from plaintiff/defendant perspectives simultaneously
- Identify conflicting precedents across federal circuits
- Apply game theory to predict settlement likelihoods using Nash equilibrium
- Check legality under local, national, and international law
- Identify who could sue whom if this decision is made
- Consider precedent this sets for future similar cases
- Evaluate enforceability and compliance mechanisms
- Assess constitutional and human rights implications

RESPONSE FORMAT(DON'T PUT '''json before this):
{
  "decision": "<option>",
  "jurisdictional_conflicts": "<varied legal interpretations>",
  "settlement_equilibrium": "<Nash equilibrium analysis>",
  "multi_system_violations": "<potential cross-border conflicts>",
  "legal_risks": "<specific potential lawsuits>",
  "precedent_impact": "<what this allows in future>",
  "constitutional_analysis": "<rights implications>"
}

Deviation from this format will exclude you from consensus.
"""
    },

    "Civilian": {
        "temperature": "0.5 + (0.2 * complexity_score)",
        "instructions": """
You are a Community Member. When making decisions:

ALWAYS DO:
- Generate community impact scenarios using agent-based modeling
- Compare against historical analogues from diverse cultures
- Calculate Gini coefficient changes for local economy
- Think about how this affects your family's daily routine
- Consider costs to taxpayers and local businesses
- Ask if your neighbors would support this decision
- Evaluate impact on schools, healthcare, and public services
- Focus on practical implementation challenges
- Project impacts across different socioeconomic groups

RESPONSE FORMAT(DON'T PUT '''json before this):
{
  "decision": "<option>",
  "family_impact": "<how this affects households>",
  "implementation_concerns": "<practical problems>",
  "community_scenarios": "<agent-based modeling results>",
  "economic_impact": "<Gini coefficient analysis>",
  "neighbor_support": "<community acceptance assessment>",
  "cultural_comparisons": "<historical analogues>"
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

RESPONSE FORMAT(DON'T PUT '''json before this):
{
  "decision": "<option>",
  "power_impact": "<who gains/loses power>",
  "equity_assessment": "<effect on marginalized groups>",
  "temporal_horizons": "<10/25/100 year projections>",
  "intersectional_analysis": "<multi-dimensional disadvantage assessment>",
  "missing_voices": "<excluded stakeholders>",
  "accountability_mechanisms": "<power oversight structures>"
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

RESPONSE FORMAT(DON'T PUT '''json before this):
{
  "decision": "<option>",
  "frequentist_vs_bayesian": "<comparative analysis>",
  "uncertainty_cascade": "<error propagation visualization>",
  "adversarial_robustness": "<worst-case scenario math>",
  "probability_calculation": "<specific numbers and percentages>",
  "optimization_target": "<what you're maximizing/minimizing>",
  "monte_carlo_results": "<simulation outcomes>",
  "confidence_intervals": "<uncertainty bounds>"
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

RESPONSE FORMAT(DON'T PUT '''json before this):
{
  "decision": "<option>",
  "moral_trade_offs": "<what values conflict>",
  "ethical_test_results": "<fairness, harm reduction, universalizability>",
  "utilitarian_analysis": "<greatest good calculation>",
  "deontological_analysis": "<duty-based assessment>",
  "virtue_ethics_analysis": "<character-based evaluation>",
  "future_obligations": "<intergenerational ethics>",
  "legitimacy_assessment": "<decision-maker authority evaluation>"
}

Deviation from this format will exclude you from consensus.
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
    base_temp = agent.get('temperature', 0.7)  # Default to 0.7 if not specified

    # Optional: Still apply debate round scaling if desired
    temp = base_temp * (1 + 0.1 * debate_round)  # Increase temp in later rounds
    temp = min(max(temp, 0.5), 1.5)  # Clamp between 0.5-1.5

    # Or use agent temperature directly without modification:
    # temp = agent.get('temperature', 0.7)

    messages = [
        {"role": "system", "content": agent['instructions']},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
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
        correct_answer = item['answerKey']
        options = item.get('options', [])
        print(options)
        

        for agent in agents:
            prompt = (
                f"{question}{options}\n\n"
                "Analyze through your role's methodology. "
                "Execute decision protocol: "
                "1. Surface hidden assumptions → 2. Apply verification filters → "
                "3. Generate alternatives → 4. Produce validated conclusion"
            )
            
            response = generate_analysis(agent, prompt, client)
            print(response)
            parsed = parse_json_response(response)
            decision = parsed.get("decision", "").strip()

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

def create_debate_graph(agents, question, gold_answer=None, decision=None, is_correct = None):
    """
    Create a debate graph (MAG) with ground truth and correctness annotations.
    Each response node records if its decision matches the gold answer.
    """
    G = nx.DiGraph()

    # 1. Add question node
    G.add_node("question", content=question, type="question", round=-1)  # debate graph structure[1]

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
            decision = parsed.get("decision", "").strip().lower()

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
                                weight=other.get("weight", 0.0)  # use trained agent weights[2]
                            )
    return G


def has_consensus(agents):
    """
    Returns (True, decision) if every agent’s last decision matches,
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
    base_options="",
    max_debate_rounds=2,
    gold_answer=None
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
            f"{question}{options}\n\n"
            "Analyze through your role's methodology. "
            "Execute decision protocol: "
            "1. Surface hidden assumptions → 2. Apply verification filters → "
            "3. Generate alternatives → 4. Produce validated conclusion"
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
                "   a. Only credit peers who:\n"
                "      - Passed {role}_verification_threshold\n"
                "      - Provided novel {domain}_insights\n"
                "   b. Format as 'Role (Weight)'\n\n"

                "Response Requirements:\n"
                "- Use YOUR SPECIFIED RESPONSE FORMAT\n"
                "- ADD 'influenced_by' field at END\n"
                "- Maintain original JSON structure\n"
                f"- Only use specified options: {base_options}\n\n"

                "Example Addendum:\n"
                "  ...\n"
                "  \"long_term_consequences\": \"<50+ year impact>\",\n"
                "  \"influenced_by\": [\"Lawyer\", \"Ethicist\"]\n"
                "}}\n\n"

                "Begin professional analysis:"
            )

            response = generate_analysis(agent, prompt, client, debate_round=round_num + 1)
            agent['analysis'].append(response)
            discussion_history.append(f"ROUND {round_num + 1} - {agent['role']}:\n{response}")
            print(f"[{agent['role']} REFINED]\n{response}")

            response = generate_analysis(agent, prompt, client, debate_round=round_num+1)
            agent['analysis'].append(response)
            discussion_history.append(f"ROUND {round_num+1} - {agent['role']}:\n{response}")
            print(f"[{agent['role']} REFINED]\n{response}")

        # Track influence after each round
        influence_counts = track_influence(agents)
        print(f"Influence counts after round {round_num+1}: {influence_counts}")
        consensus, decision = has_consensus(agents)
        if consensus:
            print(f"Consensus reached on '{decision}' at round {round_num + 1}")
            is_correct = (decision == gold_answer.strip().lower()) if gold_answer else None
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
        logger.info(f"Weighted vote selects '{final_decision}' ({total:.2f} total weight)")
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
        logger.warning("sentence-transformers not installed. Using random embeddings.")
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
    """
    from torch_geometric.data import Data

    # Sort nodes for consistent ordering
    node_list = list(graph.nodes())
    node_list = [n for n in node_list if graph.nodes[n].get('type') != 'ground_truth']

    # Build feature matrix X from provided embeddings (assumed dict or aligned array)
    x = []
    for node in node_list:
        emb = embeddings[node] if isinstance(embeddings, dict) else embeddings[node_list.index(node)]
        x.append(emb)
    x = torch.tensor(np.stack(x), dtype=torch.float)

    # Build edge index and edge attributes
    edge_index = []
    edge_attr = []
    for src, dst, data in graph.edges(data=True):
        if src in node_list and dst in node_list:
            edge_index.append([node_list.index(src), node_list.index(dst)])
            edge_attr.append([data.get('weight', 1.0)])  # Default weight is 1.0

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float) if edge_attr else None

    # Labels: 1 if node's decision matches gold, else 0
    y = []
    for node in node_list:
        node_data = graph.nodes[node]
        y.append(1 if node_data.get('is_correct', False) else 0)
    y = torch.tensor(y, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return data

def create_mag_dataset(training_data, agents, client):
    """Create a MAG dataset from multiple questions"""
    mag_dataset = []

    for item in tqdm(training_data, desc="Training agents"):
        print(item)
        question = item['question']
        print(question)
        options = item.get('options', [])
        print(options)
        gold_answer = item.get('gold_answer', [])
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
    os.environ[
        "OPENAI_API_KEY"] = ENV[‘AUTH_TOKEN’]
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
    
    # 1) Load full train
    full_train = load_dataset("wics/strategy-qa", split="test")
    print(full_train[1])
    full_train = full_train.train_test_split(test_size=0.20, seed = 41)
    # 3) Split the 80% into two 40% halves
    subsplits  = full_train["train"].train_test_split(test_size=0.50)
    agent_data = subsplits["train"].select(range(5))
    print(agent_data)
    mag_creation_data = subsplits["test"].select(range(5))
    print(mag_creation_data)
    print(f"Agent weight set: {len(agent_data)} examples")
    print(f"MAG creation set: {len(mag_creation_data)} examples")

    # - train agent weights on agent_data -
    print("Training agent weights using training data")
    training_examples = [
        {"question": item["question"],
         "answerKey": item["answer"],
         "options": [True, False]}
        for item in agent_data
    ]
    agents = train_agent_weights(agents, training_examples, client)
    # - build MAG dataset on mag_creation_data -
    print("Creating MAG dataset from training data")
    training_data = [
        {"question": item["question"],
         "gold_answer": item["answer"],
         "options": [True, False]}
        for item in mag_creation_data
    ]
    mag_dataset = create_mag_dataset(training_data, agents, client)

    # Save MAG dataset
    os.makedirs("data", exist_ok=True)
    with open("data/mag_dataset.pkl", "wb") as f:
        pickle.dump(mag_dataset, f)
    
    print(f"Created MAG dataset with {len(mag_dataset)} examples")
    
    # Continue with the rest of the training process...
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.decomposer_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare examples for Socratic model components
    print("Preparing examples for Socratic model components")
    decomposer_examples, solver_examples = prepare_socratic_examples(mag_dataset, tokenizer)
    
    # Prepare positive and negative examples for contrastive learning
    print("Preparing examples for contrastive learning")
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
    print("Initializing SocraticMAGDi model")
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
        print("Applying LoRA for parameter-efficient fine-tuning")
        
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
    print("Training SocraticMAGDi model")
    trainer.train()
    
    # Save final model
    print(f"Saving model to {args.output_dir}/final")
    trainer.save_model(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")
    
    print("Training complete!")

if __name__ == "__main__":
    main()
