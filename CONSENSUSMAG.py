#Include the SOCRATIC COT it hasnt been uploaded yet
import os
import json
import random
import torch
import networkx as nx
import openai

os.environ["OPENAI_API_KEY"] = "sk-your-api-key"
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Minimal role instructions (2 sentences each)
role_specifications = {
    "Decomposer": "Break down complex questions into clear, manageable subquestions. Focus on clarity and completeness.",
    "Answerer": "Provide accurate and concise answers to the subquestions. Use clear and relevant information only.",
    "Scientist": "Analyze using scientific principles and evidence. Keep answers fact-based and logical.",
    "Lawyer": "Consider legal and ethical implications. Provide arguments from a legal perspective.",
    "Historian": "Contextualize with historical facts and trends. Use examples from history to support analysis.",
    "Mathematician": "Apply mathematical reasoning and quantitative analysis. Be precise and exact in explanations.",
    "Economist": "Evaluate economic impacts and trade-offs. Use economic theory and data where relevant.",
    "Ethicist": "Assess moral and ethical considerations. Focus on values, rights, and justice."
}

class StudentAgent:
    def __init__(self, name, persona, role):
        self.name = name
        self.persona = persona
        self.role = role

    def respond(self, input_text):
        system_prompt = role_specifications.get(self.role, "")
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{self.persona}\n\nInput: {input_text}"}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[ERROR: {e}]"

class MAGDiFramework:
    def __init__(self, student_agent):
        self.graph = nx.DiGraph()
        self.agent = student_agent
        self.node_counter = 0
        self.log = []

    def add_node(self, agent_name, role, response):
        self.graph.add_node(self.node_counter, agent=agent_name, role=role, response=response)
        self.node_counter += 1
        return self.node_counter - 1

    def add_edge(self, parent_node, child_node):
        self.graph.add_edge(parent_node, child_node)

    def infer(self, prompt):
        decomposer_input = prompt
        decomposer_response = None
        if self.agent.role == "Decomposer":
            decomposer_response = self.agent.respond(decomposer_input)
        try:
            decomposer_json = json.loads(decomposer_response)
            subquestions = decomposer_json.get("subquestions", [])
        except Exception:
            subquestions = []
        root_node = self.add_node(self.agent.name, "Decomposer", decomposer_response)
        answers = []
        for sq in subquestions:
            original_role = self.agent.role
            self.agent.role = "Answerer"
            answer_response = self.agent.respond(sq)
            self.agent.role = original_role
            answer_node = self.add_node(self.agent.name, "Answerer", answer_response)
            self.add_edge(root_node, answer_node)
            answers.append((sq, answer_response))
        self.log.append((prompt, decomposer_response, answers))
        return decomposer_response, answers

agents = [
    {
        "id": i,
        "role": role,
        "instructions": role_specifications.get(role, ""),
        "analysis": [],
        "previous_positions": [],
        "base_temp": 0.8 + random.uniform(-0.2, 0.2)
    } for i, role in enumerate(["Scientist", "Lawyer", "Historian", "Mathematician", "Economist", "Ethicist"])
]

def generate_analysis(agent, prompt, debate_round=0):
    temp = agent['base_temp'] * (1 + 0.1 * debate_round)
    temp = min(max(temp, 0.5), 1.5)
    messages = [
        {"role": "system", "content": agent['instructions']},
        {"role": "user", "content": prompt}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=temp,
            max_tokens=400
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR: {e}]"

def check_consensus(agent_analyses):
    decisions = []
    for a in agent_analyses:
        try:
            decision = json.loads(a['analysis'][-1])['decision']
        except Exception:
            decision = None
        if decision:
            decisions.append(decision)
    if len(decisions) == 0:
        return False, None
    counts = {}
    for d in decisions:
        counts[d] = counts.get(d, 0) + 1
    max_decision = max(counts, key=counts.get)
    if counts[max_decision] > len(decisions) / 2:
        return True, max_decision
    return False, None

def run_full_debate(question, options):
    prompt = f"""You must choose one of the following options exactly:
Options: {json.dumps(options)}

Question: {question}

Please respond strictly in JSON format with keys "decision" (selected option) and "analysis" (detailed reasoning).
"""
    for agent in agents:
        agent['analysis'] = []

    consensus_reached = False
    consensus_decision = None
    round_num = 0
    max_rounds = 5

    while not consensus_reached and round_num < max_rounds:
        for agent in agents:
            analysis = generate_analysis(agent, prompt, debate_round=round_num)
            agent['analysis'].append(analysis)

        consensus_reached, consensus_decision = check_consensus(agents)
        if consensus_reached:
            break
        round_num += 1

    return consensus_reached, consensus_decision, agents

def main():
    question = "Should humanity prioritize space exploration over addressing climate change?"
    options = ["Prioritize space exploration", "Focus on addressing climate change", "Balance both equally", "Defer decision for more research"]

    decomposer_agent = StudentAgent(name="DecomposerAgent", persona="Decomposer Agent", role="Decomposer")
    magdi = MAGDiFramework(decomposer_agent)
    dec_resp, answers = magdi.infer(question)

    print("Decomposer response:")
    print(dec_resp)
    print("\nAnswers to subquestions:")
    for sq, ans in answers:
        print(f"Q: {sq}")
        print(f"A: {ans}\n")

    print("Starting multi-agent debate for final decision...")
    consensus, decision, detailed_agents = run_full_debate(question, options)
    print(f"Consensus reached: {consensus}")
    if consensus:
        print(f"Consensus decision: {decision}")
    else:
        print("No consensus reached after max rounds.")
    print("\nDetailed agent analyses:")
    for agent in detailed_agents:
        print(f"{agent['role']} analysis history:")
        for analysis in agent['analysis']:
            print(analysis)
            print("----")

if __name__ == "__main__":
    main()
