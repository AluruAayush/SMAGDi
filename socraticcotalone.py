import openai
import random

openai.api_key = ENV[‘AUTH_TOKEN’]

class Agent:
    def __init__(self, name, persona):
        self.name = name
        self.persona = persona

    def respond(self, input_text):
        system_prompt = (
            "You are a Socratic AI assistant that reasons by asking questions to reach deeper understanding. "
            "You follow Socratic Chain-of-Thought by examining assumptions, alternatives, and implications before answering."
        )
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

class Moderator:
    def __init__(self, agents):
        self.agents = agents

    def moderate(self, input_text):
        results = []
        for agent in self.agents:
            output = agent.respond(input_text)
            results.append((agent.name, output))
        combined = "\n".join(f"{name}: {out}" for name, out in results)
        consensus_prompt = "You are a moderator combining multiple Socratic opinions. Decide the most accurate final answer."
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": consensus_prompt},
                    {"role": "user", "content": combined}
                ],
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[ERROR: {e}]"

def split_dataset(data):
    random.shuffle(data)
    n = len(data)
    train = data[:int(0.6 * n)]
    val = data[int(0.6 * n):int(0.7 * n)]
    test = data[int(0.7 * n):]
    return train, val, test

def run():
    agents = [
        Agent("Agent1", "Philosopher trained in ethics and logic."),
        Agent("Agent2", "Engineer focused on causality and feasibility."),
        Agent("Agent3", "Historian contextualizing all answers."),
        Agent("Agent4", "Lawyer concerned with legality and precedent."),
        Agent("Agent5", "Economist modeling consequences."),
        Agent("Agent6", "Psychologist modeling human reactions."),
        Agent("Agent7", "Biologist focused on natural systems."),
        Agent("Agent8", "Technologist trained in systems design.")
    ]

    moderator = Moderator(agents)

    dataset = [
        "Should we deploy AI agents in hospitals without human supervision?",
        "Is universal basic income sustainable in an AI-driven economy?",
        "Can machine learning replace criminal court judges ethically?",
        "Do autonomous drones raise unique legal issues in warfare?",
        "Is AGI development a moral obligation or a threat?"
    ]

    _, _, test = split_dataset(dataset)

    for question in test:
        print(f"\nQUESTION: {question}")
        final_output = moderator.moderate(question)
        print(f"MODERATOR ANSWER:\n{final_output}")

run()
