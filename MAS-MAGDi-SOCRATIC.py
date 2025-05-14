#Pip install all of the following libraries before running
import openai
import random
import networkx as nx
import os
import json

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = ENV[‘AUTH_TOKEN’]
openai.api_key = os.getenv("OPENAI_API_KEY")

# Role instructions
role_specifications = {
    "Scientist": {
        "instructions": """
You are a Scientist.
- Select one option and justify using scientific reasoning.
- Respond strictly in JSON with:
{
  "decision": "<selected_option>",
  "analysis": "<reasoning, implications, and confidence>"
}
"""
    },
    "Lawyer": {
        "instructions": """
You are a Lawyer.
- Select one option and justify using legal frameworks.
- Respond strictly in JSON with:
{
  "decision": "<selected_option>",
  "analysis": "<reasoning, implications, and confidence>"
}
"""
    },
    "Historian": {
        "instructions": """
You are a Historian.
- Select one option and justify using historical precedent.
- Respond strictly in JSON with:
{
  "decision": "<selected_option>",
  "analysis": "<reasoning, implications, and confidence>"
}
"""
    }
}

def extract_options(question_text):
    start = question_text.find("options are: ") + len("options are: ")
    end = question_text.find(".", start)
    options_part = question_text[start:end]
    options = [opt.strip().lower() for opt in options_part.split(", ")]
    options = [opt[3:] if opt.startswith("or ") else opt for opt in options]
    return options

class Agent:
    def __init__(self, name, persona, role=None):
        self.name = name
        self.persona = persona
        self.role = role

    def respond(self, input_text):
        if self.role in role_specifications:
            system_prompt = role_specifications[self.role]["instructions"]
        else:
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
        decisions = {}
        for a in self.agents:
            out = a.respond(input_text)
            try:
                parsed = json.loads(out)
                decision = parsed["decision"].lower()
                results.append((a.name, parsed["analysis"]))
                decisions[decision] = decisions.get(decision, 0) + 1
            except:
                results.append((a.name, out))
        final_decision = max(decisions.items(), key=lambda x: x[1])[0] if decisions else "undecided"
        return final_decision, results

class MAGDi:
    def __init__(self, agents, moderator):
        self.graph = nx.DiGraph()
        self.agents = agents
        self.moderator = moderator
        self.counter = 0
        self.log = []

    def infer(self, prompt):
        final_output, responses = self.moderator.moderate(prompt)
        for agent_name, resp in responses:
            self.graph.add_node(self.counter, agent=agent_name, response=resp)
            self.counter += 1
        self.graph.add_node(self.counter, agent="MODERATOR", response=final_output)
        self.counter += 1
        self.log.append((prompt, final_output, responses))
        return final_output, responses

# Main interactive entry point
def main():
    agents = [
        Agent("Philosopher", "Philosopher trained in ethics and logic."),
        Agent("Engineer", "Engineer focused on causality and feasibility."),
        Agent("Scientist", "Scientist reasoning through data.", role="Scientist"),
        Agent("Lawyer", "Lawyer focused on legal precedent.", role="Lawyer"),
        Agent("Historian", "Historian reasoning through past events.", role="Historian")
    ]
    mod = Moderator(agents)
    mag = MAGDi(agents, mod)

    print("Enter a complex moral, legal, or scientific question including options (e.g., 'options are: deploy, delay, cancel.')")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Your question: ").strip()
        if user_input.lower() == "exit":
            break
        if "options are:" not in user_input:
            print("Please include 'options are:' followed by a list of options.\n")
            continue

        final, responses = mag.infer(user_input)
        print(f"\nFINAL DECISION: {final.upper()}")
        print("\nAGENT RESPONSES:")
        for name, resp in responses:
            print(f"\n{name}:\n{resp}")
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
