# 生成知识
from langchain.llms import OpenAI

llm = OpenAI()

# Function to generate knowledge
def generate_knowledge(question):
    prompt = f"Generate knowledge statements for the question: {question}"
    knowledge = llm(prompt)
    return knowledge

# Function to answer the question using the generated knowledge
def answer_question_with_knowledge(question, knowledge):
    prompt = f"Question: {question}\nKnowledge: {knowledge}\nAnswer:"
    answer = llm(prompt)
    return answer

# Example usage
question = "Why do leaves change color in the fall?"

# Step 1: Generate Knowledge
knowledge = generate_knowledge(question)
print(f"Generated Knowledge: {knowledge}")

# Step 2: Answer the Question using the generated knowledge
answer = answer_question_with_knowledge(question, knowledge)
print(f"Answer: {answer}")
