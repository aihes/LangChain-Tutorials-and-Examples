# 自我一致性
from langchain.llms import OpenAI
import numpy as np


def self_consistency_decoding(prompt, n_samples=5):
    """
    Self-consistency decoding strategy with marginalization.

    Parameters:
    - prompt: str, the input prompt for the language model
    - n_samples: int, the number of reasoning paths to sample

    Returns:
    - str, the most consistent answer among the sampled reasoning paths
    """

    outputs = []

    # Step 1: Sample a diverse set of reasoning paths
    for i in range(n_samples):
        # Vary temperature and top_p for diversity in reasoning paths
        temperature = np.random.uniform(0.5, 1.0)
        top_p = np.random.uniform(0.8, 1.0)

        # Initialize the OpenAI model with custom parameters
        llm = OpenAI(
            model_name="text-davinci-003",
            temperature=temperature,
            max_tokens=100,
            top_p=top_p
        )

        # Sample a reasoning path
        output = llm(prompt)
        outputs.append(output)

    # Step 2: Marginalize out reasoning paths to aggregate final answers
    # Create a new prompt to let the model choose the most consistent answer
    new_prompt = f"Based on the following answers, what is the most consistent and likely correct answer?\n{outputs}"

    print(new_prompt)
    # Use the model to choose the most consistent answer
    llm = OpenAI(model_name="text-davinci-003", temperature=0.2, max_tokens=50)
    most_consistent_answer = llm(new_prompt)

    return most_consistent_answer


# Test the self_consistency_decoding function
prompt = "一口井7米深，有只蜗牛从井底往上爬，白天爬3米，晚上往下坠2米。问蜗牛几天能从井里爬出来？Let's work this out in a step by step way to be sure we have the right answer."
most_consistent_answer = self_consistency_decoding(prompt)
print(f"The most consistent answer is: {most_consistent_answer}")
