# 自动提示词生成
from langchain.llms import OpenAI

# 初始化大型语言模型（LLM）
llm = OpenAI()

# 第一步：生成指令候选
def generate_instruction_candidates(original_prompt, num_candidates=5):
    candidates = []
    for _ in range(num_candidates):
        # 使用LLM生成与原始提示含义相同的新提示
        new_prompt = llm(f"Generate a prompt similar to: {original_prompt}")
        candidates.append(new_prompt)
    return candidates

# 第二步：评估指令
def evaluate_instructions(original_prompt, candidates):
    scores = []
    for candidate in candidates:
        # 使用LLM和一个评估Prompt来评分
        # evaluation_prompt = f"How good is the following prompt for a task? {candidate}"
        # output1 = llm(evaluation_prompt)
        # score1 = float(output1)  # 在实际应用中，这应该是模型的输出

        # 使用第二个模型（也是LLM）进行评分
        evaluation_prompt = f"""
        Task：评估我提供的Prompt的好坏；
        
        Context：
        我正在实验能否基于给出的Prompt，自动的生产一个更好的Prompt表达。但是需要量化Prompt的情况，比如Prompt的准确性，清晰性等进行综合评分。
        我会给出原始的Prompt和生产的Prompt，你给出我新Prompt的评分就行。评分在-1到1之间；

        Example:
        Input:
        原始Prompt：let us think step by step
        生成的Prompt：Think carefully and logically, explaining your answer.
        Output: 最终得分：0.8
        
        Input:
        原始Prompt：let us think step by step
        生成的Prompt：Let's think about this step by step.
        Output: 最终得分：0.7
                      

        Input: 
        原始Prompt：{original_prompt}
        生成的Prompt：{candidate}
        Output: 最终得分：
        """
        output2 = llm(evaluation_prompt)
        print(candidate, output2)

        scores.append((candidate, output2))

    return sorted(scores, key=lambda x: x[1], reverse=True)


# 第三步：重新抽样以生成指令的变体（这里简化为选择最高分的指令）
def resample_instructions(sorted_candidates):
    return sorted_candidates[0][0]


# 示例
original_prompt = "How to solve a quadratic equation?"

candidates = generate_instruction_candidates(original_prompt)
sorted_candidates = evaluate_instructions(original_prompt, candidates)
best_instruction = resample_instructions(sorted_candidates)

print(f"Best instruction for solving a quadratic equation is: {best_instruction}")
