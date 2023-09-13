# 程序辅助编程
from langchain.llms import OpenAI

# 初始化LLM
llm = OpenAI()

# 示例：用户输入的自然语言问题
user_input = "一口井7米深，有只蜗牛从井底往上爬，白天爬3米，晚上往下坠2米。问蜗牛几天能从井里爬出来？"

# 使用LLM将问题转换为Python代码
code_output = llm(f"""
I want you to act like a Python interpreter. 
Answer any questions using Python code. 
return Python code directly.

The example：
Question: 某人花19快钱买了个玩具，20快钱卖出去。他觉得不划算，又花21快钱买进，22快钱卖出去。请问它赚了多少钱？
Answer: 
```
# 购买和出售的价格
buy1, sell1 = 19, 20
buy2, sell2 = 21, 22

# 计算总收入
total_profit = (sell1 - buy1) + (sell2 - buy2)

# 输出结果
result = total_profit
```
 
Question: {user_input}
Answer: 
""")
code_output = code_output.replace("```", "")
print(f"Generated Code: {code_output}")

# 执行生成的代码
try:
    # 创建一个字典来存储局部变量和全局变量
    local_vars = {}
    global_vars = {}

    # 动态地导入所有必要的Python库
    import_statements = "import numpy\nimport math"
    exec(import_statements, global_vars, local_vars)

    # 使用exec()执行代码
    exec(code_output, global_vars, local_vars)

    # 获取结果（假设结果存储在名为'result'的变量中）
    result = local_vars.get('result', 'No result variable found.')
    print(f"Result: {result}")

except Exception as e:
    print(f"An error occurred: {e}")
