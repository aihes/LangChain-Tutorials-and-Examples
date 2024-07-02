from langchain.llms import QianfanLLMEndpoint
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import time

questions = [
    "孕妇打人，算群殴吗? ",
    # "杀人不眨眼的人，眼睛不会干吗?",
    # "老鼠生病了，吃老鼠药能好吗? ",
    # "蓝牙耳机坏了，我应该去医院的牙科还是耳科? ",
    # "不孕不育会遗传吗? ",
    # "刘备温酒斩曹操发生在什么时候? ",
    # "两个男人正常交谈，其中一个男人夸赞对方办事能力强，对方回答 \"哪里，哪里\"。这里的\"哪里，哪里\"是什么意思?  ",
    # "不孕不育会遗传吗? ",
    # "刘备温酒斩曹操发生在什么时候? ",
    # "两个男人正常交谈，其中一个男人夸赞对方办事能力强，对方回答 \"哪里，哪里\"。这里的\"哪里，哪里\"是什么意思?  "
]

models = [
    {"name": "ERNIE-4.0-Turbo-8K", "model": "ERNIE-4.0-Turbo-8K", "endpoint": "eb-instant"},
    {"name": "ERNIE-4.0-8K", "model": "ERNIE-4.0-8K", "endpoint": "eb-instant", "max_tokens": 512},
    {"name": "ERNIE-3.5-8K", "model": "ERNIE-3.5-8K", "endpoint": "eb-instant"}
]

for model_info in models:
    print(f"Testing model: {model_info['name']}")
    total_time = 0
    total_length = 0

    llm = QianfanLLMEndpoint(
        streaming=True,
        model=model_info["model"],
        endpoint=model_info["endpoint"],
        max_tokens=model_info.get("max_tokens", 2048)
    )

    callback_handler = StreamingStdOutCallbackHandler()

    for question in questions:
        print(f"\nQuestion: {question}")

        start_time = time.time()
        response = llm.generate([question], callbacks=[callback_handler])
        end_time = time.time()

        response_text = response.generations[0][0].text
        response_length = len(response_text)

        print(f"\nFull response: {response_text}")
        print(f"Response length: {response_length}")
        print(f"Time taken: {end_time - start_time:.2f} seconds\n")

        total_time += end_time - start_time
        total_length += response_length

    print(f"Total time for model {model_info['name']}: {total_time:.2f} seconds")
    print(f"Total response length for model {model_info['name']}: {total_length}")

    if total_time > 0:
        avg_chars_per_second = total_length / total_time
        print(f"Average characters per second for model {model_info['name']}: {avg_chars_per_second:.2f}\n")
    else:
        print(f"Average characters per second for model {model_info['name']}: N/A\n")