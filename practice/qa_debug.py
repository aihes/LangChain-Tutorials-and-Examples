# # 加载文档
# from langchain.document_loaders import WebBaseLoader
#
# loader = WebBaseLoader("https://www.marxists.org/chinese/maozedong/marxist.org-chinese-mao-193707.htm")
# data = loader.load()
#
# # Split
# from langchain.text_splitter import RecursiveCharacterTextSplitter
#
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
# all_splits = text_splitter.split_documents(data)
#
# # Store
# from langchain.vectorstores import Chroma
# from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
# from langchain.vectorstores.redis import Redis
#
# vectorstore = Chroma.from_documents(documents=all_splits, embedding=HuggingFaceEmbeddings())
# # rds = Redis.from_documents(
# #     all_splits, embeddings, redis_url="redis://localhost:6379", index_name="link"
# # )
#
# import requests
# from langchain import OpenAI
# from pydantic import BaseModel, Field
# from langchain.agents import AgentType
# from langchain.chat_models import ChatOpenAI
# from langchain.agents import initialize_agent
# from langchain.tools import StructuredTool
#
# def create_crowd(type: str, param: dict) -> str:
#     """
#      该工具可以用来进行人群生成：
#     当需要生成人群、分析画像、咨询问题时，使用如下的指示：url 固定为：http://localhost:3001/
#      如果请求是生成人群，请求的type为crowd; 如果请求是分析画像，请求的type为analyze; 如果是其他或者答疑，请求的type为question;
#      请求body的param把用户指定的条件传进来即可
#      只要请求有结果，你就说人群正在生成中就行
#      """
#     result = requests.post("http://localhost:3001/", json={"type": type, "param": param})
#     print(result)
#     return f"Status: {result.status_code} - {result.text}"
#
# tools = [
#     StructuredTool.from_function(func=create_crowd, return_direct=True)
# ]
#
# llm = OpenAI(temperature=0)  # Also works well with Anthropic models
# # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# agent_chain = initialize_agent(tools,
#                                llm,
#                                agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
#                                verbose=True,
#                                # memory=memory
#                                )
# agent_chain.run("我想生成一个性别为男并且在180天访问过淘特的人群?")


import re
from typing import Union

class AgentOutputParser:
    """A generic agent output parser."""
    pass

class AgentAction:
    """A generic agent action."""
    def __init__(self, action, action_input, text):
        self.action = action
        self.action_input = action_input
        self.text = text

class AgentFinish(AgentAction):
    """A generic agent finish action."""
    pass

class OutputParserException(Exception):
    """An exception thrown when the output parser fails."""
    pass

# 这里是原来的 ReActOutputParser 类

class ReActOutputParser(AgentOutputParser):
    """Output parser for the ReAct agent."""

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        action_prefix = "Action: "
        if not text.strip().split("\n")[-1].startswith(action_prefix):
            raise OutputParserException(f"Could not parse LLM Output: {text}")
        action_block = text.strip().split("\n")[-1]

        action_str = action_block[len(action_prefix) :]
        # Parse out the action and the directive.
        re_matches = re.search(r"(.*?)\[(.*?)\]", action_str)
        if re_matches is None:
            raise OutputParserException(
                f"Could not parse action directive: {action_str}"
            )
        action, action_input = re_matches.group(1), re_matches.group(2)
        if action == "Finish":
            return AgentFinish(action, action_input, text)
        else:
            return AgentAction(action, action_input, text)

    @property
    def _type(self) -> str:
        return "react"

# 测试代码

parser = ReActOutputParser()

# 这是一个有效的代理输出
valid_output = "Some text\nAction: Move[Forward]"
try:
    action = parser.parse(valid_output)
    print(f"Action: {action.action}, Input: {action.action_input}")
except OutputParserException as e:
    print(str(e))

# 这是一个无效的代理输出
invalid_output = "Some text\nInvalid action"
try:
    action = parser.parse(invalid_output)
    print(f"Action: {action.action}, Input: {action.action_input}")
except OutputParserException as e:
    print(str(e))
