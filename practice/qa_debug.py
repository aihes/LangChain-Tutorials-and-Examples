# 加载文档
from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://www.marxists.org/chinese/maozedong/marxist.org-chinese-mao-193707.htm")
data = loader.load()

# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Store
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores.redis import Redis

vectorstore = Chroma.from_documents(documents=all_splits, embedding=HuggingFaceEmbeddings())
# rds = Redis.from_documents(
#     all_splits, embeddings, redis_url="redis://localhost:6379", index_name="link"
# )
