from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

# 1. 加载PDF
loader = PyPDFLoader("docs/test.pdf")
documents = loader.load()

# 2. 文本切片
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

# 3. 向量化
embeddings = OllamaEmbeddings(model="qwen2.5:3b")

# 4. 创建向量库
vectorstore = FAISS.from_documents(docs, embeddings)

# 5. 创建模型
llm = ChatOllama(model="qwen2.5:3b")

print("===== 本地RAG问答系统 =====")

while True:
    question = input("\n请输入问题：")

    if question == "exit":
        break

    # 6. 检索相关内容
    related_docs = vectorstore.similarity_search(question, k=3)

    context = "\n".join([doc.page_content for doc in related_docs])

    # 7. 构建Prompt
    prompt = f"""
你是一个文档问答助手。

请根据以下内容回答问题：

{context}

问题：
{question}
"""

    # 8. 调用模型
    response = llm.invoke(prompt)

    print("\n回答：")
    print(response.content)