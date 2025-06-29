import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings



file_name = "download (1).pdf"
print(file_name)
if not os.path.exists(file_name):
    raise FileNotFoundError("file not found")


if file_name.endswith(".pdf"):
    loader = PyPDFLoader(file_name)
elif file_name.endswith(".txt"):
    loader = TextLoader(file_name)
else:
    raise ValueError("Only PDF or TXT file formats are supported in this script.")

documents = loader.load()
print(documents)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)


embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vector_store = FAISS.from_documents(docs, embedding_model)
print(embedding_model)
print(vector_store)


qa_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    tokenizer="google/flan-t5-base",
    device=-1  # forces CPU
)
print("Loaded LLM pipeline")
print(qa_pipeline)
llm = HuggingFacePipeline(pipeline=qa_pipeline)
print(llm)


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(),
    chain_type="stuff"
)
print(qa_chain)


print("\n our RAG is ready. Ask anything about the document!")
print("Type 'exit' to quit.")

while True:
    query = input("\nYou: ")
    if query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    try:
        answer = qa_chain.run(query)
        print(f"AI: {answer}")
    except Exception as e:
        print(f"Error: {str(e)}")
