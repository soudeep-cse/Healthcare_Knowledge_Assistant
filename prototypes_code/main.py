import os
from dotenv import load_dotenv
from langdetect import detect
from PIL import Image
import pytesseract

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)

from langchain_core.documents import Document

# ================= HELPERS =================
def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return "\n".join([p.extract_text() or "" for p in PyPDFLoader(file_path).pages])
    elif ext in [".png", ".jpg", ".jpeg"]:
        return pytesseract.image_to_string(Image.open(file_path), lang="eng+jpn")
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def ingest_files(file_paths):
    all_docs = []
    for path in file_paths:
        text = extract_text(path)
        if not text.strip():
            continue
        lang = "ja" if detect(text) == "ja" else "en"
        chunks = splitter.split_text(text)
        # Use Document class instead of dict
        all_docs.extend([Document(page_content=chunk, metadata={"lang": lang, "source": path}) for chunk in chunks])
    return all_docs

file_paths = ["docs/diabetes.txt","docs/japanese.txt"]
docs = ingest_files(file_paths)

vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("faiss_index")

query = "What are the latest recommendations for Type 2 diabetes management?"
docs_found = vectorstore.similarity_search(query, k=1)
print(docs_found[0].page_content)

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

def answer_query(query, k=3):
    query_lang = "ja" if detect(query) == "ja" else "en"

    # Retrieve top-k relevant documents from FAISS
    retrieved_docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # System prompt / instructions
    system = (
        "You are a medical assistant providing accurate English answers."
        if query_lang == "en"
        else "あなたは正確な日本語の医療アシスタントです。"
    )
    instruction = (
        "Provide a concise, professional English answer based on context."
        if query_lang == "en"
        else "コンテキストに基づいて簡潔で専門的な日本語の回答をしてください。"
    )

    # Create prompt template
    prompt = PromptTemplate(
        input_variables=["system", "context", "query", "instruction"],
        template="{system}\n\nContext:\n{context}\n\nQuestion: {query}\n{instruction}",
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
    parser = StrOutputParser()

    # Chain: prompt → LLM → parser
    chain = prompt | llm | parser

    # Invoke chain with proper input dictionary
    response = chain.invoke({
        "system": system,
        "context": context,
        "query": query,
        "instruction": instruction
    })

    return response

# ================= EXAMPLE =================
result = answer_query("2型糖尿病の最新の治療ガイドラインは何ですか？")
print(result)
