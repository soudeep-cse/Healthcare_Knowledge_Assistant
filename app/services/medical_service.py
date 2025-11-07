import os
from dotenv import load_dotenv
from langdetect import detect
from PIL import Image
import pytesseract
from googletrans import Translator

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

class MedicalChatbotService:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
        self.translator = Translator()
        self.vectorstore = None
        self._load_vectorstore()

    def _load_vectorstore(self):
        try:
            self.vectorstore = FAISS.load_local("faiss_index", self.embeddings, allow_dangerous_deserialization=True)
        except:
            self.vectorstore = None

    def extract_text(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return "\n".join([p.extract_text() or "" for p in PyPDFLoader(file_path).pages])
        elif ext in [".png", ".jpg", ".jpeg"]:
            return pytesseract.image_to_string(Image.open(file_path), lang="eng+jpn")
        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    def ingest_documents(self, file_paths):
        all_docs = []
        for path in file_paths:
            text = self.extract_text(path)
            if not text.strip():
                continue
            lang = "ja" if detect(text) == "ja" else "en"
            chunks = self.splitter.split_text(text)
            all_docs.extend([Document(page_content=chunk, metadata={"lang": lang, "source": path}) for chunk in chunks])
        
        if all_docs:
            if self.vectorstore is None:
                self.vectorstore = FAISS.from_documents(all_docs, self.embeddings)
            else:
                new_vectorstore = FAISS.from_documents(all_docs, self.embeddings)
                self.vectorstore.merge_from(new_vectorstore)
            self.vectorstore.save_local("faiss_index")
        
        return len(all_docs)

    def retrieve_documents(self, query, k=3):
        if self.vectorstore is None:
            return []
        
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        distances = [score for _, score in results]
        max_dist = max(distances) if distances else 1
        
        return [{
            "content": doc.page_content, 
            "distance": float(score),
            "similarity_percent": round((1 - score/max_dist) * 100, 2),
            "metadata": doc.metadata
        } for doc, score in results]

    def generate_answer(self, query, k=3, output_language=None):
        query_lang = "ja" if detect(query) == "ja" else "en"
        
        if self.vectorstore is None:
            return "No documents available for answering queries."
        
        retrieved_docs = self.vectorstore.similarity_search(query, k=k)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        target_lang = output_language or query_lang
        
        system = (
            "You are a medical assistant providing accurate English answers."
            if target_lang == "en"
            else "あなたは正確な日本語の医療アシスタントです。"
        )
        instruction = (
            "Provide a concise, professional English answer based on context."
            if target_lang == "en"
            else "コンテキストに基づいて簡潔で専門的な日本語の回答をしてください。"
        )

        prompt = PromptTemplate(
            input_variables=["system", "context", "query", "instruction"],
            template="{system}\n\nContext:\n{context}\n\nQuestion: {query}\n{instruction}",
        )

        chain = prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({
            "system": system,
            "context": context,
            "query": query,
            "instruction": instruction
        })

        # Translate if needed
        if output_language and output_language != query_lang:
            try:
                response = self.translator.translate(response, dest=output_language).text
            except:
                pass

        return response