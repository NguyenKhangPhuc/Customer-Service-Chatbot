from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client
import os
from dotenv import load_dotenv 
load_dotenv()
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

print(f"{supabase_url} -- {supabase_key}")

supabase = create_client(supabase_url, supabase_key)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

loader = PyPDFLoader("KPAPP_Doc.pdf")
your_long_document = loader.load() 

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(your_long_document)

vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents"
)

vector_store.add_documents(chunks)
print(f"Đã nạp thành công {len(chunks)} đoạn văn bản vào Supabase!")