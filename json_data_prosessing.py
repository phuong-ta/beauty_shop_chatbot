from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS

# Recursively split json data - If you need to access/manipulate the smaller json chunks
vector_db_path="data/vector_data"
file_path="data/data.json"

hg_api_key = "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"
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=hg_api_key, model_name="sentence-transformers/all-MiniLM-l6-v2"
)

def create_hg_embedding_model():
    hg_api_key = "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"
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=hg_api_key, 
        model_name="sentence-transformers/all-MiniLM-l6-v2"
    )
    return embeddings

def setup_data(filepath, vector_db_path):
    loader = JSONLoader(file_path=file_path, jq_schema=".[]", text_content=False)
    documents = loader.load()
    db = FAISS.from_documents(documents,embedding = create_hg_embedding_model())
    db.save_local(vector_db_path)
    return db


#setup_data(filepath=file_path, vector_db_path=vector_db_path)

# Read tu VectorDB
def read_vectors_db():
    db = FAISS.load_local(vector_db_path, 
                          embeddings = create_hg_embedding_model(),
                          allow_dangerous_deserialization=True)
    return db


db= read_vectors_db()
query = "show me two vegan products?"
docs = db.similarity_search(query, k=5)
print(docs[0])
