import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_postgres import PGVector

from llm import get_embeddings, OPENAI_EMBEDDING_MODEL

load_dotenv()

REQUIRED_VARS = ("PGVECTOR_URL", "PGVECTOR_COLLECTION", "PDF_PATH")
for var in REQUIRED_VARS:
    if not os.getenv(var):
        raise RuntimeError(f"Variável de ambiente {var} não está definida. Verifique o arquivo .env")

PDF_PATH = os.getenv("PDF_PATH")
PGVECTOR_URL = os.getenv("PGVECTOR_URL")
PGVECTOR_COLLECTION = os.getenv("PGVECTOR_COLLECTION")


def ingest_pdf():
    """
    Realiza a ingestão do PDF no banco de dados vetorial.
    
    Etapas:
    1. Carrega o PDF
    2. Divide em chunks de 1000 caracteres com overlap de 150
    3. Cria embeddings usando Gemini
    4. Salva no PostgreSQL com pgVector
    """
    print(f"Iniciando ingestão do PDF: {PDF_PATH}")
    
    # Verifica se o PDF existe
    if not Path(PDF_PATH).exists():
        raise FileNotFoundError(f"Arquivo PDF não encontrado: {PDF_PATH}")
    
    # Carrega o PDF
    print("Carregando PDF...")
    docs = PyPDFLoader(PDF_PATH).load()
    print(f"PDF carregado com sucesso! Total de páginas: {len(docs)}")
    
    # Divide em chunks
    print("Dividindo documento em chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150,
        add_start_index=False
    )
    splits = splitter.split_documents(docs)
    
    if not splits:
        print("Nenhum chunk foi criado. Verifique o conteúdo do PDF.")
        raise SystemExit(0)
    
    print(f"Documento dividido em {len(splits)} chunks")
    
    # Enriquece os documentos removendo metadados vazios
    enriched = [
        Document(
            page_content=doc.page_content,
            metadata={k: v for k, v in doc.metadata.items() if v not in ("", None)}
        )
        for doc in splits
    ]
    
    # Cria IDs únicos para cada documento
    ids = [f"doc-{i}" for i in range(len(enriched))]
    
    # Inicializa embeddings da OpenAI
    print(f"Inicializando embeddings da OpenAI (modelo: {OPENAI_EMBEDDING_MODEL})...")
    embeddings = get_embeddings()
    
    # Conecta ao PGVector
    print(f"Conectando ao banco de dados vetorial...")
    store = PGVector(
        embeddings=embeddings,
        collection_name=PGVECTOR_COLLECTION,
        connection=PGVECTOR_URL,
        use_jsonb=True,
    )
    
    # Adiciona documentos ao banco
    print("Salvando documentos no banco de dados vetorial...")
    store.add_documents(documents=enriched, ids=ids)
    
    print(f"\n[OK] Ingestão concluída com sucesso!")
    print(f"   - {len(enriched)} chunks foram salvos no banco de dados")
    print(f"   - Collection: {PGVECTOR_COLLECTION}")


if __name__ == "__main__":
    try:
        ingest_pdf()
    except Exception as e:
        print(f"\n[ERRO] Erro durante a ingestão: {e}")
        raise