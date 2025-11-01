import os
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from llm import get_embeddings, get_llm, get_prompt_template

load_dotenv()

REQUIRED_VARS = ("PGVECTOR_URL", "PGVECTOR_COLLECTION")
for var in REQUIRED_VARS:
    if not os.getenv(var):
        raise RuntimeError(f"Variável de ambiente {var} não está definida. Verifique o arquivo .env")

PGVECTOR_URL = os.getenv("PGVECTOR_URL")
PGVECTOR_COLLECTION = os.getenv("PGVECTOR_COLLECTION")


def get_vector_store():
    """
    Inicializa e retorna a conexão com o banco de dados vetorial.
    """
    embeddings = get_embeddings()
    
    store = PGVector(
        embeddings=embeddings,
        collection_name=PGVECTOR_COLLECTION,
        connection=PGVECTOR_URL,
        use_jsonb=True,
    )
    
    return store


def format_docs(docs_with_scores):
    """
    Formata os documentos recuperados do banco vetorial para o contexto.
    """
    if not docs_with_scores:
        return ""
    
    formatted = []
    for doc, score in docs_with_scores:
        formatted.append(doc.page_content.strip())
    
    return "\n\n".join(formatted)


def search_with_score(question):
    """
    Busca documentos relevantes no banco vetorial.
    Retorna os top 10 resultados mais similares conforme especificação (k=10).
    
    Nota: Em alguns casos raros, informações relevantes podem ter scores ligeiramente
    piores e não aparecer nos top 10 devido à natureza do embedding vetorial.
    """
    store = get_vector_store()
    results = store.similarity_search_with_score(question, k=10)
    return results


def search_prompt(question=None):
    """
    Cria e retorna a chain de busca e resposta usando RAG.
    
    Se uma pergunta for fornecida, executa a busca e retorna a resposta.
    Caso contrário, retorna a chain para uso posterior.
    """
    try:
        # Inicializa o LLM do Gemini
        llm = get_llm()
        
        # Cria o template de prompt
        prompt = PromptTemplate.from_template(get_prompt_template())
        
        # Cria a chain RAG
        def retrieve_and_format(inputs):
            question = inputs["pergunta"]
            docs_with_scores = search_with_score(question)
            
            contexto = format_docs(docs_with_scores)
            
            return {"contexto": contexto, "pergunta": question}
        
        chain = (
            RunnablePassthrough.assign(pergunta=lambda x: x["pergunta"])
            | retrieve_and_format
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Se uma pergunta foi fornecida, executa a chain
        if question:
            response = chain.invoke({"pergunta": question})
            return response
        
        # Caso contrário, retorna a chain para uso posterior
        return chain
        
    except Exception as e:
        print(f"Erro ao inicializar a busca: {e}")
        return None