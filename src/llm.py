import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()

REQUIRED_VARS = ("OPENAI_API_KEY",)
for var in REQUIRED_VARS:
    if not os.getenv(var):
        raise RuntimeError(f"Variável de ambiente {var} não está definida. Verifique o arquivo .env")

OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")

PROMPT_TEMPLATE = """CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def get_embeddings():
    """
    Retorna a instância configurada de embeddings da OpenAI.
    
    Returns:
        OpenAIEmbeddings: Instância de embeddings configurada
    """
    return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)


def get_llm(temperature=0, max_retries=2):
    """
    Retorna a instância configurada do LLM OpenAI.
    
    Args:
        temperature (float): Controla a aleatoriedade das respostas (0 = determinístico)
        max_retries (int): Número máximo de tentativas em caso de erro
    
    Returns:
        ChatOpenAI: Instância do LLM configurada
    """
    return ChatOpenAI(
        model=OPENAI_LLM_MODEL,
        temperature=temperature,
        max_retries=max_retries,
    )


def get_prompt_template():
    """
    Retorna o template de prompt configurado.
    
    Returns:
        str: Template de prompt
    """
    return PROMPT_TEMPLATE

