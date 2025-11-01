# Desafio MBA Engenharia de Software com IA - Full Cycle

## Descrição

Sistema de ingestão e busca semântica em documentos PDF utilizando LangChain, OpenAI e PostgreSQL com extensão pgVector. O sistema permite realizar perguntas sobre o conteúdo de um PDF através de um chat interativo via CLI, utilizando técnicas de RAG (Retrieval-Augmented Generation).

## Tecnologias Utilizadas

- **Python 3.10+**
- **LangChain**: Framework para aplicações com LLMs
- **OpenAI**: LLM (GPT-4o-mini) e embeddings (text-embedding-3-small)
- **PostgreSQL + pgVector**: Banco de dados vetorial
- **Docker & Docker Compose**: Para containerização do banco de dados

## Pré-requisitos

- Python 3.10 ou superior
- Docker e Docker Compose instalados
- Conta OpenAI com API Key ativa

## Instalação e Configuração

### 1. Clone o repositório

```bash
git clone <url-do-repositorio>
cd mba-ia-desafio-ingestao-busca
```

### 2. Crie e ative o ambiente virtual

```bash
python3 -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Configure as variáveis de ambiente

Copie o arquivo `.env.example` para `.env` e configure suas credenciais:

```bash
cp .env.example .env
```

Edite o arquivo `.env` e adicione sua API Key da OpenAI:

```env
OPENAI_API_KEY=sua_api_key_aqui
PGVECTOR_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
PGVECTOR_COLLECTION=pdf_documents
PDF_PATH=document.pdf
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_LLM_MODEL=gpt-4o-mini
```

**Como obter a API Key da OpenAI:**

1. Acesse [OpenAI Platform](https://platform.openai.com/api-keys)
2. Faça login com sua conta OpenAI
3. Clique em "Create new secret key"
4. Copie a chave gerada e cole no arquivo `.env`

### 5. Adicione seu documento PDF

Coloque o arquivo PDF que deseja processar na raiz do projeto com o nome `document.pdf`, ou ajuste a variável `PDF_PATH` no arquivo `.env` para apontar para o caminho do seu PDF.

## Como Executar

### Passo 1: Subir o banco de dados PostgreSQL

```bash
docker compose up -d
```

Este comando irá:

- Baixar a imagem do PostgreSQL com extensão pgVector
- Criar um container com o banco de dados
- Instalar a extensão pgVector automaticamente

Verifique se o banco está rodando:

```bash
docker compose ps
```

### Passo 2: Executar a ingestão do PDF

```bash
python src/ingest.py
```

Este script irá:

1. Carregar o arquivo PDF especificado em `PDF_PATH`
2. Dividir o documento em chunks de 1000 caracteres com overlap de 150
3. Gerar embeddings para cada chunk usando OpenAI
4. Salvar os vetores no banco de dados PostgreSQL

**Saída esperada:**

```
Iniciando ingestão do PDF: document.pdf
Carregando PDF...
PDF carregado com sucesso! Total de páginas: X
Dividindo documento em chunks...
Documento dividido em X chunks
Inicializando embeddings da OpenAI (modelo: text-embedding-3-small)...
Conectando ao banco de dados vetorial...
Salvando documentos no banco de dados vetorial...

[OK] Ingestão concluída com sucesso!
   - X chunks foram salvos no banco de dados
   - Collection: pdf_documents
```

### Passo 3: Iniciar o chat interativo

```bash
python src/chat.py
```

Agora você pode fazer perguntas sobre o conteúdo do PDF!

**Exemplo de uso:**

```
============================================================
Chat com IA - Sistema de Busca Semântica em PDF
============================================================

Digite sua pergunta ou 'sair' para encerrar.

------------------------------------------------------------

Faça sua pergunta: Qual o faturamento da Empresa SuperTechIABrazil?

Buscando informações...

RESPOSTA:
------------------------------------------------------------
O faturamento da Empresa SuperTechIABrazil é R$ 10.000.000,00.

------------------------------------------------------------

Faça sua pergunta: Quantos clientes temos em 2024?

Buscando informações...

RESPOSTA:
------------------------------------------------------------
Não tenho informações necessárias para responder sua pergunta.
```

Para sair do chat, digite: `sair`, `exit`, `quit` ou `q`

## Estrutura do Projeto

```
mba-ia-desafio-ingestao-busca/
├── docker-compose.yml          # Configuração do PostgreSQL + pgVector
├── requirements.txt            # Dependências Python
├── .env.example                # Template de variáveis de ambiente
├── .env                        # Variáveis de ambiente (não versionado)
├── document.pdf                # PDF para ingestão
├── README.md                   # Este arquivo
└── src/
    ├── llm.py                 # Módulo de configuração LLM/Embeddings
    ├── ingest.py              # Script de ingestão do PDF
    ├── search.py              # Lógica de busca vetorial e RAG
    └── chat.py                # CLI interativo
```

## Como Funciona

### Ingestão (ingest.py)

1. Carrega o PDF usando `PyPDFLoader`
2. Divide o texto em chunks usando `RecursiveCharacterTextSplitter`
3. Gera embeddings para cada chunk usando OpenAI
4. Armazena os vetores no PostgreSQL com pgVector

### Busca e Resposta (search.py + chat.py)

1. Usuário faz uma pergunta via CLI
2. A pergunta é convertida em embedding
3. Busca os 10 chunks mais similares no banco vetorial (k=10)
4. Monta um prompt com o contexto recuperado
5. Envia para o LLM OpenAI responder
6. Retorna a resposta baseada apenas no contexto do PDF

## Solução de Problemas

### Erro ao conectar no banco de dados

Verifique se o Docker está rodando e o container está up:

```bash
docker compose ps
docker compose logs postgres
```

### Erro de API Key inválida

Confirme que:

1. A variável `OPENAI_API_KEY` está definida no arquivo `.env`
2. A API Key está correta e ativa
3. Você tem créditos disponíveis na sua conta OpenAI

### Erro ao carregar o PDF

Verifique se:

1. O arquivo PDF existe no caminho especificado
2. O PDF não está corrompido
3. Você tem permissão de leitura no arquivo

### Chat não encontra informações

Possíveis causas:

1. A ingestão não foi executada ou falhou
2. A pergunta está usando termos muito diferentes do conteúdo do PDF
3. O banco de dados foi reiniciado (execute novamente a ingestão)

## Limpeza

Para parar e remover o banco de dados:

```bash
docker compose down -v
```

Para desativar o ambiente virtual:

```bash
deactivate
```

## Notas Técnicas

- **Chunk size**: 1000 caracteres com overlap de 150
- **Embeddings**: `text-embedding-3-small` (OpenAI)
- **LLM**: `gpt-4o-mini` (OpenAI)
- **Busca**: Top 10 resultados mais similares (k=10)
- **Temperatura do LLM**: 0 (respostas mais determinísticas)

## Autor

Fernando Chaves - MBA Engenharia de Software com IA - Full Cycle

## Licença

Este projeto é parte de um desafio acadêmico.
