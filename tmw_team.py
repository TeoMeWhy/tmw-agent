import os
import dotenv
dotenv.load_dotenv()

from agno.agent import Agent
from agno.team import Team, TeamMode

from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.hackernews import HackerNewsTools
from agno.tools.tavily import TavilyTools

from agno.os import AgentOS
from agno.models.openai.like import OpenAILike
from agno.db.sqlite import SqliteDb

import qdrant_client
from qdrant_client import models
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding

DENSE_MODEL = os.getenv("DENSE_MODEL")
SPARSE_MODEL = os.getenv("SPARSE_MODEL")
COLBERT_MODEL = os.getenv("COLBERT_MODEL")

QDRANT_CLUSTER_ENDPOINT = os.getenv("QDRANT_CLUSTER_ENDPOINT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


dense_model = TextEmbedding(DENSE_MODEL)
sparse_model = SparseTextEmbedding(SPARSE_MODEL)
colbert_model = LateInteractionTextEmbedding(COLBERT_MODEL)

quadrant_client = qdrant_client.QdrantClient(
    url = QDRANT_CLUSTER_ENDPOINT,
    api_key = QDRANT_API_KEY,
)

def read_instructions(file_path: str) -> str:
    """
    Lê o conteúdo de um arquivo de instruções.

    Args:
        file_path (str): O caminho para o arquivo de instruções.

    Returns:
        str: O conteúdo do arquivo de instruções.
    """
    with open(file_path, "r") as f:
        instructions = f.read()
    return instructions


def get_relevant_documents(query: str, top_k: int = 10):
    """
    Obtem documentos relevantes sobre o contexto de Téo Calvo e Téo Me Why, utilizando o Qdrant para busca semântica com base em uma query.

    Args:
        query (str): The query string.
        top_k (int, optional): The number of top results to return. Defaults to 10.

    Returns:
        str: The concatenated context from the top relevant documents.
    """
    
    dense_query = list(dense_model.passage_embed(query))[0].tolist()
    sparse_query = list(sparse_model.passage_embed(query))[0].as_object() 
    colbert_query = list(colbert_model.passage_embed(query))[0].tolist()

    results = quadrant_client.query_points(
        collection_name="ragia",
        prefetch={
            "prefetch": [
                {"query": dense_query, "using":"dense", "limit": 15},
                {"query": sparse_query, "using":"sparse", "limit": 15},
            ],
            "query": models.FusionQuery(fusion=models.Fusion.RRF),
            "limit":30,
        },
        query = colbert_query,
        using="colbert",
        limit=top_k,
    )

    context = "\n".join([f'- {r.payload["text"]}\n' for r in results.points])
    return context


model = OpenAILike(
    api_key="not-used",
    base_url="http://localhost:5001/v1",
    temperature=0.7,
)


## AGENTE DO CONTEXTO TMW
instructions_tmw_context = read_instructions("./instructions-tmw-research-agent.md")

tmw_research_agent = Agent(
    name="TMW Research Agent",
    instructions=instructions_tmw_context,
    role="Especialista em pesquisa e análise de informações sobre a Área de Dados, sobre o material de Téo Calvo e Téo Me Why, com foco em fornecer respostas detalhadas e contextualizadas.",
    markdown=True,
    tools=[get_relevant_documents],
    tool_call_limit=5,
)


## AGENTE DO CONTEXTO WEB
instructions_web_research_agent = read_instructions("./instructions-web-research-agent.md")

tmw_web_agent = Agent(
    name="WEB Research Agent",
    instructions=instructions_web_research_agent,
    role="Especialista em pesquisa e análise de informações na Web sobre temas quentes na área de dados e tecnologia.",
    markdown=True,
    tools=[TavilyTools()],
    tool_call_limit=5,
)


## AGENTE DA ESCRITA
instructions_writing_agent = read_instructions("./instructions-writing-agent.md")

tmw_writing_agent = Agent(
    name="Writing Agent",
    instructions=instructions_writing_agent,
    role="Especialista em escrita e comunicação, com foco em fornecer respostas detalhadas e contextualizadas sobre a Área de Dados, sobre o material de Téo Calvo e Téo Me Why.",
    markdown=True,
)

## TIME TMW
instructions_team = read_instructions("./instructions-team.md")


team = Team(
    name="TMW Team",
    members=[tmw_research_agent, tmw_web_agent, tmw_writing_agent],
    mode=TeamMode.tasks,
    instructions=instructions_team,
    model=model,
    db=SqliteDb("tmw_team.db"),
)

agent_os = AgentOS(teams=[team], tracing=True)
app = agent_os.get_app()
