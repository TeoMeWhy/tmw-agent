import os
import dotenv
dotenv.load_dotenv()

from agno.agent import Agent
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.hackernews import HackerNewsTools
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

with open("./instructions-assistent-agent.md", "r") as f:
    instructions = f.read()

agent = Agent(
    name="Agent Assistant",
    model=model,
    db=SqliteDb("database.db"),
    instructions=instructions,
    role="Especialista em roteiros de conteúdo técnico para público geral, com foco em tecnologia, ciência e inovação",
    markdown=True,
    tools=[get_relevant_documents, DuckDuckGoTools(), HackerNewsTools()],
    tool_call_limit=5,
)


agent_os = AgentOS(agents=[agent], tracing=True)
app = agent_os.get_app()

