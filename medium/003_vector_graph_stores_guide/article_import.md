![A field guide to LlamaIndex vector and graph stores — hero banner](https://raw.githubusercontent.com/amadou-6e/blog-components/main/medium/003_vector_graph_stores_guide/images/title.png)

# A field guide to vector and graph stores in LlamaIndex: Neo4j, pgvector, Qdrant, and OpenSearch
*When to use each one, what the local setup looks like, and where each fits in production*

---

The LlamaIndex documentation lists over forty storage integrations. In practice, if you are building a GraphRAG or RAG pipeline with a local development workflow, you are choosing from a much shorter list. Four stores cover the meaningful ground: Neo4j for knowledge graphs, pgvector for SQL-native vector search, Qdrant for purpose-built vector indexing, and OpenSearch for hybrid text-plus-vector retrieval.

This post covers those four. For each one: what it actually stores, when it is the right fit, what the local Docker setup looks like, how you connect it through LlamaIndex, and what the production deployment path is. The goal is a reference you can return to when a new pipeline needs a storage decision — not a ranking, not a benchmark, not a verdict about which one wins.

Selection criteria: I picked stores that (1) have a maintained LlamaIndex integration, (2) have a well-understood local Docker setup, and (3) map to meaningfully different use cases. Chroma and Milvus Lite are excluded because they are better fits for rapid prototyping with ephemeral data than for pipelines you intend to deploy. Weaviate and Pinecone are excluded because their local development story requires more infrastructure than the four covered here.

![Neo4j, pgvector, Qdrant, OpenSearch — store comparison table](https://raw.githubusercontent.com/amadou-6e/blog-components/main/medium/003_vector_graph_stores_guide/images/store_comparison.png)

---

## Neo4j: graph database for knowledge graphs and GraphRAG

**What it is:** A native graph database. Data is stored as nodes, relationships, and properties — not tables. Queries are written in Cypher, which describes the graph patterns you want to find.

**When to use it:** When your retrieval questions are about relationships between entities, not just semantic proximity. The canonical use case is GraphRAG with LlamaIndex's `PropertyGraphIndex`: entity extraction from documents, knowledge graph construction, community detection, and multi-hop retrieval queries like "find papers that cite this author and were also tagged with this topic." If your questions require following chains of connections across a corpus, Neo4j is the right backend. If they are primarily similarity lookups, it is not.

**Local Docker setup:**

```bash
docker run --name neo4j-dev \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5
```

Port 7474 is the browser UI (useful for inspecting the graph). Port 7687 is the Bolt protocol endpoint that LlamaIndex connects to. The first run pulls the image and initializes the database, which takes 20–30 seconds. You can verify it is ready by opening `http://localhost:7474` or running:

```bash
docker exec neo4j-dev cypher-shell -u neo4j -p password "RETURN 1 AS ok"
```

**LlamaIndex connection:**

```python
pip install llama-index-graph-stores-neo4j neo4j
```

```python
from llama_index.graph_stores.neo4j import Neo4jGraphStore

graph_store = Neo4jGraphStore(
    username="neo4j",
    password="password",
    url="bolt://localhost:7687",
    database="neo4j",
)
```

For GraphRAG V2, pass the store to `PropertyGraphIndex`:

```python
from llama_index.core import PropertyGraphIndex

index = PropertyGraphIndex.from_documents(
    documents,
    graph_store=graph_store,
    show_progress=True,
)
```

**Production path:** Neo4j Aura (managed cloud). Swap `url` from `bolt://localhost:7687` to your Aura connection string. The rest of the code is unchanged. Aura Free tier is available for development. Aura Professional supports production workloads.

**Verdict:** The strongest fit for knowledge graph construction and GraphRAG. The wrong choice for pure similarity search — there is no native vector index in the base `Neo4jGraphStore`. (Neo4j does support vector indexing via `Neo4jPropertyGraphStore` with embeddings stored on nodes, but that is a separate integration with a different tradeoff profile.)

---

## pgvector (PostgreSQL): SQL-native vector search

**What it is:** A PostgreSQL extension that adds a `vector` column type and approximate nearest-neighbor index support. You get vector search inside a database you already know how to operate, with full SQL available alongside it.

**When to use it:** When your application already uses PostgreSQL, or when you want to store structured metadata alongside vectors and filter on it using SQL. The `PGVectorStore` in LlamaIndex supports metadata filters and hybrid search (vector similarity + keyword match). If your team knows Postgres and your data has relational structure that matters for filtering, pgvector removes one infrastructure dependency from your stack.

**Local Docker setup:**

```bash
docker run --name pgvector-dev \
  -p 5432:5432 \
  -e POSTGRES_USER=testuser \
  -e POSTGRES_PASSWORD=testpassword \
  -e POSTGRES_DB=vectordb \
  pgvector/pgvector:pg16
```

The `pgvector/pgvector:pg16` image ships with the extension pre-installed. After the container starts, the extension still needs to be activated in the target database:

```bash
docker exec -it pgvector-dev psql -U testuser -d vectordb -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

**LlamaIndex connection:**

```python
pip install llama-index-vector-stores-postgres psycopg2-binary pgvector
```

```python
from llama_index.vector_stores.postgres import PGVectorStore

vector_store = PGVectorStore.from_params(
    database="vectordb",
    host="localhost",
    password="testpassword",
    port=5432,
    user="testuser",
    table_name="llamaindex_vectors",
    embed_dim=1536,  # match your embedding model's output dimension
)
```

**Production path:** Any managed PostgreSQL with the pgvector extension enabled. Supabase (free tier available, pgvector on by default), Neon (serverless Postgres, pgvector supported), Amazon RDS for PostgreSQL (pgvector available from PostgreSQL 15+), or Google Cloud SQL. For teams already on AWS or GCP with an existing RDS instance, adding a `vector` column to an existing database is often the path of least resistance.

**Verdict:** The most familiar option for teams with existing Postgres infrastructure. The right choice when you need to join vector results with structured data in the same query. Not the best choice for very large vector collections (>10M embeddings) where purpose-built ANN indexes (HNSW in Qdrant) will outperform PostgreSQL's ivfflat.

---

## Qdrant: purpose-built vector search

**What it is:** A vector database built specifically for approximate nearest-neighbor search. Uses HNSW (Hierarchical Navigable Small World) indexing as its default, with payload filtering that runs inside the ANN search rather than as a post-filter.

**When to use it:** When vector similarity search is the primary operation and you need it to be fast at scale. Qdrant's payload filtering is applied during the HNSW traversal, which means filtering on metadata (date range, category, source document) does not degrade recall the way post-filtering does. If you are building a semantic search API over a large corpus with rich metadata, Qdrant is the strongest fit in this list for that specific workload.

**Local Docker setup:**

```bash
docker run --name qdrant-dev \
  -p 6333:6333 \
  -p 6334:6334 \
  qdrant/qdrant
```

Port 6333 is the REST API and browser dashboard. Port 6334 is the gRPC endpoint. The dashboard at `http://localhost:6333/dashboard` shows collections, vectors, and payload counts — useful for inspecting what LlamaIndex has stored.

**LlamaIndex connection:**

```python
pip install llama-index-vector-stores-qdrant qdrant-client
```

```python
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore

client = QdrantClient(host="localhost", port=6333)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="my_documents",
)
```

**Production path:** Qdrant Cloud (managed, free tier available at 1GB storage). Self-hosted Qdrant on Kubernetes is also well-documented for teams that need to keep data on-premise. The same `QdrantClient` constructor accepts a cloud URL and API key:

```python
client = QdrantClient(
    url="https://your-cluster.qdrant.io",
    api_key="your-api-key",
)
```

**Verdict:** The strongest pure vector search option in this list at scale. The HNSW + payload filter combination is Qdrant's primary differentiation from pgvector. The tradeoff is that it is a single-purpose service: if your pipeline later needs relational queries over the same data, you are running two stores.

---

## OpenSearch: hybrid text and vector search

**What it is:** A fork of Elasticsearch (Apache 2.0 licensed). Stores documents with an inverted index for full-text search, plus an approximate nearest-neighbor index for dense vector search. Supports hybrid scoring that combines BM25 keyword relevance with vector similarity in a single query.

**When to use it:** When your retrieval pipeline needs both keyword precision and semantic recall. The canonical case is a RAG system over technical documentation or code where exact-match queries ("what is the return type of function X") should rank highly alongside semantic queries ("how does the authentication system work"). Hybrid search handles both in one store, one query. It is also the natural fit for teams migrating from Elasticsearch.

**Local Docker setup:**

```bash
docker run --name opensearch-dev \
  -p 9200:9200 \
  -p 9600:9600 \
  -e "discovery.type=single-node" \
  -e "DISABLE_SECURITY_PLUGIN=true" \
  opensearchproject/opensearch:2.11.0
```

`DISABLE_SECURITY_PLUGIN=true` skips TLS certificate setup for local development. Do not use this flag in production. The container takes 30–45 seconds to be ready. Check with:

```bash
curl http://localhost:9200/_cluster/health
```

**LlamaIndex connection:**

```python
pip install llama-index-vector-stores-opensearch opensearch-py
```

```python
from llama_index.vector_stores.opensearch import (
    OpensearchVectorStore,
    OpensearchVectorClient,
)

client = OpensearchVectorClient(
    endpoint="http://localhost:9200",
    index="my_documents",
    dim=1536,
    embedding_field="embedding",
    text_field="content",
)

vector_store = OpensearchVectorStore(client)
```

**Production path:** Amazon OpenSearch Service (managed, integrates with existing AWS infrastructure). OpenSearch Serverless is available for workloads with unpredictable traffic. Self-hosted OpenSearch clusters are common in enterprises with data residency requirements.

**Verdict:** The right choice when keyword search precision matters alongside semantic recall, or when the team is already operating an Elasticsearch/OpenSearch cluster. The local Docker image is the heaviest of the four options here (the single-node container uses 1–2GB RAM at startup), which is a meaningful operational cost for local development sessions.

---

## How to choose

The choice depends on two questions: what kind of retrieval do you need, and what does your infrastructure already look like.

![Store selection decision tree — choosing a backend for your pipeline](https://raw.githubusercontent.com/amadou-6e/blog-components/main/medium/003_vector_graph_stores_guide/images/decision_tree.png)

**Use Neo4j** when: your queries are about relationships between entities. You are building GraphRAG, a knowledge graph, or a pipeline where multi-hop traversal is the primary retrieval mechanism.

**Use pgvector** when: your team already runs PostgreSQL, or you need to join vector results with structured relational data. Best for medium-scale deployments where simplicity and familiarity matter.

**Use Qdrant** when: similarity search is the primary operation and scale matters. Best for semantic search APIs over large corpora with rich metadata filtering requirements.

**Use OpenSearch** when: you need hybrid text-plus-vector retrieval, or you are already running an Elasticsearch cluster.

The four stores are not mutually exclusive in a production pipeline. LlamaIndex's `RouterQueryEngine` can dispatch queries to different stores based on the query type — routing graph traversal queries to Neo4j and similarity queries to Qdrant, for example. The field guide framing here is meant for choosing the primary store for a given pipeline, not for choosing a single store for all use cases forever.

One thing all four have in common: getting any of them running locally requires a `docker run` command before you can write a single line of retrieval logic. That setup step is not in the LlamaIndex documentation. It is in the Docker Hub pages, the individual store documentation, and scattered across community tutorials. The next post addresses that directly: what it looks like when the Python import handles it for you.

---

*Next: one import swap — spinning up a local Neo4j GraphRAG pipeline without leaving Python.*

---
