# What Neo4j actually does and how it fits into a GraphRAG pipeline (with a working LlamaIndex example)

Every LlamaIndex GraphRAG tutorial starts the same way. Before the first useful Python line, there is a `docker run` command for Neo4j. That makes Neo4j feel like plumbing, not part of the retrieval design.

![What Neo4j actually does - hero banner](https://raw.githubusercontent.com/amadou-6e/blog-components/main/medium/002_neo4j_llamaindex/images/title.png)

That framing is backwards. If your questions are relational, the storage layer is not incidental. It determines whether the retrieval system can follow the path the answer depends on. This post explains what Neo4j actually stores, why that storage model fits knowledge graphs better than a relational database for traversal-heavy queries, and how the LlamaIndex integration turns that graph into a working GraphRAG pipeline. Here, GraphRAG means a retrieval pipeline that uses both text artifacts and graph structure, so the system can answer questions about relationships between entities instead of only returning semantically similar passages.

---

## The property graph model: what Neo4j actually stores

***Neo4j stores a property graph, not tables that get reconstructed into a graph later.***

In Neo4j's data model, a graph is made of **nodes**, **relationships**, and **properties**. Neo4j's own getting-started docs define graph databases in exactly those terms: data is stored as nodes, relationships, and properties rather than as tables or documents ([Neo4j Docs](https://neo4j.com/docs/getting-started/graph-database/)). That structure matters because GraphRAG questions are usually about entities plus typed links between them, not about isolated records.

That model has four parts:

1. **Nodes** represent entities such as papers, researchers, institutions, or topics.
2. **Relationships** are directed, typed connections between nodes, such as `CITES` or `AUTHORED_BY`.
3. **Properties** are key-value pairs attached to either nodes or relationships.
4. **Labels and relationship types** tell Neo4j what kind of thing each record is, and they support indexing and constraints.

In practice, that looks like this:

```cypher
(p:Paper {openalex_id: "W123", title: "Attention Is All You Need", year: 2017})
(paper)-[:CITES {source: "openalex"}]->(other_paper)
```

The query language for that model is **Cypher**, Neo4j's pattern-matching language. The `MATCH` clause lets you describe the pattern you want the database to traverse ([Cypher Manual](https://neo4j.com/docs/cypher-manual/current/clauses/match/)):

```cypher
MATCH (candidate:Paper)-[:CITES]->(anchor:Paper {title: "Attention Is All You Need"})
RETURN candidate.title, candidate.year
ORDER BY candidate.year DESC
LIMIT 10
```

The useful part is not just the syntax. It is that the query shape matches the graph shape. You ask for a pattern, not a stack of joins that only implies a pattern. If your retrieval question depends on who authored what, which paper cites which result, or which topic connects two otherwise distant papers, that difference becomes practical very quickly.

![The Neo4j property graph model - nodes, relationships, properties, labels](https://raw.githubusercontent.com/amadou-6e/blog-components/main/medium/002_neo4j_llamaindex/images/property_graph_model.png)
*The graph is already there in storage. Query time is just following it.*

## Why a graph database fits knowledge graphs better than a relational one

A **knowledge graph** is not just a bag of facts. It is a representation where entities are stored as nodes and their connections are stored as typed edges, so the system can answer questions about how things relate, not only what text mentions them.

You can model the same domain in Postgres. That part is not controversial.

```sql
CREATE TABLE papers (id TEXT PRIMARY KEY, title TEXT, year INT);
CREATE TABLE citations (citing_id TEXT REFERENCES papers(id), cited_id TEXT REFERENCES papers(id));
CREATE TABLE authorships (paper_id TEXT REFERENCES papers(id), author_id TEXT REFERENCES authors(id));
```

But the moment the query becomes relational, the database has to recover the path through repeated joins. Questions like these are where the difference starts to matter:

- Which papers cite Paper A?
- Which of those papers share an author with a paper tagged as retrieval augmentation?
- Which path actually proves that connection?

In Postgres, that becomes a chain over citation, authorship, and topic tables. In Cypher, the same request is still expressed as the path itself:

```cypher
MATCH (anchor:Paper {openalex_id: $anchor_id})
MATCH (candidate:Paper)-[:CITES]->(anchor)
MATCH (candidate)-[:AUTHORED_BY]->(bridge:Researcher)<-[:AUTHORED_BY]-(rag_paper:Paper)
MATCH (rag_paper)-[:TAGGED]->(:Topic {name: 'retrieval augmentation'})
RETURN candidate.title, collect(DISTINCT bridge.name) AS bridge_authors
```

The performance idea you need here is **index-free adjacency**. Neo4j describes this as a model where the database uses an index to find the starting node, then follows stored links between node and relationship records during traversal instead of reconstructing every hop through join tables ([Neo4j Overview](https://neo4j.com/graphacademy/training-overview-40/02-overview40-neo4j-graph-platform/)). That is why graph databases are a natural fit when the dominant workload is multi-hop traversal instead of row aggregation. This does not make Neo4j universally better. It makes it a better fit for a specific query shape.

- If your workload is mostly semantic similarity over chunks, use a vector store.
- If your workload is mostly aggregates, filters, and tabular reporting, use a relational database.
- If your workload depends on typed paths across entities, a graph database earns its place quickly.

![The same question in SQL and Cypher - query complexity comparison](https://raw.githubusercontent.com/amadou-6e/blog-components/main/medium/002_neo4j_llamaindex/images/neo4j_vs_sql.png)
*This is where row logic starts to fight a path-shaped question.*

## How LlamaIndex connects to Neo4j

LlamaIndex's Neo4j property-graph integration exposes Neo4j as the storage layer behind a property graph index ([LlamaIndex Docs](https://docs.llamaindex.ai/en/stable/api_reference/storage/graph_stores/neo4j/)). At a high level, that integration does two jobs:

1. **Execute** Cypher against Neo4j for graph reads and writes.
2. **Back** a `PropertyGraphIndex` so extracted entities and relationships can be stored in a real graph database.

That second part matters for GraphRAG. A **PropertyGraphIndex** is LlamaIndex's index structure for storing text-derived entities and relations in a property graph, so retrieval can use both document context and graph structure. In LlamaIndex's property-graph workflow, documents are chunked, **path extractors** pull entity-to-entity relationships from those chunks, and the result is written into Neo4j as nodes and typed edges ([LlamaIndex example](https://docs.llamaindex.ai/en/stable/examples/property_graph/property_graph_neo4j/)). For this article, the graph construction is written by hand instead of delegating entity extraction to an LLM pipeline. That makes the mechanics easier to inspect and keeps the example grounded in the actual nodes and edges being inserted.

```bash
pip install llama-index-graph-stores-neo4j neo4j pandas requests scikit-learn
```

The local Neo4j container for the example looks like this:

```bash
docker run --name neo4j-dev \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5
```

Then the Python side is just the store connection and a minimal connectivity check:

```python
import os
from llama_index.graph_stores.neo4j import Neo4jGraphStore
graph_store = Neo4jGraphStore(
    username=os.getenv("NEO4J_USER", "neo4j"),
    password=os.getenv("NEO4J_PASSWORD", "password"),
    url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    database=os.getenv("NEO4J_DATABASE", "neo4j"),
    refresh_schema=False,
)
graph_store.query("RETURN 1 AS ok")
```

`refresh_schema=False` skips a startup schema refresh. When you already know the graph shape, that avoids extra overhead before the first query.

## Building the arXiv citation graph

OpenAlex's `Works` API is the source for the paper records in this notebook, including outgoing references and linked metadata ([OpenAlex Docs](https://docs.openalex.org/api-entities/works)).

The fetch in this example does four things:

1. **Search** topic areas related to retrieval and transformer literature.
2. **Filter** to records that are linked to arXiv.
3. **Normalize** each work into one consistent Python shape.
4. **Load** papers, authors, institutions, topics, and citation edges into Neo4j.

The topic field is normalized from two sources: manual seed topics used to drive the fetch, and OpenAlex topic or concept identifiers attached to each returned work. After normalization, each paper record carries the fields the graph needs: the paper ID, basic paper metadata, author records, institution records, topic tags, and outgoing references.

```python
import requests
import time
OPENALEX_URL = "https://api.openalex.org/works"
def is_arxiv_linked(work):
    doi = (work.get("doi") or "").lower()
    if "10.48550/arxiv" in doi:
        return True
    for loc in work.get("locations") or []:
        if "arxiv.org" in str(loc.get("landing_page_url") or "").lower():
            return True
    return False
def fetch_seed_works(query_text, pages=2, per_page=100):
    rows = []
    for page in range(1, pages + 1):
        params = {
            "search": query_text,
            "filter": "from_publication_date:2017-01-01,has_abstract:true",
            "per-page": per_page,
            "page": page,
            "mailto": "demo@example.com",
        }
        r = requests.get(OPENALEX_URL, params=params, timeout=60)
        r.raise_for_status()
        rows.extend(r.json().get("results", []))
        time.sleep(0.2)
    return rows
```

The ingest step uses `MERGE`, which is Cypher's idempotent write pattern. It creates the node if it does not exist and matches it if it already does. That is what makes reruns safe when you are rebuilding the graph from the same source records.

The constraint setup looks like this:

```cypher
CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.openalex_id IS UNIQUE;
CREATE CONSTRAINT researcher_id IF NOT EXISTS FOR (r:Researcher) REQUIRE r.author_id IS UNIQUE;
CREATE CONSTRAINT institution_id IF NOT EXISTS FOR (i:Institution) REQUIRE i.institution_id IS UNIQUE;
CREATE CONSTRAINT topic_id IF NOT EXISTS FOR (t:Topic) REQUIRE t.topic_id IS UNIQUE;
```

Then the Python ingest loop can stay small and mechanical:

```python
for paper in graph_papers:
    graph_store.query(
        "MERGE (p:Paper {openalex_id: $paper_id}) "
        "SET p.title = $title, p.year = $year, p.cited_by_count = $cited_by_count",
        param_map={
            "paper_id": paper["openalex_id"],
            "title": paper["title"],
            "year": paper["year"],
            "cited_by_count": paper["cited_by_count"],
        },
    )
```

In the notebook run used for this article, the ingest produced roughly 450 arXiv-linked papers plus paper, researcher, institution, and topic nodes with around 15,000 citation edges. The point is not the absolute size. The point is that the graph is now queryable as a network rather than as a pile of paper records.

## The multi-hop query

***This is the point where a graph backend stops being theoretical.***

The core query asks for papers that cite "Attention Is All You Need" whose authors also published papers tagged as retrieval augmentation.

```cypher
MATCH (anchor:Paper {openalex_id: $attention_id})
MATCH (candidate:Paper)-[:CITES]->(anchor)
MATCH (candidate)-[:AUTHORED_BY]->(bridge:Researcher)<-[:AUTHORED_BY]-(rag_paper:Paper)
MATCH (rag_paper)-[:TAGGED]->(:Topic {name: 'retrieval augmentation'})
RETURN candidate.title AS citing_paper,
       candidate.year AS year,
       collect(DISTINCT bridge.name)[0..3] AS bridge_authors,
       collect(DISTINCT rag_paper.title)[0..3] AS related_rag_papers
ORDER BY year DESC
LIMIT 20
```

Here, the `bridge` researcher is simply the author node that connects the citing paper to a second paper in the retrieval-augmentation topic set. The retrieval path is easier to scan when written as hops:

```text
anchor paper
-> cited by candidate paper
-> authored by bridge researcher
-> also authored retrieval-augmentation paper
```

That is a **multi-hop query**. Multi-hop means the answer depends on following more than one relationship in sequence. A vector ranking system can retrieve papers that mention the right terms. It cannot prove that those papers satisfy this citation-plus-authorship path unless the path itself is encoded somewhere the retriever can traverse.

![Multi-hop traversal path in the arXiv citation graph](https://raw.githubusercontent.com/amadou-6e/blog-components/main/medium/002_neo4j_llamaindex/images/multihop_path.png)
*Similarity can suggest the paper. Only the path can prove it.*

In Python, the execution is direct:

```python
ATTENTION_PAPER_ID = "https://openalex.org/W2626778328"
results = graph_store.query(
    cypher_multihop,
    param_map={"attention_id": ATTENTION_PAPER_ID}
)
import pandas as pd
pd.DataFrame(results)
```

A representative row looks like this:

```text
citing_paper: "Dense Passage Retrieval for Open-Domain Question Answering"
year: 2020
bridge_authors: ["Danqi Chen", "Wen-tau Yih"]
related_rag_papers: ["Open-domain question answering via contextual word vectors", ...]
```

## Validating results and comparing against a vector baseline

The validation pass reruns each returned title against the actual graph constraints:

1. **Check** that the paper exists.
2. **Check** that it cites the anchor paper.
3. **Check** that at least one author bridges to a retrieval-augmentation paper.
4. **Reject** any row that fails any part of that path.

In the recorded notebook run used for this article, the graph query returned 18 rows and all 18 satisfied the full constraint set. Precision at 18 was 1.000. The comparison baseline uses TF-IDF cosine similarity over paper titles for the query text "papers that cite attention is all you need with co-authors active in retrieval augmentation."

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
corpus = [p["title"] for p in graph_papers if p.get("title")]
query = "papers that cite attention is all you need with co-authors active in retrieval augmentation"
tfidf = TfidfVectorizer(stop_words="english")
matrix = tfidf.fit_transform(corpus)
scores = cosine_similarity(tfidf.transform([query]), matrix)[0]
```

In that same run, the vector top-20 match rate was 0.15. Most of the high-scoring titles reused words like "attention", "retrieval", or "augmentation" but did not satisfy the citation and authorship constraints. That is the core contrast:

- Vector search is good at semantic resemblance.
- Graph traversal is good at relational validity.

The caveat is straightforward. This is a small corpus and one anchor paper. The exact metric values will move on a different sample, but the mechanism survives the example: similarity and path validity are different retrieval signals.

## When to use Neo4j and when not to

Use Neo4j when your retrieval questions sound like this:

- Which papers cite this result through the same bridge author?
- Which institutions connect researchers across both topic clusters?
- Which entity chain explains why these two documents belong together?

Those are graph questions. Do not use Neo4j just because a graph looks sophisticated on a slide.

- For pure top-k semantic retrieval over chunks, a vector store is usually the simpler fit.
- For aggregates, dashboards, and row-heavy analytics, a relational database is often the better fit.
- For GraphRAG systems that need both local semantic retrieval and relationship-aware traversal, the graph store and the vector index complement each other.

Neo4j also brings operational cost. You need a running database, persistent storage, a Bolt endpoint, and real lifecycle management once the project moves beyond a notebook. That cost is justified when the graph answers questions the vector path cannot. The full notebook behind this article includes the OpenAlex fetch, graph construction, multi-hop query, validation, and vector baseline comparison.

---

*Next: the repeated infrastructure cost hidden inside every "just run Neo4j first" tutorial, and why that friction keeps showing up before the interesting part of the experiment even starts.*
