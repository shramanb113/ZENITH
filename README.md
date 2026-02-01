# ZENITH - A Next Genration Search and Recommendation Engine

> From lexical matching to true understanding

---

## 1. The Taxonomy of Research

- Search is ofen treated as a single problem. It is not.
- It is a **spectrum of intent**, and each layer solves a fundamentally different question

### 1.1 The Lexical Search - "Does the exact text match?"

- **Defintion** - Lexical Searching is about matching a sequence of characters excatly or a slight variant maybe in cases as given by the user.

> "Where does this word appear in the given document?"

**Example :-**

- Query : `Apple`
- Matches :
  - `I ate an apple yesterday.`
  - `Apple watches are great.`

**How it works :-**

- Tokenization
- BM25 / TF-ID scoring
- Inverted Indexes

**Strengths :-**

- Extremely Fast
- Precise for exact terminology
- Deterministic and explainable

**Limitations :-**

- No understanding of meaning
- Cannot generalize beyond seen words
- Fails on paraphrase and intent

**Mental Model**

> Lexical Search hears **sounds** , not **ideas**.

---

### 1.2 Semantic Search - "What does this mean"

- **Definiton -** Semantic search is about **meaning matching**. It represents text as points in a high-dimensional space and answers:

> "What concepts are similar to this idea?"

**Examples :**

- Query : `"Red Crunchy Fruit"` -> apple

- Query : `"IPhone Maker"` -> Apple Inc

- Query : `"language for fasr backend services"` -> Go (😊)

**How it works :**

- Natural Embeddings

- Vector similarity (cosine/dot product)

- Approximate Mearest Neighbor (ANN) indexes

---

### 1.3 Personlized Recommendation - "Who is asking?"

- **Definition** - Personalized recommendation incorported **context and identity**. It answers:

> "What is the most relevant result result for this ser right now?"

**Examples**

- A backend engineer searching `"Apple"` -> _Apple Silicon performance benchmarks_

- A nutritionist searching `"Apple"` -> _Nutritional breakdown_

- A student searching `"Apple"` -> _Company overview_

**Signals Used :**

- Past searches
- Click Behaviors
- Dwell Time
- User preferences
- Similar users (collaborative signals)

**Key Shifts**

Search is no lnger just **document** -> **query**

It becomes:

`(user,context,query) -> ranked results`

**Mental Model**

> Recommendation is **search with memory**

---
