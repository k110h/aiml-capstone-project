import streamlit as st
import pickle
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from openai import OpenAI

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Farmer Advisory System",
    layout="wide"
)

st.title("🌾 Farmer Advisory RAG System")

st.markdown(
    "Ask agriculture-related questions using Hybrid RAG Retrieval"
)

# =====================================
# LOAD RESOURCES
# =====================================

@st.cache_resource
def load_resources():

    # Load embedding model
    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    # Load FAISS index
    index = faiss.read_index(
        "faiss_index.bin"
    )

    # Load chunks
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    # Load BM25
    with open("bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)

    return model, index, chunks, bm25


model, index, chunks, bm25 = load_resources()

# =====================================
# OPENAI CLIENT
# =====================================

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# =====================================
# TOKENIZER
# =====================================

def tokenize(text):

    return text.lower().split()

# =====================================
# HYBRID SEARCH
# =====================================

def hybrid_search(query, k=5, alpha=0.5):
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    query_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, len(chunks))

    embedding_scores = np.zeros(len(chunks))
    for rank, idx in enumerate(indices[0]):
        embedding_scores[idx] = 1 / (1 + distances[0][rank])

    # Normalize
    bm25_scores = np.array(bm25_scores)
    if bm25_scores.max() != 0:
        bm25_scores = bm25_scores / bm25_scores.max()

    if embedding_scores.max() != 0:
        embedding_scores = embedding_scores / embedding_scores.max()

    #final_scores = alpha * bm25_scores + (1 - alpha) * embedding_scores

    # Rank fusion
    bm25_rank = np.argsort(np.argsort(-bm25_scores))
    embedding_rank = np.argsort(np.argsort(-embedding_scores))

    # Combine ranks
    final_scores = -(bm25_rank + embedding_rank)

    # Sort all results
    sorted_indices = np.argsort(final_scores)[::-1]

    # Remove duplicates based on normalized prefix (first 100 chars)
    # This catches near-duplicates and overlapping chunks
    seen_prefixes = set()
    results = []

    for i in sorted_indices:
        chunk = chunks[i]
        # Normalize and take a prefix for similarity checking
        normalized_text  = " ".join(chunk['text'].split()).lower()
        prefix = normalized_text

        if prefix not in seen_prefixes:
            results.append((i, chunk, final_scores[i]))
            seen_prefixes.add(prefix)

        if len(results) == k:
            break

    return results

# =====================================
# ANSWER GENERATION
# =====================================

def generate_answer(query, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks[:3])  # keep top 3 chunks

    prompt = f"""
You are an agriculture expert assistant.

Answer the question ONLY using the provided context.

Format your answer clearly:
- Use bullet points if suitable
- Keep answers clear and concise
- Use line breaks for readability
- Do not make up information

If the answer is not found in the context, say 
"Answer not found in documents provided.".

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",   # fast + cheap + good
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content
# =====================================
# USER INPUT
# =====================================

query = st.text_input(
    "Enter your agriculture question"
)

# =====================================
# RUN RAG PIPELINE
# =====================================

if st.button("Get Answer"):

    if query.strip() == "":

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Retrieving information..."
        ):

            # Retrieve chunks
            results = hybrid_search(
                query,
                k=9
            )

            # Top 3 chunks for LLM
            retrieved_chunks = [

                item["text"]

                for _, item, _
                in results[:3]
            ]

            # Generate answer
            answer = generate_answer(
                query,
                retrieved_chunks
            )

        # =====================================
        # DISPLAY ANSWER
        # =====================================

        st.subheader(
            "🧠 Generated Answer"
        )

        st.write(answer)

        # =====================================
        # DISPLAY SOURCES
        # =====================================

        st.subheader(
            "📄 Retrieved Chunks"
        )

        for idx, item, score in results[:3]:

            with st.expander(

                f"📄 {item['source']} | Score: {score:.4f}"

            ):

                st.write(
                    item["text"]
                )