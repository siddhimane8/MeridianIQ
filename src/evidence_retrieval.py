import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def build_clause_queries(detected_clauses_df, mvp_config):
    detected_clause_names = (
        detected_clauses_df[detected_clauses_df["detected"] == True]["clause_name"]
        .tolist()
    )

    detected_config = mvp_config[
        mvp_config["clause_name"].isin(detected_clause_names)
    ].copy()

    query_records = []

    for _, row in detected_config.iterrows():
        query_records.append({
            "clause_name": row["clause_name"],
            "query": f"{row['clause_name']}. {row['business_description']}"
        })

    return pd.DataFrame(query_records)


def retrieve_evidence(chunks_df, detected_clauses_df, mvp_config, embedding_model, filename, top_k=2):
    clause_queries_df = build_clause_queries(detected_clauses_df, mvp_config)

    if clause_queries_df.empty or chunks_df.empty:
        return pd.DataFrame(columns=[
            "filename", "clause_name", "rank", "similarity_score",
            "chunk_id", "evidence_text"
        ])

    chunk_embeddings = embedding_model.encode(
        chunks_df["chunk_text"].tolist(),
        show_progress_bar=True,
        convert_to_numpy=True
    )

    query_embeddings = embedding_model.encode(
        clause_queries_df["query"].tolist(),
        show_progress_bar=True,
        convert_to_numpy=True
    )

    similarity_matrix = cosine_similarity(query_embeddings, chunk_embeddings)

    evidence_records = []

    for query_idx, row in clause_queries_df.iterrows():
        scores = similarity_matrix[query_idx]
        top_indices = np.argsort(scores)[::-1][:top_k]

        for rank, chunk_idx in enumerate(top_indices, start=1):
            chunk = chunks_df.iloc[chunk_idx]

            evidence_records.append({
                "filename": filename,
                "clause_name": row["clause_name"],
                "rank": rank,
                "similarity_score": round(float(scores[chunk_idx]), 4),
                "chunk_id": int(chunk["chunk_id"]),
                "evidence_text": chunk["chunk_text"]
            })

    return pd.DataFrame(evidence_records)