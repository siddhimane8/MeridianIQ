import re
import pandas as pd


def to_snake_case(text):
    text = str(text).lower()
    text = text.replace("/", "_")
    text = text.replace("-", "_")
    text = text.replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]", "", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def predict_clauses(clean_text, tfidf, clause_detector, mvp_config, filename="uploaded_contract.txt"):
    label_cols = [
        to_snake_case(clause)
        for clause in mvp_config["clause_name"].tolist()
    ]

    X_tfidf = tfidf.transform([clean_text])
    predictions = clause_detector.predict(X_tfidf)

    predicted_fingerprint = pd.DataFrame(
        predictions,
        columns=label_cols
    )

    predicted_fingerprint.insert(0, "filename", filename)

    detected_records = []

    for clause_name, col in zip(mvp_config["clause_name"], label_cols):
        detected_records.append({
            "clause_name": clause_name,
            "clause_column": col,
            "detected": bool(predicted_fingerprint.loc[0, col])
        })

    detected_clauses_df = pd.DataFrame(detected_records)

    return predicted_fingerprint, detected_clauses_df