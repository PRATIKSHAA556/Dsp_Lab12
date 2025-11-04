
import re
import pandas as pd
from typing import List, Dict, Any

# ---------- PII Patterns ----------

PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"\+?\d[\d\s-]{7,14}\b"),
    "zipcode": re.compile(r"\b\d{5,6}\b"),
    "name": re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b"),  # naive full name detection
}

# ---------- Function to Detect PII ----------

def detect_pii_in_text(text: str) -> List[Dict[str, Any]]:
    hits = []
    if not isinstance(text, str) or not text:
        return hits
    for ptype, pattern in PATTERNS.items():
        for m in pattern.finditer(text):
            hits.append({"pii_type": ptype, "match": m.group(0)})
    return hits

# ---------- Classify Dataset ----------

def classify_dataset(df: pd.DataFrame, state_tag="at_rest") -> pd.DataFrame:
    annotations = []
    for idx, row in df.iterrows():
        row_text = " | ".join([str(x) for x in row.values if x])
        hits = detect_pii_in_text(row_text)
        annotations.append({
            "row_index": idx,
            "pii_detected": [h["pii_type"] for h in hits],
            "pii_matches": [h["match"] for h in hits],
            "num_pii": len(hits),
            "data_type": "structured",   # since CSV is tabular
            "state": state_tag
        })
    ann_df = pd.DataFrame(annotations)
    return pd.concat([df.reset_index(drop=True), ann_df], axis=1)



if __name__ == "__main__":

    input_file = "data.csv"  
    df = pd.read_csv(input_file)

    # Annotate with PII classification

    annotated = classify_dataset(df, state_tag="at_rest")

    # Save output
    output_file = "data_output.csv"
    annotated.to_csv(output_file, index=False)

    print(f" Annotated dataset saved to {output_file}")
    print(annotated.head())
