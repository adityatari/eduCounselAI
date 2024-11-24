import sqlite3
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def process_text(text, max_chars=500):
    # Remove special characters but keep basic punctuation
    cleaned_text = ''.join(char for char in text if char.isalnum() or char in '. ,\n')
    
    # Initialize variables
    chunks = []
    current_chunk = ""
    
    # Split by sentences (roughly) and build chunks
    sentences = cleaned_text.split('.')
    
    for sentence in sentences:
        # Skip empty sentences
        if not sentence.strip():
            continue
            
        # If adding this sentence would exceed max_chars, start a new chunk
        if len(current_chunk) + len(sentence) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + "."
        else:
            current_chunk += sentence + "."
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def search_programs_from_sqlite(db_path, query, table_name="university_programs", top_k=5):
    """
    Search for university programs based on a student query using text embeddings,
    fetching data from an SQLite database.
    """
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    query_sql = f"SELECT * FROM {table_name};"
    data = pd.read_sql_query(query_sql, conn)
    conn.close()

    # Preprocess fields
    data["QS_Ranking"] = data["QS_Ranking"].astype(str)
    data["Expense_ranking"] = data["Expense_ranking"].astype(str)
    data["Study_Level"] = data["Study_Level"].astype(str)
    data["Program_Name"] = data["Program_Name"].astype(str)
    data["University"] = data["University"].astype(str)
    data["Location"] = data["Location"].astype(str)

    # Concatenate relevant fields
    data["text"] = (
        data["Program_Name"] + " " +
        data["University"] + " " +
        data["Location"] + " " +
        data["QS_Ranking"] + " " +
        data["Study_Level"] + " " +
        data["Expense_ranking"]
    )

    # Generate embeddings for the text fields
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(data["text"].tolist())

    # Create a FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    # Encode the query
    query_embedding = embedding_model.encode([query])

    # Search for the top_k results
    distances, indices = index.search(np.array(query_embedding), top_k)

    # Fetch results
    results = data.iloc[indices[0]].copy()
    results["distance"] = distances[0]
    return results
