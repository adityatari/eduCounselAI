import sqlite3
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig
import torch



model_name = "meta-llama/Llama-3.2-1B-Instruct"  # Adjust for 1B if you have it
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)


# Function to search programs in SQLite using FAISS
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

# Llama-3.2-1B-Instruct setup
class LlamaGuidanceBot:
    def __init__(self, tokenizer,model):
        self.tokenizer =tokenizer
        self.model = model
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def generate_response(self, chat_history, user_input):
        # Construct prompt using conversation history
        prompt = (
            "You are a helpful and knowledgeable guidance counselor for students.\n"
            f"Here is the conversation so far:\n{chat_history}\n"
            f"User: {user_input}\n"
            "Provide a helpful and detailed response:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        outputs = self.model.generate(**inputs, max_length=1024, temperature=0.7, top_p=0.9)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("Provide a helpful and detailed response:")[-1].strip()



def main():
    db_path = "/media/keagan/ssd_part2/buildathon/data/VideoBotDatabase.db"  # Update with your SQLite DB path
    chatbot = LlamaGuidanceBot( tokenizer,model)

    print("Guidance Counselor Chatbot\nType 'exit' to quit.\n")

    chat_history = ""  # To store conversation history

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        # Check if the user asks about programs
        if "program" in user_input.lower():
            print("Searching database...")
            results = search_programs_from_sqlite(db_path, user_input)
            response = "Here are some programs that match your query:\n"
            for idx, row in results.iterrows():
                response += (
                    f"{idx+1}. {row['Program_Name']} at {row['University']} "
                    f"(Location: {row['Location']}, QS Rank: {row['QS_Ranking']}, "
                    f"Expense Rank: {row['Expense_ranking']})\n"
                )
            chat_history += f"User: {user_input}\nChatbot: {response}\n"
            print(f"Chatbot: {response}")
        else:
            # Generate response using Llama
            response = chatbot.generate_response(chat_history, user_input)
            chat_history += f"User: {user_input}\nChatbot: {response}\n"
            print(f"Chatbot: {response}")

# Run the chatbot
if __name__ == "__main__":
    main()
