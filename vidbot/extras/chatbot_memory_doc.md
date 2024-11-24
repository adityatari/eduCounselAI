# Guidance Counselor Chatbot Documentation
## Overview
This script implements a Guidance Counselor Chatbot using a combination of the Llama-3.2-1B-Instruct model and FAISS-based search for querying university programs from a SQLite database. The bot serves as a guidance counselor for students, helping them find university programs that align with their queries and providing personalized responses based on their preferences.

# Dependencies
Transformers: For interacting with Llama model.
Note: make sure u have access to Llama 3.2 1B since its a gated repo on hugging face you need to accept their terms & conditions
SentenceTransformers: For generating text embeddings used in FAISS similarity search.
FAISS: For efficient similarity search based on embeddings.
Langchain: For managing conversation memory.
SQLite3 & Pandas: For querying and processing data from the university database.


## LlamaGuidanceBot
Purpose: The bot class integrates Llama-3.2-1B-Instruct and handles user queries and conversation context.


The LlamaGuidanceBot is instantiated with the Llama-3.2-1B-Instruct model and a tokenizer.
A connection to the SQLite database is established, and a query function is defined to search for university programs using FAISS and SentenceTransformers.
User Interaction:

The chatbot continuously prompts the user for input.
If the user mentions "program" in their query, the chatbot searches the database using search_programs_from_sqlite to return relevant university programs.
If the user asks a general question, the chatbot generates a response using the Llama model, with conversation history maintained to ensure coherent dialogue.
Conversation History:

The chatbot uses ConversationBufferMemory from Langchain to store chat history, enabling the model to provide context-aware responses.
Exiting:

The loop terminates when the user types "exit".
Usage
Setup: Make sure the db_path points to a valid SQLite database containing university programs.
Running the bot:
The bot will prompt you for input. Type a question or statement.
The bot will search the database for relevant programs or provide a detailed response based on the conversation history.
Exit: Type exit to end the conversation.