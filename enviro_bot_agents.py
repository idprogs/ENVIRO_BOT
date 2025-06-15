# Code by Ian Drumm
import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM
from tools.my_utils_f import (
    log_comment, some_llm_filter, my_reddit_scrape, evaluate_data,
    create_chain3, add_to_vector_db_with_metadata, getAllDBItems,
    MyCustomCallbackHandler, convert_to_chroma_filter
)
from tools.my_classifiers import get_viewpoint, get_emotion, get_sentiment, get_style, get_training_or_test_data
import argparse
import json
from tools.my_logging import my_log

# Load environment variables from .env file if it exists
load_dotenv()

# --- LLM Initialization ---
# Initialize various language models for different tasks within the agentic workflow.
# Temperature settings are adjusted based on the desired creativity/determinism for each task.

# General purpose LLMs
mistral = OllamaLLM(model="mistral-nemo", temperature=0.0)
llama3_70b = OllamaLLM(model="llama3:70b",temperature=0.3)

# OpenAI's GPT-4o model, can be used if you have an openai API key set up in a .env file in this directory.
# gpt_4o = ChatOpenAI(model="gpt-4o", temperature=0, api_key=os.getenv('OPEN_AI_API_KEY'))

filter_llm=mistral
classify_llm=llama3_70b
use_llm=llama3_70b
evaluate_llm=llama3_70b

# --- State Definition for LangGraph ---
# Defines the structure of the data that flows through the graph.
class ItemState(TypedDict):
    items: list[dict]
    filtered_items: list[dict]
    classified_items: list[dict]
    all_classified: list[dict]

# Initialize the LangGraph
graph = StateGraph(ItemState)

# --- Node Definitions for LangGraph ---
# Each function represents a node in the graph, performing a specific data processing step.

def filter_items(state):
    """Filters items based on a predefined LLM-based filter."""
    filtered_items = [item for item in state['items'] if some_llm_filter(item=item)]
    return {"filtered_items": filtered_items}

def classify_viewpoint(state):
    """Classifies the viewpoint of each filtered item."""
    classified_items = []
    for item in state['filtered_items']:
        classification = get_viewpoint(item=item, llm=classify_llm)
        if classification:
            item["Viewpoint"] = classification
            classified_items.append(item)
    # Note: This overwrites previous classifications if nodes are chained sequentially modifying 'classified_items'.
    # Consider appending to a new list or updating items in place if preserving prior step's full list is needed.
    return {"classified_items": classified_items} 

def classify_emotion(state):
    """Classifies the emotion of each filtered item."""
    classified_items = []
    for item in state['filtered_items']:
        classification = get_emotion(item=item, llm=classify_llm)
        if classification:
            item["Emotion"] = classification
            classified_items.append(item)
    return {"classified_items": classified_items}

def classify_sentiment(state):
    """Classifies the sentiment of each filtered item."""
    classified_items = []
    for item in state['filtered_items']:
        classification = get_sentiment(item=item, llm=classify_llm)
        if classification:
            item["Sentiment"] = classification
            classified_items.append(item)
    return {"classified_items": classified_items}

def classify_style(state):
    """Classifies the style of each filtered item."""
    classified_items = []
    for item in state['filtered_items']:
        classification = get_style(item=item, llm=classify_llm)
        if classification:
            item["Style"] = classification
            classified_items.append(item)
    return {"classified_items": classified_items}

def classify_training_or_test_data(state):
    """Marks data as 'Training' or 'Test' for evaluation purposes (e.g., 75/25 split)."""
    classified_items = []
    for item in state['filtered_items']:
        classification = get_training_or_test_data(item=item)
        if classification:
            item["Training_or_Test_Data"] = classification
            classified_items.append(item)
    return {"classified_items": classified_items}

def final_data(state):
    """Consolidates all classified items into a final list."""
    all_items = []
    for item in state['classified_items']:
        all_items.append(item)
    return {"all_classified": all_items}

# --- Graph Construction ---
# Add nodes to the graph
graph.add_node("filter", filter_items)
graph.add_node("classify_viewpoint", classify_viewpoint)
graph.add_node("classify_emotion", classify_emotion)
graph.add_node("classify_sentiment", classify_sentiment)
graph.add_node("classify_style", classify_style)
graph.add_node("classify_training_or_test_data", classify_training_or_test_data)
graph.add_node("finish", final_data)

# Set entry point
graph.set_entry_point("filter")

# Define edges
graph.add_edge("filter", "classify_viewpoint")
graph.add_edge("classify_viewpoint", "classify_emotion")
graph.add_edge("classify_emotion", "classify_sentiment")
graph.add_edge("classify_sentiment", "classify_style")
graph.add_edge("classify_style", "classify_training_or_test_data")
graph.add_edge("classify_training_or_test_data", "finish")
graph.set_finish_point("finish")

# Compile the graph
app = graph.compile()

# --- Utility and Processing Functions ---
def split_into_chunks(items: list[dict[str, str]], chunk_size: int) -> list[list[dict[str, str]]]:
    """Splits a list of items into smaller chunks of a specified size."""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

def process_chunks(graph, items: list[dict[str, str]], chunk_size: int):
    """Processes items in chunks through the compiled LangGraph application."""
    chunks = split_into_chunks(items, chunk_size)
    all_classified = []
    
    for i, chunk in enumerate(chunks, 1):
        inputs = {"items": chunk}
        try:
            result = graph.invoke(inputs)
            all_classified.extend(result['all_classified'])
            print(f"Processed chunk {i} of {len(chunks)}")
        except Exception as e:
            print(f"Error processing chunk {i}: {e}")
    
    return all_classified

# --- Main Execution Block ---
def main():
    print("Hello!")
    parser = argparse.ArgumentParser(description='Create or access a ChromaDB vector store.')
    parser.add_argument('action', choices=['create','chat','evaluate','scrape','pull'], help='Actions to perform: create a new database or chat to an existing one')
    parser.add_argument('--posts_limit', type=int, default=50000, help='Limit for the number of posts to fetch')
    parser.add_argument('--comments_limit', type=int, default=10000, help='Limit for the number of comments to fetch')
    parser.add_argument('--vdb', default="./vdb/climate_rag_vdb", help='path to vector database')
    parser.add_argument('--delete_old_vdb', type=bool, default=False, help='delete previous vector database')
    parser.add_argument('--s', default="mmr", help='search_type e.g. similarity or mmr')
    parser.add_argument('--k', type=int, default=1, help='number of retrival documents')
    parser.add_argument('--t', type=float, default=0.0, help='temperature of model')
    parser.add_argument('--samples', type=int, default=10, help='sample size')
    parser.add_argument('--llm_name', default="llama3:70b", help='temperature of model')
    parser.add_argument('--out_file', default="eval", help='output file name')
    parser.add_argument('--json', default="./data/all_classified.json", help='json to add to database')
    parser.add_argument('--search', default='"wind turbines" OR "wind farms" OR "wind energy" OR "windmills"', help='search terms')
    parser.add_argument('--subreddits', nargs='+', default=["environment", "climate", "sustainability", "climateskeptics"], help='List of subreddits (space separated)')
    parser.add_argument('--viewpoint', default=None, help='"Left Wing", "Right Wing"')
    parser.add_argument('--emotion', default=None, help='Happy, Sad, Angry, Fearful, Suprised, etc')
    parser.add_argument('--sentiment', default=None, help='Supportive, Critical, Sceptical')
    parser.add_argument('--style', default=None, help='Humorous, Sarcastic, Serious')


    args = parser.parse_args()
    
    if args.action == 'create':
        # Scrapes data, classifies it, and creates/populates a vector database.
        search_terms = args.search
        search_subreddits=args.subreddits
        vdb_path=args.vdb
        if args.delete_old_vdb:
            log_comment("Deleting and recreating the vector database.", file_path='./log.txt')
        else:
            log_comment("Appending to existing vector database.", file_path='./log.txt')
        log_comment("Scrape, classify and create vector database  ", file_path='./log.txt')
        log_comment("search_terms="+search_terms, file_path='./log.txt')
        log_comment("vector_db="+vdb_path, file_path='./log.txt')

        input_list = my_reddit_scrape(
            search_terms=search_terms,
            search_subreddits=search_subreddits, 
            posts_limit=args.posts_limit,
            comments_limit=args.comments_limit
        )
        
        log_comment("processing input_list of size="+str(len(input_list)), file_path='./log.txt')
        all_classified = process_chunks(app, input_list, chunk_size=3)

        # --- Deduplicate against existing vector DB ---
        existing_items = getAllDBItems(vdb_path=vdb_path)
        existing_keys = set()

        if not existing_items.empty:
            # Safely get 'Post' column, defaulting to a series of empty strings if not present or converting NaNs
            if 'Post' in existing_items.columns:
                posts = existing_items['Post'].fillna('').astype(str)
            else:
                posts = pd.Series([''] * len(existing_items), index=existing_items.index, dtype=str)

            # Safely get 'Comment' column, defaulting to a series of empty strings if not present or converting NaNs
            if 'Comment' in existing_items.columns:
                comments = existing_items['Comment'].fillna('').astype(str)
            else:
                comments = pd.Series([''] * len(existing_items), index=existing_items.index, dtype=str)
            
            existing_keys = set(posts + comments)

        new_items = []
        for item in all_classified:
            post = item.get("Post", "")
            comment = item.get("Comment", "")
            key = post + comment
            if key not in existing_keys:
                new_items.append(item)
        log_comment(f"Filtered out {len(all_classified) - len(new_items)} duplicates. Keeping {len(new_items)} new items.", file_path='./log.txt')



        # log_comment(
        #         "all_classified="+ str(len(all_classified)),
        #         file_path='./log.txt')
        
        add_to_vector_db_with_metadata(all_classified, vdb_path=vdb_path, delete_old=args.delete_old_vdb)
        log_comment("created vdb at " + vdb_path, file_path='./log.txt')

    elif args.action == 'scrape':
        # Scrapes data from Reddit, classifies it, and saves the classified data to a JSON file.
        log_comment("Scrape and classify ", file_path='./log.txt')
        search_terms = args.search
        search_subreddits=args.subreddits

        input_list = my_reddit_scrape(
            search_terms=search_terms,
            search_subreddits=search_subreddits, 
            posts_limit=args.posts_limit,
            comments_limit=args.comments_limit
        )

        with open('./data/'+args.search+'_posts_comments.json', 'w') as f:
             json.dump(input_list, f)

        log_comment("search_terms="+search_terms, file_path='./log.txt')
        log_comment("input_list="+str(len(input_list)), file_path='./log.txt')

        all_classified = process_chunks(app, input_list, chunk_size=3)

        log_comment(
                "all_classified="+ str(len(all_classified)),
                file_path='./log.txt')

        with open('./data/'+args.json, 'w') as f:
            json.dump(all_classified, f, indent=4)
        log_comment(f"Saved all_classified data to ./data/{args.json}", file_path='./log.txt')


    elif args.action =='chat':
        # Allows interactive chat using a specified vector database, filtering by class and classification.
        vdb_path = args.vdb
        log_comment(
            f"Chat using {vdb_path} vector database",
            file_path='./log.txt'
        )

        # Create the chain
        my_chain = create_chain3(vdb_path=vdb_path, llm=use_llm)

        # Set the filter criteria for the retriever
        filter_criteria = {}
        if args.viewpoint:
            filter_criteria["Viewpoint"] = args.viewpoint.title().strip()
        if args.sentiment:
            filter_criteria["Sentiment"] = args.sentiment.title().strip()
        if args.emotion:
            filter_criteria["Emotion"] = args.emotion.title().strip()
        if args.style:
            filter_criteria["Style"] = args.style.title().strip()

        filter_criteria = convert_to_chroma_filter(filter_criteria)

        log_comment(
            f"Applying retriever filter: {filter_criteria}",
            file_path='./log.txt'
        )

        my_chain.retriever.base_retriever.search_kwargs["filter"] = filter_criteria

        # Interactive loop
        print("Enter your query (or type 'quit' to exit):")
        while True:
            user_input = input("Enter your post>> ").strip()
            if user_input.lower() in {"quit", "exit"}:
                print("Goodbye!")
                break
            if not user_input:
                continue

            try:
                result = my_chain.invoke(
                    {"query": user_input},
                    config={"callbacks": [MyCustomCallbackHandler()]}
                )
                print("\nResponse:\n", result.get("result", result))
            except Exception as e:
                print("Error during query:", e)

    elif args.action == 'pull':
        # Pulls all items from a specified vector database and saves them to CSV and Excel files.
        log_comment("Pull items from database ", file_path='./log.txt')
        vdb_path = args.vdb
        df = getAllDBItems(vdb_path=vdb_path)
        df.to_excel("chroma_vector_data_filtered.xlsx", index=False)  # For Excel
        df.to_csv("chroma_vector_data_filtered.csv", index=False) 
        log_comment(f"Data pulled from {vdb_path} and saved to chroma_vector_data_filtered.xlsx/csv", file_path='./log.txt')

    elif args.action == 'evaluate':
        # Evaluates the performance of the RAG system using a predefined test set and metrics.
        log_comment("Evaluate the RAG system ", file_path='./log.txt')

        # Set the filter criteria for evaluation
        additional_filter_criteria = {}
        if args.viewpoint:
            additional_filter_criteria["Viewpoint"] = args.viewpoint.title().strip()
        if args.sentiment:
            additional_filter_criteria["Sentiment"] = args.sentiment.title().strip()
        if args.emotion:
            additional_filter_criteria["Emotion"] = args.emotion.title().strip()
        if args.style:
            additional_filter_criteria["Style"] = args.style.title().strip()

        evalstr=evaluate_data(vdb_path=args.vdb, llm_name=args.llm_name, k=args.k, temperature=args.t, search_type=args.s, metrics_filename=args.out_file, sample_size=args.samples, additional_filter_criteria=additional_filter_criteria)

        print(evalstr)
        
        log_comment(f"Evaluation complete. Results saved to {args.out_file}.xlsx and associated summary statistics file.", file_path='./log.txt')

if __name__ == "__main__":
    main()