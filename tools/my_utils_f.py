# Code by Ian Drumm
from PIL import Image
import io
from neo4j import GraphDatabase
import pandas as pd
import json
import os
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
import operator
import random
import os
from dotenv import load_dotenv
from langgraph.graph import Graph
from langchain_community.llms import Ollama
from langchain_core.callbacks import CallbackManager
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from tools.reddit_scraper import RedditScraper
import random
from langchain.chains import RetrievalQA, LLMChain

import csv
import pandas as pd
from langchain_core.callbacks import CallbackManager
from langchain_core.callbacks import StreamingStdOutCallbackHandler
import praw
import streamlit as st
import sys
import json
from PIL import Image
import io
import pandas as pd
import json
import argparse

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.callbacks import CallbackManager
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from numba import jit, cuda
from langchain_community.document_loaders import JSONLoader
from langchain_core.messages.system import SystemMessage
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_experimental.graph_transformers.llm import UnstructuredRelation, examples
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder
from typing import List

from langchain_core.documents import Document
from typing import List
from langchain_core.retrievers import BaseRetriever


import os
import shutil
from tools.text_evaluation import TextEvaluation
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
import datetime
from langchain_community.graphs import Neo4jGraph
from yfiles_jupyter_graphs import GraphWidget
import re
import pprint


from langchain_core.callbacks import BaseCallbackHandler


def log_prompts(prompt, file_path='logPrompts.txt'):
    with open(file_path, 'a') as log_file:
        log_file.write(f"{prompt}\n")

def log_docs(record, file_path='logDocs.txt'):
    with open(file_path, 'a') as log_file:
        log_file.write(f"{record},\n")

def log_comment(comment, file_path='./log.txt'):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] {comment}\n")
    with open(file_path, 'a') as log_file:
        log_file.write(f"[{current_time}] {comment}\n")

def save_output(json_str, filename):
    data = json.loads(json_str)
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Data saved successfully as {filename}")

def save_graph(app):
    image_data = app.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API,)
    image = Image.open(io.BytesIO(image_data))
    image.save('./mermaid_diagram.png')

def add_json_to_file(data_to_add, file_path, start_empty=False):
    # Check if the file exists and if we should start with an empty file
    if os.path.exists(file_path) and not start_empty:
        # Read the existing large JSON file
        with open(file_path, 'r') as large_file:
            large_data = json.load(large_file)
    else:
        # Initialize an empty list if the file does not exist or we are starting empty
        large_data = []

    # If data_to_add is a string, parse it to a dictionary/list

    if isinstance(data_to_add, str):
        try:
            data_to_add = json.loads(data_to_add)
        except json.JSONDecodeError:
            print("Invalid JSON data_to_add")
            print("Data to add [" + data_to_add + "]")
            return

    # Concatenate the two lists
    combined_data = large_data + data_to_add

    # Write the combined data back to a JSON file
    with open(file_path, 'w') as combined_file:
        json.dump(combined_data, combined_file, indent=4)

    print(f"Combined JSON data has been saved to {file_path}")

def load_dataframe_from_excel(file_path):
    try:
        df = pd.read_excel(file_path)
        print(f"DataFrame loaded successfully from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading DataFrame from {file_path}: {e}")
        return None

from langchain_core.callbacks.base import BaseCallbackHandler


class MyCustomCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        # Initialize an empty DataFrame with columns for 'prompts' and 'response'
        #self.df = pd.DataFrame(columns=['prompts', 'response'])
        self.current_prompt = None  # To store the current prompt during the operation
        self.context = None  # To store the current prompt during the operation

    def on_llm_start(self, serialized, prompts, **kwargs):
        #print("LLM operation started with prompts:", prompts)
        # Assuming there's only one prompt, but can be adapted for multiple
        self.current_prompt = prompts[0] if isinstance(prompts, list) else prompts
        # self.context = re.findall(r'<ctx>(.*?)</ctx>', self.current_prompt)
        # log_prompts(self.context, file_path='logPrompts.txt')
        #log_prompts(self.current_prompt)
        log_prompts(prompts)


    def on_llm_end(self, response, **kwargs):
        #print("LLM operation ended with response:", response)
        # Add a new row with the prompt and response
        #new_row = {'prompts': self.current_prompt, 'response': response}
        #self.df = self.df.append(new_row, ignore_index=True)
        # Reset the current prompt after saving it
        self.current_prompt = None

    # def get_dataframe(self):
    #     # Returns the stored DataFrame
    #     return self.df
    def get_context(self):
        return self.context


from pydantic import Field

class RerankRetriever(BaseRetriever):
    base_retriever: BaseRetriever = Field(...)
    reranker: CrossEncoder = Field(...)
    top_k: int = Field(default=5)

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieves and reranks documents based on the query.

        :param query: The search query.
        :return: A list of reranked Document objects.
        """
        # Step 1: Initial Retrieval
        initial_docs = self.base_retriever.invoke(query)
        if not initial_docs:
            return []

        # Step 2: Prepare query-document pairs for reranking
        pairs = [(query, doc.page_content) for doc in initial_docs]

        # Step 3: Get relevance scores from the reranker
        scores = self.reranker.predict(pairs)

        # Step 4: Attach scores to documents and sort them
        scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)

        # Step 5: Select the top_k documents
        ranked_docs = [doc for doc, score in scored_docs[:self.top_k]]

        return ranked_docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """
        Asynchronously retrieves and reranks documents based on the query.

        :param query: The search query.
        :return: A list of reranked Document objects.
        """
        # Step 1: Initial Retrieval
        initial_docs = await self.base_retriever.aget_relevant_documents(query)
        if not initial_docs:
            return []

        # Step 2: Prepare query-document pairs for reranking
        pairs = [(query, doc.page_content) for doc in initial_docs]

        # Step 3: Get relevance scores from the reranker
        scores = self.reranker.predict(pairs)

        # Step 4: Attach scores to documents and sort them
        scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)

        # Step 5: Select the top_k documents
        ranked_docs = [doc for doc, score in scored_docs[:self.top_k]]

        return ranked_docs


def getAllDBItems(
        vdb_path: str = "./vectorDBGraph"
):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=vdb_path, embedding_function=embeddings)

    # Fetch all items
    items = vectordb.get()
    
    # Check that the keys exist and are populated with data
    if not all(key in items for key in ['ids', 'embeddings', 'documents', 'metadatas']):
        print("Some expected keys are missing in the items dictionary.")
        return pd.DataFrame()


    # Create the initial DataFrame with document and metadata fields
    df = pd.DataFrame({
        "document": items["documents"],  # Original text
#        "metadata": items["metadatas"]   # Metadata associated with each document
    })

    # Parse 'document' JSON strings into dictionaries
    df["document"] = df["document"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

    # Expand 'document' JSON into separate columns (Post and Comment)
    document_df = pd.json_normalize(df["document"])
    df = df.drop("document", axis=1).join(document_df)  # Drop document column and join expanded fields

    return df



def create_chain3(
    vdb_path: str,
    llm=None,
    search_type: str = "mmr",
    k: int = 3,
    prompt_template_string = None,
):
    print("DEBUG: my_utils_f.py - create_chain3 called")
    # Step 1: Set up embeddings and vector database
    print("DEBUG: create_chain3 - Initializing SentenceTransformerEmbeddings")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    print("DEBUG: create_chain3 - SentenceTransformerEmbeddings initialized")
    print(f"DEBUG: create_chain3 - Initializing Chroma with vdb_path: {vdb_path}")
    vectordb = Chroma(persist_directory=vdb_path, embedding_function=embeddings)
    print("DEBUG: create_chain3 - Chroma initialized")

    # Step 2: Set up the retriever with dynamic filtering
    if search_type == "mmr":
        print("DEBUG: create_chain3 - Setting up MMR base_retriever")
        base_retriever = vectordb.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k * 2}
        )
    else:
        print("DEBUG: create_chain3 - Setting up similarity_score_threshold base_retriever")
        base_retriever = vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.5, "k": k * 2}
        )

    reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
    rerank_retriever = RerankRetriever(
        base_retriever=base_retriever,
        reranker=reranker_model,
        top_k=k
    )
    print("DEBUG: create_chain3 - RerankRetriever initialized")

    # Step 3: Define the prompt template
    if prompt_template_string == None:
        print("DEBUG: create_chain3 - Using default prompt_template_string")
        prompt_template_string = """Your role is to provide a single one-sentence comment 
            of less than 100 words in response to a post within the pst markup tags. 
            When formulating your comment, use only the information given between 
            the ctx markup tags, which is a JSON of real posts and corresponding real comments. 
            Give your comment in the typical style, language, and sentiment of the comments 
            in the context, aligning with the viewpoint. Do not simply repeat 
            an existing comment. If the new post is unrelated to the context, 
            respond with 'I don't know'. 
            <ctx>{context}</ctx>, 
            <pst>{question}</pst>, 
            Your output must be just the comment."""

    prompt = PromptTemplate(
        input_variables=["context", "question", "vp"],
        template=prompt_template_string
    )
    print("DEBUG: create_chain3 - PromptTemplate created")
    my_chain = RetrievalQA.from_llm( #from_chain_type(
        llm=llm,
        retriever=rerank_retriever,
        #retriever=base_retriever,
        prompt=prompt,
        verbose=True,
        return_source_documents=False,
    )
    print("DEBUG: create_chain3 - RetrievalQA chain initialized")
    print("Chain Created with Reranking") # Existing print
    return my_chain








def create_chain2(
    vdb_path: str = "./vectorDBGraph",
    llm=None,
    search_type: str = "mmr",
    k: int = 3,
    classification="Left Wing",
    prompt_template_string: str = """Your role is to provide a single one-sentence comment of less than 100 words in response to a post within the pst markup tags. When formulating your comment use only the information given between the ctx markup tags which is a JSON of real posts and corresponding real comments. Give your comment in the typical style, language and sentiment of the comments in the context. Though do not just repeat an existing comment. If the new post is unrelated to the context, simply respond with 'I don't know'. <ctx>{context}</ctx>, <pst>{question}</pst>, Your output must be just the comment."""
) -> RetrievalQA:

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=vdb_path, embedding_function=embeddings)
    filter_criteria = {
        "classification": classification,
        #"emotion": emotion  # Will be ignored if None
    }

    filter = {"classification": classification}
    if search_type == "mmr":
        base_retriever = vectordb.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k * 2, "filter": filter}
        )
    else:
        base_retriever = vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.5, "k": k * 2}  # Adjust k as needed
        )
    reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
    rerank_retriever = RerankRetriever(
        base_retriever=base_retriever,
        reranker=reranker_model,
        top_k=k
    )
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template_string
    )

    my_chain = RetrievalQA.from_llm( #from_chain_type(
        llm=llm,
        retriever=rerank_retriever,
        #retriever=base_retriever,
        prompt=prompt,
        verbose=True,
        return_source_documents=False,
    )

    print("Chain Created with Reranking")
    return my_chain




def create_chain(
    vdb_path: str = "./vectorDBGraph",
    llm=None,
    search_type: str = "mmr",
    k: int = 1,
    prompt_template_string: str = """Your role is to provide a single one-sentence comment of less than 100 words in response to a post within the pst markup tags. When formulating your comment use only the information given between the ctx markup tags which is a JSON of real posts and corresponding real comments. Give your comment in the typical style, language and sentiment of the comments in the context. Though do not just repeat an existing comment. If the new post is unrelated to the context, simply respond with 'I don't know'. <ctx>{context}</ctx>, <pst>{question}</pst>, Your output must be just the comment."""
) -> RetrievalQA:
    """
    Creates a RetrievalQA chain with reranking.

    :param vdb_path: Path to the vector database.
    :param llm: The language model to use.
    :param search_type: Type of search ('mmr' or 'similarity_score_threshold').
    :param k: Number of top documents to retrieve.
    :return: A RetrievalQA chain.
    """

    # Initialize embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Initialize Chroma vector database
    vectordb = Chroma(persist_directory=vdb_path, embedding_function=embeddings)

    # Create the base retriever
    if search_type == "mmr":
        base_retriever = vectordb.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k * 2}  # Retrieve more for better reranking
        )
    else:
        base_retriever = vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.5, "k": k * 2}  # Adjust k as needed
        )

    # Initialize a cross-encoder model for reranking
    reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

    # Initialize the RerankRetriever with the base retriever and reranker
    rerank_retriever = RerankRetriever(
        base_retriever=base_retriever,
        reranker=reranker_model,
        top_k=k
    )

    # Define the prompt template
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        #template="""Your role is to provide a single one-sentence comment of less than 100 words in response to a post delimited by <pst> and </pst>. When formulating your comment use only the information given in the context delimited by <ctx> and </ctx> which is a JSON of real posts and corresponding real comments. Give your comment in the typical style, language and sentiment of the comments in the context. Though do not just repeat an existing comment. If the new post is unrelated to the context, simply respond with 'I don't know'. <ctx>{context}</ctx>, <pst>{question}</pst>, Your output must be just the comment."""
        #template="""Your role is to provide a single one-sentence comment of less than 100 words in response to a post within the pst markup tags. When formulating your comment use only the information given between the ctx markup tags which is a JSON of real posts and corresponding real comments. Give your comment in the typical style, language and sentiment of the comments in the context. Though do not just repeat an existing comment. If the new post is unrelated to the context, simply respond with 'I don't know'. <ctx>{context}</ctx>, <pst>{question}</pst>, Your output must be just the comment."""
        template=prompt_template_string
    )

    # Create the RetrievalQA chain with the reranked retriever
    my_chain = RetrievalQA.from_llm( #from_chain_type(
        llm=llm,
        retriever=rerank_retriever,
        #retriever=base_retriever,
        prompt=prompt,
        verbose=True,
        return_source_documents=False,
    )

    print("Chain Created with Reranking")
    return my_chain

def my_reddit_scrape(search_terms="test",search_subreddits=["all"], posts_limit=3, comments_limit=2):
    log_comment("Scraping Reddit: search " + search_terms + " from " + ' '.join(search_subreddits),file_path='./log.txt')
    try:
        reddit_scraper = RedditScraper(search_subreddits=search_subreddits, posts_limit=posts_limit, comments_limit=comments_limit)
        df = reddit_scraper.search(search_terms)
        df.to_excel('./my_reddit.xlsx', index=False)
        return df.to_dict(orient='records')
    except Exception as e:
        return []

def some_llm_filter(item=None, llm=None):
    # Placeholder for actual LLM filter logic
    return True

def some_llm_classifier(item=None, llm=None):
    try:
        with open('./my_json_item_schema.json', 'r') as file:
            my_schema = file.read()
        jstr = json.dumps(item)
        str = "{\"Post\":\"something\", \"Comment\": \"something\" , \"Classify\": \"something\", \"Justification\": \"something\"}"
        prompt = f"Your task is to look at the following json {jstr}. \
            Look at the post and comment, Hence classify according to the opinion implicit in the comment. \
            Is the opinion 'Left Wing', 'Neutral', 'Right Wing'.\
            Follow this with a very brief justification for the classification. \
            Your output must be of the form {str}.\
            Return it directly without any explanations and code snippet formating!"
        res = llm.invoke(prompt)
        dictionary = json.loads(res)
        return dictionary.get("Classify")
    except json.JSONDecodeError as e:
        return None


from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

class MetadataModel(BaseModel):
    ViewPoint: str = Field(
        description="The viewpoint expressed",
        pattern="^(Left Wing|Neutral|Right Wing)$"
    )
    Emotion: str = Field(
        description="The emotion conveyed",
        pattern="^(Happy|Sad|Angry)$"
    )
    # Add additional fields as needed

class RedditDataModel(BaseModel):
    Post: str = Field(description="The Reddit post")
    Comment: str = Field(description="The Reddit comment")
    Metadata: MetadataModel = Field(description="Additional metadata about the post and comment")

from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate

class ViewpointDataModel(BaseModel):
    Viewpoint: str = Field(
        description="The viewpoint expressed in the comment",
        pattern="^(Left Wing|Neutral|Right Wing)$"
    )

def get_viewpoint(item=None, llm=None):
    # Create the prompt template
    prompt = PromptTemplate(
        template=(
            "Given the following Reddit post and comment, classify the viewpoint of the comment as 'Left Wing', 'Neutral', or 'Right Wing'.\n\n"
            "Post: {post}\n"
            "Comment: {comment}\n\n"
            "Please output only the classification."
        ),
        input_variables=["post", "comment"],
    )

    # Format the prompt with the actual variables
    prompt_text = prompt.format(
        post=item['Post'],
        comment=item['Comment']
    )

    # Invoke the LLM with the formatted prompt
    res = llm.invoke(prompt_text)

    # Strip any leading/trailing whitespace
    viewpoint = res.strip()

    # Normalize the output (optional)
    viewpoint = viewpoint.title().replace("-", " ")

    # Validate the output using the Pydantic model
    try:
        parsed_output = ViewpointDataModel(Viewpoint=viewpoint)
        return parsed_output.Viewpoint
    except Exception as e:
        print("Failed to parse the output:", e)
        return None


class EmotionDataModel(BaseModel):
    Emotion: str = Field(
        description="The emotion conveyed",
        pattern="^(Happy|Sad|Angry|Surprised|Fearful|Disgusted|Excited|Anxious|Neutral|Content|Proud|Love|Amused|Disappointed|Frustrated)$"
    )

def get_emotion(item=None, llm=None):
    prompt = PromptTemplate(
        template=(
            "Given the following Reddit post and comment, classify the emotion conveyed in the comment as one of the following:\n"
            "'Happy', 'Sad', 'Angry', 'Surprised', 'Fearful', 'Disgusted', 'Excited', 'Anxious', 'Neutral', 'Content', 'Proud', 'Love', 'Amused', 'Disappointed', or 'Frustrated'.\n\n"
            "Post: {post}\n"
            "Comment: {comment}\n\n"
            "Please output only the classification."
        ),
        input_variables=["post", "comment"],
    )
    prompt_text = prompt.format(post=item['Post'], comment=item['Comment'])
    res = llm.invoke(prompt_text)
    emotion = res.strip().title()
    try:
        parsed_output = EmotionDataModel(Emotion=emotion)
        return parsed_output.Emotion
    except Exception as e:
        print("Failed to parse the output:", e)
        return None



def some_llm_classifier_big(item=None, llm=None):
    parser = PydanticOutputParser(pydantic_object=RedditDataModel)
    fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

    format_instructions = fixing_parser.get_format_instructions()
    format_instructions += "\n\nPlease output only the JSON object and nothing else."

    prompt = PromptTemplate(
        template=(
            "Given the following Reddit post and comment, extract the metadata.\n\n"
            "Post: {post}\n"
            "Comment: {comment}\n\n"
            "{format_instructions}"
        ),
        input_variables=["post", "comment"],
        partial_variables={"format_instructions": format_instructions},
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    res = chain.run({
        'post': item['Post'],
        'comment': item['Comment']
    })

    # Use the fixing parser to parse the output
    try:
        parsed_output = fixing_parser.parse(res)
        jstr = parsed_output.model_dump_json(indent=2)
        print(jstr)
        exit(0)
        return jstr
    except Exception as e:
        print("Failed to parse the output:", e)
        return None



# Convert training set to List[str]
def convert_to_list_of_strings(items: list[dict[str, str]]) -> list[str]:
    return [f"Post: {item['post']} Comment: {item['comment']}" for item in items]



def add_to_vector_db_with_metadata(training_set, vdb_path, delete_old=False):
    print(f"Adding to vector database for RAG")
    try:
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        # Extract documents and metadata from the training set
        documents = []
        metadatas = []
        for record in training_set:
            # Convert each record to a JSON string for the document
            document = json.dumps({
                "Post": record.get('Post', ''),
                "Comment": record.get('Comment', ''),
                "Viewpoint": record.get('Viewpoint', ''),
                "Emotion": record.get('Emotion', ''),
                "Sentiment": record.get('Sentiment', ''),
                "Style": record.get('Style', ''),
                "Training_or_Test_Data": record.get('Training_or_Test_Data', '')
            })
            documents.append(document)

            # Extract the 'classification' field for metadata
            metadata = {
                "Viewpoint":  record.get('Viewpoint', ''),
                "Emotion":  record.get('Emotion', ''),
                "Sentiment": record.get('Sentiment', ''),
                "Style": record.get('Style', ''),
                "Training_or_Test_Data": record.get('Training_or_Test_Data', '')
            }
            metadatas.append(metadata)

        # Log the documents (optional)
        log_docs(documents)

        # Remove the old vector database if needed
        if os.path.exists(vdb_path) and os.path.isdir(vdb_path) and delete_old:
            shutil.rmtree(vdb_path)

        # Create the vector database with documents and metadata
        vectordb = Chroma.from_texts(
            texts=documents, 
            embedding=embeddings, 
            metadatas=metadatas, 
            persist_directory=vdb_path
        )

        print(f"Created vector database: {vdb_path}")
    except Exception as e:
        print(f"Creating vector database failed: {e}")

def add_to_vector_db(training_set, vdb_path, delete_old=False):
    print(f"Adding to vector database for RAG")
    try:
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        #vdb_path = f"./vdb/{vector_db_name}"
        #documents = convert_to_list_of_strings(training_set)

        #documents = [json.dumps(training_set)]
        documents = [json.dumps(record) for record in training_set]
        #log_docs(documents)
        #print("\n\nDOCS>>",documents,"<<\n")
        if os.path.exists(vdb_path) and os.path.isdir(vdb_path) and delete_old==True:
            shutil.rmtree(vdb_path)
        Chroma.from_texts(texts=documents, embedding=embeddings, persist_directory=vdb_path)
        print(f"Created vector database: {vdb_path}")
    except Exception as e:
        print(f"Creating vector database failed: {e}")



def replace_null_and_empty_strings(obj, replacement):
    if isinstance(obj, dict):
        return {k: replace_null_and_empty_strings(v, replacement) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_null_and_empty_strings(elem, replacement) for elem in obj]
    elif obj is None or obj == "":
        return replacement
    else:
        return obj

def showGraph(cypher, uri, username, password):
    # create a neo4j session to run queries
    driver = GraphDatabase.driver(
        uri = uri,
        auth = (username, password)
    )
    session = driver.session()
    widget = GraphWidget(graph = session.run(cypher).graph())
    widget.node_label_mapping = 'id'
    #display(widget)
    return widget


from langchain_core.prompts.chat import HumanMessagePromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate

def add_to_graph_db(training_set,
    graph_db_path, 
    delete_old=False, 
    url="bolt://localhost:7687", 
    username="neo4j", 
    password="",
    llm=None):

    print(f"Adding to graph database for RAG")

    system_prompt = """
    You are a data scientist working for a company that is building a knowledge graph database. 
    Your task is to extract information from data and convert it into a knowledge graph database.
    Provide a set of Nodes in the form [head, head_type, relation, tail, tail_type].
    It is important that the head and tail exists as nodes that are related by the relation. If you can't pair a relationship with a pair of nodes don't add it.
    When you find a node or relationship you want to add try to create a generic TYPE for it that describes the entity you can also think of it as a label.
    You must generate the output in a JSON format containing a list with JSON objects. Each object should have the keys: "head", "head_type", "relation", "tail", and "tail_type".
    """

    system_message = SystemMessage(content=system_prompt)
    parser = JsonOutputParser(pydantic_object=UnstructuredRelation)

    human_prompt = PromptTemplate(
        template="""
    Examples:
    {examples}

    For the following text, extract entities and relations as in the provided example.
    {format_instructions}\nText: {input}""",
        input_variables=["input"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
            "node_labels": None,
            "rel_types": None,
            "examples": examples,
        },
    )

    human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message, human_message_prompt]
    )

    try:
        graph = Neo4jGraph(url=url, username=username, password=password)
        llm_transformer = LLMGraphTransformer(llm=llm)
        #print(training_set)

        #jq_schema = '[.[] | {Subreddit: .Subreddit, Post: .Post, Comment: .Comment, classification: .classification}]'
        jq_schema = '.[]'

        loader = JSONLoader(file_path='/root/idwork/my_graph_fun/data/left_training_set.json', jq_schema=jq_schema, text_content=False)

        documents = loader.load()

        #print(data)


        #documents = [Document(page_content=data)]

        print(len(documents))

        graph_documents = llm_transformer.convert_to_graph_documents(documents)

        print("Add graph documents\n")

        print(graph_documents)

        graph.add_graph_documents(graph_documents)

        print(f"Creating graph database succeeded!")
    except Exception as e:
        print(f"Creating graph database failed: {e}")

def convert_to_chroma_filter(filter_dict):
    if not filter_dict:
        return {}
    clauses = []
    for key, value in filter_dict.items():
        if isinstance(value, list):
            clauses.append({key: {"$in": value}})
        else:
            clauses.append({key: {"$eq": value}})
    return {"$and": clauses}

def evaluate_data(
    vdb_path,
    llm_name="llama3:70b",
    temperature=0.3,
    k=1,
    search_type="mmr",
    sample_size=10,
    metrics_filename="default_metrics",
    additional_filter_criteria=None
):
    handler= MyCustomCallbackHandler()
    llm = Ollama(model=llm_name,temperature=temperature, callbacks=[handler])

    if additional_filter_criteria is None:
        additional_filter_criteria = {}

    # Base filter
    train_filter = {"Training_or_Test_Data": "Training"}
    test_filter = {"Training_or_Test_Data": "Test"}

    # Merge with additional criteria
    train_filter.update(additional_filter_criteria)
    test_filter.update(additional_filter_criteria)

    train_filter = convert_to_chroma_filter(train_filter)
    test_filter = convert_to_chroma_filter(test_filter)

    my_chain = create_chain3(vdb_path=vdb_path, llm=llm, k=k, search_type=search_type)
    my_chain.retriever.base_retriever.search_kwargs["filter"] = train_filter
    
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=vdb_path, embedding_function=embeddings)

    test_data_dictionary = vectordb.get(where=test_filter)

    #pprint.pprint(test_data_dictionary)


    test_data = []
    for meta, doc in zip(test_data_dictionary["metadatas"], test_data_dictionary["documents"]):
        try:
            doc_data = json.loads(doc)
            if "Post" in doc_data and "Comment" in doc_data:
                combined = {**meta, **doc_data}
                test_data.append(combined)
        except json.JSONDecodeError:
            continue

    #my_chain = create_chain(vdb_path, llm=llm, k=k, search_type=search_type)

    posts = []
    real_comments_to_post = []
    bot_comments_to_post = []
    contexts = []
    ragas_scores = []

    # sampled_data = test_data
    test_data_size = len(test_data)
    if len(test_data) < sample_size:
        sample_size = len(test_data)
    random.seed(42)
    sampled_data = random.sample(test_data, sample_size)

    test_count=0
    for item in sampled_data:
        query = item["Post"]
        result = my_chain({'query': query})
        posts.append(query)
        real_comments_to_post.append(item["Comment"])
        bot_comments_to_post.append(result['result'])
        contexts.append(handler.get_context())

    print("Real comments to post:", len(real_comments_to_post))
    print("Bot comments to post:", len(bot_comments_to_post))
    print("Sample post and comment pairs:")
    for i in range(min(3, len(real_comments_to_post))):
        print(f"{i+1}. Post: {posts[i]}")
        print(f"   Real: {real_comments_to_post[i]}")
        print(f"   Bot:  {bot_comments_to_post[i]}")


    textEvaluation=TextEvaluation()
    vals = textEvaluation.evaluate(real_comments_to_post, bot_comments_to_post)

    print("Vals " + str(vals))

    # print("\n\n\nReal and Bot comments\n")
    # for post_item, real_item, bot_item in zip(posts, real_comments_to_post, bot_comments_to_post):
    #     print(f"\n\nPost:[{post_item}]\nReal:[{real_item}],\nBot:[{bot_item}]")

    # Prepare the data for the DataFrame
    combined_data = []

    for i, post in enumerate(posts):
        val = vals[i % len(vals)]  # Use modulo to cycle through vals if len(posts) > len(vals)
        combined_item = {
            'Post': post,
            'Real Comment': real_comments_to_post[i],
            'Bot Comment': bot_comments_to_post[i],
            #'Contexts': contexts[i],
            #'Ragas Scores': ragas_scores[i],
            'Comment_BLEU': val['Comment_BLEU'],
            'Comment_ROUGE_rouge1_precision': val['Comment_ROUGE']['rouge1'].precision,
            'Comment_ROUGE_rouge1_recall': val['Comment_ROUGE']['rouge1'].recall,
            'Comment_ROUGE_rouge1_fmeasure': val['Comment_ROUGE']['rouge1'].fmeasure,
            'Comment_ROUGE_rougeL_precision': val['Comment_ROUGE']['rougeL'].precision,
            'Comment_ROUGE_rougeL_recall': val['Comment_ROUGE']['rougeL'].recall,
            'Comment_ROUGE_rougeL_fmeasure': val['Comment_ROUGE']['rougeL'].fmeasure,
            'BERT_P': val['BERT_P'],
            'BERT_R': val['BERT_R'],
            'BERT_F': val['BERT_F'],
            'Embedding_Similarity': val['embedding_similarity'],
            'Comment_Meteor': val['Comment_METEOR'],
            'Real_Perplexity': val['real_perplexity'],
            'Gen_Perplexity': val['gen_perplexity'],
            'real_sentiment_intensity' : val['real_sentiment_intensity'],
            'gen_sentiment_intensity' : val['gen_sentiment_intensity'],
            'Emotional_Similarity': val['emotional_similarity'],
            'real_anger': val['real_anger'],
            'real_anticipation': val['real_anticipation'],
            'real_disgust': val['real_disgust'],
            'real_fear': val['real_fear'],
            'real_joy': val['real_joy'],
            'real_sadness': val['real_sadness'],
            'real_surprise': val['real_surprise'],
            'real_trust': val['real_trust'],
            'gen_anger': val['gen_anger'],
            'gen_anticipation': val['gen_anticipation'],
            'gen_disgust': val['gen_disgust'],
            'gen_fear': val['gen_fear'],
            'gen_joy': val['gen_joy'],
            'gen_sadness': val['gen_sadness'],
            'gen_surprise': val['gen_surprise'],
            'gen_trust': val['gen_trust'],
            'k': k,
            'temperature': temperature
        }
        combined_data.append(combined_item)



    # Create a DataFrame
    df = pd.DataFrame(combined_data)

    # Save the DataFrame to an Excel file
    df.to_excel(metrics_filename+'.xlsx', index=False)


    # List of columns you want to compute statistics for
    metric_columns = [
        'Comment_BLEU',
        'Comment_ROUGE_rouge1_precision',
        'Comment_Meteor',
        'BERT_P',
        'BERT_R',
        'BERT_F',
        'Embedding_Similarity',
        'Real_Perplexity',
        'Gen_Perplexity',
        'real_sentiment_intensity',
        'gen_sentiment_intensity',
        'Emotional_Similarity',
        'real_anger',
        'gen_anger',
        'real_anticipation',
        'gen_anticipation',
        'real_disgust',
        'gen_disgust',
        'real_fear',
        'gen_fear',
        'real_joy',
        'gen_joy',
        'real_sadness',
        'gen_sadness',
        'real_surprise',
        'gen_surprise',
        'real_trust',
        'gen_trust',
        'k',
        'temperature'
    ]

    # List of statistics you want to compute
    stats = ['median', 'max', 'min', 'mean', 'std']

    # Compute the summary statistics
    summary_stats = {}
    for column_name in metric_columns:
        # Compute the statistics and convert them to a dictionary
        metric_stats = df[column_name].agg(stats).to_dict()
        summary_stats[column_name] = metric_stats

    summary_df = pd.DataFrame(summary_stats)
    # summary_df.to_csv('summary_'+metrics_filename+'_k_'+str(k)+'.csv', index=True)
    filename=metrics_filename+'_k_'+str(k)+'_t_'+str(temperature)+'_'+search_type+'_'+llm_name+'_samples_'+str(sample_size)+'_n_'+str(test_data_size)+'.xlsx'
    summary_df.to_excel(filename, index=True)

    print("Written to file " + filename)

