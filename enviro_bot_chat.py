# Code by Ian Drumm
import streamlit as st
import os
import random
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from tools.my_utils_f import (
    create_chain3,
    MyCustomCallbackHandler,
)
import json
import argparse  # Import argparse

# Load environment variables from .env file if it exists
load_dotenv()

# Parse command-line arguments for vdb
parser = argparse.ArgumentParser(description="Run Streamlit app with specified vdb_path.")
parser.add_argument("--vdb", type=str, default="./vdb/climate_rag_vdb", help="Path to the vector database")
args = parser.parse_args()

# Set vdb_path from command-line argument or default
if "vdb_path" not in st.session_state:
    st.session_state.vdb_path = args.vdb

if "llm" not in st.session_state:
    st.session_state.llm = None

if "my_chain" not in st.session_state:
    st.session_state.my_chain = None
    st.session_state.chain_config_key = None # To track if config changes

if "messages" not in st.session_state:
    st.session_state.messages = []

def my_write2(txt):
    new_title = (
        '<p style="font-family:sans-serif; color:Black; font-size: 16px; font-weight: bold;">'
        + txt
        + "</p>"
    )
    st.markdown(new_title, unsafe_allow_html=True)

def get_or_create_rag_chain(vdb_path, llm_instance, current_filter_criteria):
    """
    Retrieves a cached RAG chain or creates a new one if configuration changes.
    """
    # Create a unique key based on relevant configuration to decide if chain needs recreation
    config_tuple = (
        vdb_path, 
        llm_instance.model if llm_instance else None, # Use a stable attribute of the llm
        json.dumps(current_filter_criteria, sort_keys=True) # Filter criteria might change retriever behavior
    )

    if st.session_state.chain_config_key != config_tuple or st.session_state.my_chain is None:
        print("DEBUG: Chain configuration changed or chain not initialized. Re-creating chain.")
        print(f"DEBUG: VDB Path: {vdb_path}")
        print(f"DEBUG: LLM Model: {llm_instance.model if llm_instance else 'None'}")
        print(f"DEBUG: Filter Criteria: {current_filter_criteria}")

        if llm_instance is None:
            st.error("LLM not initialized. Please select an LLM in the sidebar.")
            return None

        chain = create_chain3(vdb_path=vdb_path, llm=llm_instance)
        if chain:
            chain.retriever.base_retriever.search_kwargs["filter"] = current_filter_criteria
            print("DEBUG: Filter applied to new chain's retriever")
        st.session_state.my_chain = chain
        st.session_state.chain_config_key = config_tuple
    elif st.session_state.my_chain:
        # Ensure filter is up-to-date on existing chain if only filter changed (though key includes it)
        st.session_state.my_chain.retriever.base_retriever.search_kwargs["filter"] = current_filter_criteria
        print("DEBUG: Using cached chain. Filter updated.")
    return st.session_state.my_chain

def main():
    print("DEBUG: enviro_bot_chat.py - main() started", flush=True)

    handler = MyCustomCallbackHandler()

    # Sidebar for controls
    with st.sidebar:
        st.header("Configuration Panel")
        
        # Display VDB path
        st.sidebar.markdown(f"**VDB:** `{st.session_state.vdb_path}`")

        # Debug Mode checkbox
        debug_mode = st.checkbox("Debug Mode", value=False)
        
        llm_options = ["llama3:70b", "mistral-nemo"]
        llm_opt = st.selectbox("Select an LLM from the options below:", llm_options)
        print(f"DEBUG: LLM option selected: {llm_opt}", flush=True)
        st.session_state.llm = Ollama(model=llm_opt, temperature=0.0, callbacks=[handler])
        st.write("Using LLM " + llm_opt)
        print(f"DEBUG: Ollama LLM initialized with model: {llm_opt}", flush=True)

        # Viewpoint options and checkbox
        include_viewpoint = st.checkbox("Include Viewpoint in filter criteria", value=True)
        if include_viewpoint:
            viewpoint_options = ["Left Wing", "Right Wing"]
            vp_opt = st.selectbox("Select a viewpoint from the options below:", viewpoint_options)
        else:
            vp_opt = None

        # Emotion options and checkbox
        include_emotion = st.checkbox("Include Emotion in filter criteria", value=False)
        if include_emotion:
            emotion_options = [
                "Happy", "Sad", "Angry", "Surprised", "Fearful", "Disgusted", 
                "Excited", "Anxious", "Neutral", "Content", "Proud", 
                "Love", "Amused", "Disappointed", "Frustrated"
            ]
            em_opt = st.selectbox("Select an emotion from the options below:", emotion_options)
        else:
            em_opt = None

        # Sentiment options and checkbox
        include_sentiment = st.checkbox("Include Sentiment in filter criteria", value=False)
        if include_sentiment:
            sentiment_options = ["Supportive", "Critical", "Sceptical"]
            sentiment_opt = st.selectbox("Select a sentiment from the options below:", sentiment_options)
        else:
            sentiment_opt = None

        # Style options and checkbox
        include_style = st.checkbox("Include Style in filter criteria", value=False)
        if include_style:
            style_options = ["Humorous", "Sarcastic", "Serious"]
            style_opt = st.selectbox("Select a style from the options below:", style_options)
        else:
            style_opt = None

    # Display vector database path and filter criteria only if Debug Mode is enabled
    if debug_mode:
        st.write("Using vector database " + st.session_state.vdb_path)

    # Build filter criteria based on selected options
    filter_list = []

    if vp_opt is not None:
        filter_list.append({"Viewpoint": vp_opt})
    if em_opt is not None:
        filter_list.append({"Emotion": em_opt})
    if sentiment_opt is not None:
        filter_list.append({"Sentiment": sentiment_opt})
    if style_opt is not None:
        filter_list.append({"Style": style_opt})

    if len(filter_list) == 1:
        filter_criteria = filter_list[0]
    elif len(filter_list) > 1:
        filter_criteria = {"$and": filter_list}
    else:
        filter_criteria = {}

    # Display filter criteria only if Debug Mode is enabled
    if debug_mode:
        st.write("Filter Criteria: " + json.dumps(filter_criteria))
    print(f"DEBUG: Filter criteria built: {json.dumps(filter_criteria)}", flush=True)

    # Set avatar images based on viewpoint
    you = "./person.png"
    if vp_opt == "Left Wing":
        person = "./hippy.png"
    elif vp_opt == "Right Wing":
        person = "./redneck.png"
    else:
        person = "./default_avatar.png"

    prompt_template_string = """Your role is to provide a single one-sentence comment 
            of less than 100 words in response to a post within the pst markup tags. 
            When formulating your comment, use only the information given between 
            the ctx markup tags, which is a JSON of real posts and corresponding real comments. 
            Give your comment in the typical style, language, and sentiment of the comments 
            in the context, aligning with the prevailing viewpoint, emotion, sentiment and style. Do not simply repeat 
            an existing comment. If the new post is unrelated to the context, 
            respond with 'I don't know'. 
            <ctx>{context}</ctx>, 
            <pst>{question}</pst>, 
            Your output must be just the comment."""
    
    # Get or create the cached chain
    my_chain = get_or_create_rag_chain(
        st.session_state.vdb_path, 
        st.session_state.llm, 
        filter_criteria
    )

    if my_chain is None: # If chain creation failed in the helper
        st.warning("RAG chain could not be initialized. Please check settings and try again.")
        return # Stop further execution if chain is not available
    with st.chat_message("assistant", avatar=person):
        my_write2("Hello, make a post about climate change")
    print("DEBUG: Initial assistant message displayed", flush=True)

    if question := st.chat_input("Type your question here:"):
        print(f"DEBUG: User entered question: {question}", flush=True)
        try:
            print("DEBUG: About to call my_chain.run(question)", flush=True)
            result = my_chain.run(question)
            print(f"DEBUG: my_chain.run() completed. Result: {result}", flush=True)
            st.session_state.messages.append(
                {"role": "assistant", "content": result, "avatar": person}
            )
            st.session_state.messages.append({"role": "user", "content": question})
            print("DEBUG: Messages appended to session state", flush=True)

            for message in reversed(st.session_state.messages):
                avatar = message["avatar"] if message["role"] == "assistant" else you
                with st.chat_message(message["role"], avatar=avatar):
                    my_write2(message["content"])
        except Exception as e:
            st.error(f"An error occurred: {e}")
            print(f"ERROR: Exception during chain execution or message display: {e}", flush=True)

if __name__ == "__main__":
    main()
