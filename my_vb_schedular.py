# by Ian Drumm

# This is helper application to populate vector database based on popular search terms
# runs enviro_bot_agents.py create at regular intervals with different searches to account for Reddit rate limits
# recommend you use local llms and a tmux to run in background
# tmux new -s mysession   or   tmux attach -t mysession
# python my_vb_schedular.py
# Scheduler with profiling and Chroma vector DB status reporting



import os
import subprocess
import time
import random
from datetime import datetime
import chromadb
from tools.my_utils_f import log_comment

# --- Configuration ---
VDB_NAME = "./vdb/climate_rag_vdb"
LOCK_FILE = "scraper.lock"
TIME_BETWEEN_CALLS = 70  # seconds
MAX_PER_RUN = 10         # max (term, subreddit) combinations per run

# --- Search terms ---
SEARCH_TERMS = [
    "climate change", "carbon emissions", "global warming", "fossil fuels",
    "carbon capture", "net zero", "renewable energy", "greenhouse gases",
    "recycling", "electric vehicles", "climate policy", "eco-anxiety",
    "IPCC report", "El Niño", "wildfires", "flooding",
    "biodiversity", "ecosystems", "deforestation", "habitat loss",
    "whales", "polar bears", "birds", "bees", "marine life", "wildlife conservation",
    "oceans", "coral reefs", "rainforests", "wetlands", "desertification",
    "wind", "water scarcity", "rivers", "air quality",
    "plastic pollution", "microplastics", 
    "oil spills", "overfishing",
    "climate justice", "climate strike", "green new deal", "energy transition",
    "sustainable development", "solar power", "wind power", "EV charging",
    "geoengineering", "climate hoax", "climate scepticism", "climate alarmism",
    "carbon footprint", "zero waste", "environmental activism",
]

# --- Subreddits ---
SUBREDDITS = [
    "environment", "climate", "sustainability", "Conservative"
]

# --- Locking logic ---
def is_locked():
    return os.path.exists(LOCK_FILE)

def set_lock():
    with open(LOCK_FILE, "w") as f:
        f.write("locked")

def release_lock():
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)

# --- Run subprocess and report size and document count ---
def run_scrape(search, subreddit):
    args = [
        "python", "enviro_bot_agents.py", "create",
        f"--search={search}",
        f"--vdb={VDB_NAME}",
        f"--s={subreddit}"
    ]
    start_time = time.time()
    log_comment(f"[{datetime.now().strftime('%H:%M:%S')}] Running: {' '.join(args)}", file_path='./log.txt')
    subprocess.run(args)
    elapsed = time.time() - start_time

    # Get disk usage
    try:
        size_output = subprocess.check_output(['du', '-sh', VDB_NAME])
        size_str = size_output.decode().split()[0]
    except Exception as e:
        size_str = f"Error measuring size: {e}"

    # Get record count from Chroma
    try:
        client = chromadb.PersistentClient(path=VDB_NAME)
        collections = client.list_collections()
        total_count = 0
        for coll in collections:
            c = client.get_collection(coll.name)
            count = c.count()
            total_count += count
        count_str = f"{total_count} records across {len(collections)} collections"
    except Exception as e:
        count_str = f"Error counting records: {e}"

    log_comment(f"[{datetime.now().strftime('%H:%M:%S')}] Finished in {elapsed:.2f} sec | VDB size: {size_str} | {count_str}", file_path='./log.txt')

# --- Main scheduler logic ---
def main():
    if is_locked():
        log_comment(f"[{datetime.now().strftime('%H:%M:%S')}] Another instance is already running. Exiting.", file_path='./log.txt')
        return

    try:
        set_lock()
        overall_start = time.time()

        all_combinations = [(term, sub) for term in SEARCH_TERMS for sub in SUBREDDITS]
        #random.shuffle(all_combinations)

        for i, (term, subreddit) in enumerate(all_combinations[:MAX_PER_RUN]):
            run_scrape(term, subreddit)
            if i < MAX_PER_RUN - 1:
                time.sleep(TIME_BETWEEN_CALLS)

        total_time = time.time() - overall_start
        log_comment(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ All done in {total_time:.2f} seconds.", file_path='./log.txt')

    finally:
        release_lock()

if __name__ == "__main__":
    main()
