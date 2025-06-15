# by Ian Drumm
# This is helper application to populate vector database based on popular search terms
# runs enviro_bot_agents.py create at regular intervals with different searches to account for Reddit rate limits
# recommend you use local llms and a tmux to run in background
# tmux new -s mysession   or   tmux attach -t mysession
# python my_vb_schedular.py


import os
import subprocess
import time
import random
from datetime import datetime

# --- Configuration ---
VDB_NAME = "./vdb/climate_rag_vdb"
LOCK_FILE = "scraper.lock"
TIME_BETWEEN_CALLS = 70  # seconds
MAX_PER_RUN = 10         # max (term, subreddit) combinations per run

# --- Search terms ---
SEARCH_TERMS = [
    # Core environmental/climate topics
    # "climate change", "carbon emissions", "global warming", "fossil fuels",
    # "carbon capture", "net zero", "renewable energy", "greenhouse gases",
    # "recycling", "electric vehicles", "climate policy", "eco-anxiety",
    # "IPCC report", "El Niño", "wildfires", "flooding",

    # Nature and biodiversity
    "biodiversity", "ecosystems", "deforestation", "habitat loss",
    "whales", "polar bears", "birds", "bees", "marine life", "wildlife conservation",
    "oceans", "coral reefs", "rainforests", "wetlands", "desertification",
    "wind", "water scarcity", "rivers", "air quality",

    # Pollution and degradation
    "plastic pollution", "microplastics", "oil spills", "overfishing"

    # Broader sustainability and scepticism spectrum
    "climate justice", "climate strike", "green new deal", "energy transition",
    "sustainable development", "solar power", "wind power", "EV charging",
    "geoengineering", "climate hoax", "climate scepticism", "climate alarmism",
    "carbon footprint", "zero waste", "environmental activism",
]

# --- Subreddits (left, right, centrist, contrarian) ---
SUBREDDITS = [
    "environment", "climate", "sustainability","Conservative"
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

# --- Run your app with selected arguments ---
def run_scrape(search, subreddit):
    args = [
        "python", "enviro_bot_agents.py", "create",
        f"--search={search}",
        f"--vdb={VDB_NAME}",
        f"--s={subreddit}"
    ]
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Running: {' '.join(args)}")
    subprocess.run(args)

# --- Main scheduler logic ---
def main():
    if is_locked():
        print("Another instance is already running. Exiting.")
        return

    try:
        set_lock()
        all_combinations = [(term, sub) for term in SEARCH_TERMS for sub in SUBREDDITS]
        random.shuffle(all_combinations)

        for i, (term, subreddit) in enumerate(all_combinations[:MAX_PER_RUN]):
            run_scrape(term, subreddit)
            if i < MAX_PER_RUN - 1:
                time.sleep(TIME_BETWEEN_CALLS)

    finally:
        release_lock()

if __name__ == "__main__":
    main()