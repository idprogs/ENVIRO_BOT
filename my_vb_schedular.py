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
    "climate change", "carbon emissions", "global warming", "fossil fuels",
    "carbon capture", "net zero", "renewable energy", "greenhouse gases",
    "recycling", "electric vehicles", "climate policy", "eco-anxiety",
    "IPCC report", "El Niño", "wildfires", "flooding"
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