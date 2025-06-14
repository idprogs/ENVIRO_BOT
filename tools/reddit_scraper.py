# Code by Ian Drumm
import json
import os
import praw
import csv
import pandas as pd
import re
from datetime import datetime
import datetime

def my_log(comment, file_path='./log.txt'):
    """Logs a comment with a timestamp to a specified file and prints it."""
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] {comment}\n")
    with open(file_path, 'a') as log_file:
        log_file.write(f"[{current_time}] {comment}\n")

data_dir = '../datasets/environment/'



class RedditScraper:
    name: str = "Reddit Scraper"
    """A class to scrape posts and comments from Reddit based on search terms and subreddits."""
    description: str = "Fetches reddit posts and comments based on a search term."
    
    def __init__(self, search_subreddits=["all"], posts_limit=3, comments_limit=5):
        self.search_subreddits = search_subreddits
        self.posts_limit = posts_limit
        self.comments_limit = comments_limit

    def search(self, search_term, verbose=False):
        print("searching reddit")
        my_log(f"Searching Reddit for '{search_term}' in subreddits: {', '.join(self.search_subreddits)}", file_path='./log.txt')
        reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent='MyScraper1',
            username=os.getenv('REDDIT_USERNAME'),
            password=os.getenv('REDDIT_PASSWORD')
        )

        # Check login status
        try:
            username = reddit.user.me()
            print(f"Login successful! Logged in as: {username}")
        except Exception as e:
            print("Login failed:", e)
            username = reddit.user.me()
            print("my reddit username", username)
        
        # Optional: Test search query and check rate limit
        try:
            results = reddit.subreddit('all').search('climate change', limit=1)
            for post in results:
                print(f"Title: {post.title} | Upvotes: {post.score}")
            print("Remaining requests until rate limit:", reddit.auth.limits['remaining'])
        except Exception as e:
            print("Error during search or rate limit check:", e)

        # Define search parameters
        sort = 'new'
        syntax = 'lucene'
        columns = ['Subreddit', 'Post', 'Comment']
        df_out = pd.DataFrame(columns=columns)
        
        for sub_name in self.search_subreddits:
            subreddit = reddit.subreddit(sub_name)
            search_results = subreddit.search(search_term, limit=self.posts_limit, sort=sort, syntax=syntax)
            
            i = 1
            for submission in search_results:
                j = 1
                if verbose:
                    print(f"{i} of {self.posts_limit} Title: {submission.title} (Subreddit: {sub_name})\n")
                submission.comments.replace_more(limit=self.comments_limit)  # Load all comments; limit=0 means no limit
                
                for comment in submission.comments.list():
                    if j > self.comments_limit:
                        break

                    sm_post = f"{submission.title} {submission.selftext}"
                    sm_post = sm_post.replace("\n", " ")

                    sm_comment = f"{comment.body}"
                    sm_comment = sm_comment.replace("\n", " ")

                    if self.invalid_string(input_string=sm_post) or self.invalid_string(input_string=sm_comment):
                        continue
                    
                    df_out.loc[len(df_out)] = [sub_name, sm_post, sm_comment]
                    if verbose:
                        print(f"{i},{j} of {self.comments_limit} \nSubreddit: {sub_name} \nPost: [{sm_post}] \nComment: [{sm_comment}]\n")

                    j += 1
                i += 1

        return df_out

    def sanitize_filename(self, filename=""):
        """Removes invalid characters from a filename string."""
        invalid_chars = r'[/\0\"\'\*]'
        sanitized_filename = re.sub(invalid_chars, '_', filename)
        return sanitized_filename
    
    def invalid_string(self, input_string=""):
        """Checks if a string is invalid (contains URL or is too long)."""
        url_pattern = re.compile(r'http[s]?://|www\.\S+')
        if url_pattern.search(input_string):
            return True

        if len(input_string) > 512:
            return True
        
        return False

def testCode():
    """Function to test the RedditScraper class."""
    reddit = RedditScraper(search_subreddits=["environment", "climate", "sustainability"], posts_limit=1000, comments_limit=10)
    results = reddit.search("wind farms")
    print(results)

if __name__ == "__main__":
    testCode()