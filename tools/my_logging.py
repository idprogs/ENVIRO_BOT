# Code by Ian Drumm
from datetime import datetime

def my_log(comment, file_path='./log.txt'):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] {comment}\n")
    with open(file_path, 'a') as log_file:
        log_file.write(f"[{current_time}] {comment}\n")