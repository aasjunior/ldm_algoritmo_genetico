from datetime import datetime

def current_time():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def generate_log(error, traceback_details):
    with open('error_log.txt', 'w') as file:
        file.write(f'[{current_time()}] Ocorreu um erro:\n{error}\n')
        file.write(traceback_details)