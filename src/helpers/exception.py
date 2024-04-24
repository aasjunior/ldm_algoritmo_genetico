def generate_log(error, traceback_details):
    with open('error_log.txt', 'a') as file:
        file.write(f'Ocorreu um erro:\n{error}\n')
        file.write(traceback_details)