import config
def debug(message):
    if config.DEBUG:
        print(f"DEBUG: {message}")