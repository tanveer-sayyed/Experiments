import logging
logger = logging.getLogger('main')
fileHandler = logging.FileHandler('/var/my_logs')
formatter = logging.Formatter('%(asctime)s %(message)s')
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)
logger.setLevel(logging.INFO)
def log_this_error(e):
    global logger
    logger.exception(str(e))
