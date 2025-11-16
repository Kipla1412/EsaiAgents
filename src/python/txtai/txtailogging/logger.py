import logging
import os
from datetime import datetime

def get_logger(name="Agent", log_dir ="logs"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        os.makedirs(log_dir, exist_ok=True) # change the dir using config file , file fixed , fixed directory
        
        log_file = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y-%m-%d')}.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter) # using class intialiaze (singleton) ,use factory design 
        logger.addHandler(file_handler)


        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console.setFormatter(fmt)
        logger.addHandler(console)
    
    return logger
