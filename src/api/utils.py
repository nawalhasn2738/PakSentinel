import logging
from logging.handlers import RotatingFileHandler
import os
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

def setup_logging():
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger("PakSentinel")
    logger.setLevel(logging.INFO)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    fh = RotatingFileHandler("logs/api.log", maxBytes=5*1024*1024, backupCount=3)
    fh.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)
        
    return logger

limiter = Limiter(key_func=get_remote_address)
