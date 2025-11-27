import logging
import os

log_dirs = "logs"
os.makedirs(log_dirs, exist_ok=True)

logfile = os.path.join(log_dirs, "app.log")

logging.basicConfig(
   level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[
        logging.FileHandler(logfile, encoding="utf-8"),
        logging.StreamHandler()
        ]
    
)


logger = logging.getLogger("nephrology_app")