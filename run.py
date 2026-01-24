import os

from dotenv import load_dotenv
from icecream import ic
import uvicorn

if __name__ == "__main__":
    log_dir = os.path.abspath("config/logging/log.ini")
    load_dotenv(override=True)
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False, access_log=True, log_config=log_dir)
