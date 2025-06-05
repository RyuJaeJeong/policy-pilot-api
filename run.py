from dotenv import load_dotenv
from icecream import ic
import os
import uvicorn

# .venv\Scripts\activate

if __name__ == "__main__":
    log_dir = os.path.abspath("app/logging/log.ini")
    ic(log_dir)
    load_dotenv(override=True)
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False, access_log=True, log_config=log_dir)
