from icecream import ic
from dotenv import load_dotenv
import uvicorn
import os


if __name__ == "__main__":
    log_dir = os.path.abspath("./app/log_conf/log.ini")
    ic(log_dir)
    load_dotenv()
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False, access_log=True, log_config=log_dir)
