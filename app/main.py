from typing import Union
from fastapi import FastAPI 
from fastapi.middleware.cors import CORSMiddleware
from icecream import ic
from app.routers import chat_router
import uvicorn
import os


app = FastAPI()
app.include_router(chat_router.router)

origins = [
    "http://localhost:8000",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


if __name__ == "__main__":
   log_dir = os.path.abspath("./app/log_conf/log.ini")
   ic(log_dir)
   uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, access_log=True, log_config=log_dir, reload_dirs=["app/routers/chat.py"])
