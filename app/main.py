from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import chat_router

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


