from dotenv import load_dotenv
load_dotenv(override=True)

import pytest
from fastapi.testclient import TestClient
from app.main import app
from icecream import ic
import logging


client = TestClient(app)

def test_chat():
    response = client.get("/chat/completion?query=\"테스트메세지\"")
    assert response.status_code == 200