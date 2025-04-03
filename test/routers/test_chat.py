from fastapi.testclient import TestClient
from app.main import app
from icecream import ic

client = TestClient(app)

def test_chat():
    response = client.get("/chat")
    assert response.status_code == 200