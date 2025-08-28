from fastapi import APIRouter
from pydantic import BaseModel
from app.services.chatbot_service import get_advice

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    context: dict = None

@router.post("/farmer-query/")
def farmer_query(request: QueryRequest):
    response = get_advice(request.query, request.context)
    return {"advice": response}
