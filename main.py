from fastapi import FastAPI
from service.routes import predict, chat

app = FastAPI(title="AI Farmer Advisory System")

app.include_router(predict.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
