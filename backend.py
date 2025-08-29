import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import requests
from models.prediction import PredictionPipeline

# --- LangChain / OpenAI ---
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langdetect import detect
from dotenv import load_dotenv
load_dotenv()

# --- FastAPI App ---
app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Chatbot Memory ---
memory = ConversationBufferMemory(memory_key="chat_history")

# --- Chat Model ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)

prompt_template = PromptTemplate(
    input_variables=["user_input", "weather_info", "location"],
    template="""
You are an AI farmer advisor. Answer in the same language as the user query.  
Consider the weather information: {weather_info}  

Conversation so far: {chat_history}  
User: {user_input}  
AI:"""
)

chat_chain = LLMChain(llm=llm, prompt=prompt_template, memory=memory)

# --- Pydantic Models ---
class FarmerQuery(BaseModel):
    query: str
    city: str = "Thiruvananthapuram"  # default location
    lang: str = "en"

# --- Disease Prediction API ---
@app.post("/api/predict-disease/")
async def predict_disease(file: UploadFile = File(...)):
    try:
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        pipeline = PredictionPipeline(temp_file)
        result = pipeline.predict()

        os.remove(temp_file)

        return {"prediction": result[0]["image"], "probabilities": result[0]["probabilities"]}

    except Exception as e:
        return {"error": str(e)}
import requests

def get_user_location():
    try:
        token = os.getenv("IP_INFO")
        url = f"https://ipinfo.io/json?token={token}" if token else "https://ipinfo.io/json"
        res = requests.get(url, timeout=5)
        data = res.json()
        city = data.get("city")
        if city:
            return city
    except Exception as e:
        print("Location detection failed:", e)
    return "Thiruvananthapuram"

    
    
    
def get_weather(city, lang="en"):
    WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric&lang={lang}"
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            temp = data["main"]["temp"]
            desc = data["weather"][0]["description"]
            return f"In {city}, the weather is {temp}°C with {desc}."
        else:
            print("Weather API error:", res.text)
    except Exception as e:
        print("Weather fetch error:", e)
    return f"Weather info not available for {city}."



# --- Chatbot API ---
@app.post("/api/farmer-query/")
async def farmer_query(payload: FarmerQuery):
    try:
        city = get_user_location() or payload.city
        weather_info = get_weather(city, payload.lang)

        response = chat_chain.run({
        "user_input": payload.query,
        "weather_info": weather_info,
        "location": city
        })


        if payload.lang != "en":
            translator_prompt = f"Translate this into {payload.lang}: {response}"
            response = llm.predict(translator_prompt)

        return {"advice": response, "weather": weather_info, "location": city}

    except Exception as e:
        # return the full error so you can debug
        return {
            "advice": "Could not generate advice due to error.",
            "weather": "Weather info not available",
            "location": "Unknown",
            "error": str(e)
        }
