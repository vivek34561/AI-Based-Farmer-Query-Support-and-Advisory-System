from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key="OPENAI_API_KEY")

def get_advice(query, context=None):
    context_text = f"Location: {context.get('location', 'Unknown')}, Crop: {context.get('crop', 'Unknown')}" if context else ""
    prompt = f"{context_text}\nFarmer Query: {query}\nProvide advice in Malayalam:"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
