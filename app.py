from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
import json

app = Flask(__name__)
CORS(app)

# Connects to Groq's free servers
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def get_weather(location):
    return f"The weather in {location} is currently sunny and 25°C."

AVAILABLE_TOOLS = {
    "get_weather": get_weather
}

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")
    
    system_prompt = (
        "You are FRIDAY, an advanced, highly adaptive personal AI assistant. "
        "Your goal is to assist the Boss with any doubts, questions, or automation tasks. "
        "Keep your tone crisp, intelligent, slightly witty, and address the user as 'Boss'."
    )
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a specific location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city and state"}
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        tools=tools,
        tool_choice="auto"
    )
    
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    
    if tool_calls:
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            if function_name in AVAILABLE_TOOLS:
                tool_output = AVAILABLE_TOOLS[function_name](**function_args)
                
                final_response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                        response_message,
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": tool_output,
                        }
                    ]
                )
                return jsonify({"reply": final_response.choices[0].message.content})
    
    return jsonify({"reply": response_message.content})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
