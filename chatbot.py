import boto3
import json
from tools import get_weather_for_airport, get_all_airport_weather, get_model_metrics

bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

SYSTEM_PROMPT = """You are an expert flight delay analyst AI assistant with access to REAL live data tools.
You have access to a machine learning model trained on 7 million flight records from 2018 BTS data.
The model achieves ROC AUC of 0.732.

When asked about weather at specific airports, ALWAYS use the weather tool to get real numbers.
When asked about model performance, use the metrics tool.

Key facts:
- Airports tracked: JFK, LAX, ORD, ATL, DFW, DEN, SFO, SEA, MIA, BOS
- Overall delay rate is ~19%
- Summer and winter holidays have highest delays
- Friday and Sunday are worst days for delays
- ORD and ATL have highest delay rates due to hub congestion
- High precipitation (>0.3in) correlates with ~35% higher delay rates
- High winds (>25mph) correlate with ~20% higher delay rates
- Snow causes the most severe delays

Always cite the real data when you have it. Be specific with numbers."""

TOOLS = [
    {
        "toolSpec": {
            "name": "get_airport_weather",
            "description": "Get real live weather data for a specific airport from the database. Use this when asked about weather at a specific airport.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "airport_code": {
                            "type": "string",
                            "description": "3-letter airport code e.g. SEA, JFK, LAX"
                        }
                    },
                    "required": ["airport_code"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "get_all_weather",
            "description": "Get current weather for all 10 tracked airports. Use when asked to compare airports or general weather overview.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {}
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "get_metrics",
            "description": "Get the latest ML model performance metrics.",
            "inputSchema": {
                "json": {
                    "type": "object", 
                    "properties": {}
                }
            }
        }
    }
]

def handle_tool_call(tool_name, tool_input):
    if tool_name == 'get_airport_weather':
        return get_weather_for_airport(tool_input.get('airport_code', ''))
    elif tool_name == 'get_all_weather':
        return get_all_airport_weather()
    elif tool_name == 'get_metrics':
        return get_model_metrics()
    return "Unknown tool"

def ask_bedrock(question, chat_history=[]):
    messages = []
    for h in chat_history:
        messages.append({
            "role": h["role"],
            "content": [{"text": h["content"]}]
        })
    messages.append({
        "role": "user",
        "content": [{"text": question}]
    })

    # Agentic loop — keep going until no more tool calls
    while True:
        response = bedrock.invoke_model(
            modelId='amazon.nova-micro-v1:0',
            body=json.dumps({
                "messages": messages,
                "system": [{"text": SYSTEM_PROMPT}],
                "toolConfig": {"tools": TOOLS},
                "inferenceConfig": {"maxTokens": 512, "temperature": 0.7}
            })
        )

        result = json.loads(response['body'].read())
        stop_reason = result.get('stopReason', '')
        content = result['output']['message']['content']

        # Add assistant response to messages
        messages.append({"role": "assistant", "content": content})

        if stop_reason == 'tool_use':
            # Handle tool calls
            tool_results = []
            for block in content:
                if block.get('toolUse'):
                    tool = block['toolUse']
                    tool_result = handle_tool_call(tool['name'], tool.get('input', {}))
                    tool_results.append({
                        "toolResult": {
                            "toolUseId": tool['toolUseId'],
                            "content": [{"text": tool_result}]
                        }
                    })
            messages.append({"role": "user", "content": tool_results})
        else:
            # Final response
            for block in content:
                if block.get('text'):
                    return block['text']
            return "I couldn't generate a response."
