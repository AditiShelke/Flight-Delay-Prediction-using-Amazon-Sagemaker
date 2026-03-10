import boto3
import json
import re
from tools import get_weather_for_airport, get_all_airport_weather, get_model_metrics, get_weather_forecast
from datetime import datetime

bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

AIRPORTS = ['JFK','LAX','ORD','ATL','DFW','DEN','SFO','SEA','MIA','BOS']

def extract_airport(text):
    for a in AIRPORTS:
        if a in text.upper():
            return a
    return None

def extract_date(text):
    # Match patterns like March 15, 2026-03-15, Mar 15
    patterns = [
        r'(\d{4}-\d{2}-\d{2})',
        r'(march|mar)\s+(\d{1,2})',
        r'(april|apr)\s+(\d{1,2})',
        r'(may)\s+(\d{1,2})',
    ]
    text_lower = text.lower()
    month_map = {'january':'01','february':'02','march':'03','april':'04',
                 'may':'05','june':'06','july':'07','august':'08',
                 'september':'09','october':'10','november':'11','december':'12',
                 'jan':'01','feb':'02','mar':'03','apr':'04','jun':'06',
                 'jul':'07','aug':'08','sep':'09','oct':'10','nov':'11','dec':'12'}
    
    # Try YYYY-MM-DD first
    m = re.search(r'(\d{4}-\d{2}-\d{2})', text)
    if m:
        return m.group(1)
    
    # Try Month DD
    m = re.search(r'(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s+(\d{1,2})', text_lower)
    if m:
        month = month_map[m.group(1)]
        day   = m.group(2).zfill(2)
        return f"2026-{month}-{day}"
    return None

SYSTEM_PROMPT = """You are an expert flight delay analyst AI assistant with access to REAL live data.
You have been provided with actual weather forecast data above. Use these REAL numbers in your answer.
Give specific delay probability estimates based on the weather data provided.
Be direct and specific — give a percentage estimate for delay risk.

Key correlations:
- Wind >25mph → ~20% higher delay rate (base 19% → ~23%)
- Precipitation >0.3in → ~35% higher delay rate (base 19% → ~26%)  
- Snow any amount → severe delays possible (base 19% → 40%+)
- Multiple bad factors → compound the risk
- JFK, ORD, EWR historically highest delay airports (+5% base)

Always give a specific delay probability percentage in your answer."""

def ask_bedrock(question, chat_history=[]):
    # Auto-inject forecast data if question mentions future date + airport
    airport = extract_airport(question)
    date    = extract_date(question)
    
    extra_context = ""
    if airport and date:
        forecast = get_weather_forecast(airport, date)
        extra_context = f"\n\nREAL FORECAST DATA (use these numbers):\n{forecast}\n"
    elif airport and any(word in question.lower() for word in ['tomorrow','next','forecast','will','going to']):
        from datetime import timedelta
        tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        forecast = get_weather_forecast(airport, tomorrow)
        extra_context = f"\n\nREAL FORECAST DATA (use these numbers):\n{forecast}\n"

    messages = []
    for h in chat_history:
        messages.append({
            "role": h["role"],
            "content": [{"text": h["content"]}]
        })
    
    full_question = question + extra_context
    messages.append({
        "role": "user",
        "content": [{"text": full_question}]
    })

    response = bedrock.invoke_model(
        modelId='amazon.nova-micro-v1:0',
        body=json.dumps({
            "messages": messages,
            "system": [{"text": SYSTEM_PROMPT}],
            "inferenceConfig": {"maxTokens": 512, "temperature": 0.7}
        })
    )

    result  = json.loads(response['body'].read())
    content = result['output']['message']['content']
    for block in content:
        if block.get('text'):
            return re.sub(r'<thinking>.*?</thinking>', '', block['text'], flags=re.DOTALL).strip()
    return "I couldn't generate a response."
