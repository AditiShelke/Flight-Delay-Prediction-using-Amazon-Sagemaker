import boto3
import json
import re
from tools import get_weather_for_airport, get_all_airport_weather, get_model_metrics, get_weather_forecast
from datetime import datetime, timedelta

bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
AIRPORTS = ['JFK','LAX','ORD','ATL','DFW','DEN','SFO','SEA','MIA','BOS']

def extract_airport(text):
    for a in AIRPORTS:
        if a in text.upper():
            return a
    return None

def extract_date(text):
    text_lower = text.lower()
    month_map = {'january':'01','february':'02','march':'03','april':'04',
                 'may':'05','june':'06','july':'07','august':'08',
                 'september':'09','october':'10','november':'11','december':'12',
                 'jan':'01','feb':'02','mar':'03','apr':'04','jun':'06',
                 'jul':'07','aug':'08','sep':'09','oct':'10','nov':'11','dec':'12'}
    m = re.search(r'(\d{4}-\d{2}-\d{2})', text)
    if m: return m.group(1)
    m = re.search(r'(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s+(\d{1,2})', text_lower)
    if m: return f"2026-{month_map[m.group(1)]}-{m.group(2).zfill(2)}"
    if 'tomorrow' in text_lower:
        return (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    return None

SYSTEM_PROMPT = """You are an expert flight delay analyst AI assistant with access to REAL live weather data.

IMPORTANT — How to calculate delay probability using MULTIPLICATIVE compounding:
Start with base rate 19%. Apply multipliers sequentially:
- High-delay airport (JFK, ORD, EWR): multiply by 1.25
- Wind >25mph: multiply by 1.20
- Precipitation >0.3in: multiply by 1.35
- Snow any amount: multiply by 2.1
- Peak hours (6-10am, 4-8pm): multiply by 1.15
- Friday or Sunday: multiply by 1.10
- Previous aircraft delay >15min: multiply by 1.30

EXAMPLE: JFK with 35mph wind, no rain:
  19% x 1.25 (JFK) = 23.8%
  23.8% x 1.20 (wind) = 28.5%
  Final: ~29% delay probability

DO NOT add percentages. MULTIPLY the current rate by each factor.
Cap final probability at 85% maximum.
Show your step-by-step multiplication.

Our v3 XGBoost model (ROC AUC 0.879) was trained on 7M+ flights with 27 features including actual NOAA weather data, delay propagation chains, and operational context.

Always use the actual weather data provided. Never say you lack access — you have live data.
Give one clear final percentage."""

def ask_bedrock(question, chat_history=[]):
    airport = extract_airport(question)
    date = extract_date(question)
    extra_context = ""

    if airport and date:
        try:
            forecast = get_weather_forecast(airport, date)
            if forecast and 'error' not in forecast.lower():
                extra_context = f"\n\nREAL WEATHER FORECAST for {airport} on {date}:\n{forecast}\n"
        except: pass

    if airport:
        try:
            live_weather = get_weather_for_airport(airport)
            if live_weather:
                extra_context += f"\n\nCURRENT LIVE WEATHER from RDS for {airport}:\n{live_weather}\n"
        except: pass

    if not airport and any(word in question.lower() for word in ['all airport','which airport','compare','worst','best','most delay','least delay']):
        try:
            all_weather = get_all_airport_weather()
            if all_weather:
                extra_context += f"\n\nCURRENT WEATHER ALL AIRPORTS:\n{all_weather}\n"
        except: pass

    if any(word in question.lower() for word in ['model','accuracy','auc','performance','metrics']):
        try:
            m = get_model_metrics()
            if m: extra_context += f"\n\nMODEL METRICS:\n{m}\n"
        except: pass

    messages = []
    for h in chat_history:
        messages.append({"role": h["role"], "content": [{"text": h["content"]}]})
    messages.append({"role": "user", "content": [{"text": question + extra_context}]})

    response = bedrock.invoke_model(
        modelId='amazon.nova-micro-v1:0',
        body=json.dumps({
            "messages": messages,
            "system": [{"text": SYSTEM_PROMPT}],
            "inferenceConfig": {"maxTokens": 512, "temperature": 0.7}
        })
    )
    result = json.loads(response['body'].read())
    content = result['output']['message']['content']
    for block in content:
        if block.get('text'):
            return re.sub(r'<thinking>.*?</thinking>', '', block['text'], flags=re.DOTALL).strip()
    return "I couldn't generate a response."
