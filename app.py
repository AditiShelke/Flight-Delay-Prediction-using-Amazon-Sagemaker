import streamlit as st
import boto3
import pymysql
import pandas as pd
import plotly.express as px
import json
from datetime import datetime

REGION   = 'us-east-1'
BUCKET   = 'flight-delay-aditi-2026'
RDS_HOST = 'flight-delay-db.ca522oasyz0w.us-east-1.rds.amazonaws.com'
RDS_USER = 'admin'
RDS_PASS = 'FlightDelay2026!'
RDS_DB   = 'flightdelaydb'

s3 = boto3.client('s3', region_name=REGION)

st.set_page_config(page_title='✈️ Flight Delay Predictor', page_icon='✈️', layout='wide')
st.title('✈️ Real-Time Flight Delay Prediction')
st.markdown('**Built with AWS SageMaker + XGBoost + Live Weather Data**')

@st.cache_data
def load_mappings():
    obj = s3.get_object(Bucket=BUCKET, Key='sagemaker/mappings/category_mappings.json')
    return json.loads(obj['Body'].read())

@st.cache_data
def load_metrics():
    obj = s3.get_object(Bucket=BUCKET, Key='sagemaker/metrics/latest.json')
    return json.loads(obj['Body'].read())

def get_weather():
    try:
        conn = pymysql.connect(host=RDS_HOST, user=RDS_USER,
                               password=RDS_PASS, database=RDS_DB, connect_timeout=5)
        df = pd.read_sql("""
            SELECT airport, fetch_date, AWND, PRCP, SNOW, TMAX, TMIN
            FROM weather_data ORDER BY fetch_date DESC LIMIT 100
        """, conn)
        conn.close()
        return df
    except:
        return pd.DataFrame()

mappings = load_mappings()
metrics  = load_metrics()

st.sidebar.header('📊 Model Performance')
st.sidebar.metric('ROC AUC',  metrics.get('roc_auc', 'N/A'))
st.sidebar.metric('Accuracy', metrics.get('accuracy', 'N/A'))
st.sidebar.metric('Recall',   metrics.get('recall', 'N/A'))
st.sidebar.metric('F1 Score', metrics.get('f1_score', 'N/A'))
st.sidebar.markdown('---')
st.sidebar.markdown('**Trained on:** 7M+ flight records')
st.sidebar.markdown('**Data:** BTS 2018 On-Time Performance')
st.sidebar.markdown('**Model:** XGBoost with HPO')

st.header('🔮 Predict Flight Delay')
col1, col2, col3 = st.columns(3)

with col1:
    airline = st.selectbox('Airline', sorted(mappings['airlines']))
    origin  = st.selectbox('Origin Airport', sorted(mappings['origins']))
    dest    = st.selectbox('Destination Airport', sorted(mappings['dests']))

with col2:
    month       = st.slider('Month', 1, 12, datetime.now().month)
    day         = st.slider('Day of Month', 1, 31, datetime.now().day)
    day_of_week = st.selectbox('Day of Week', [1,2,3,4,5,6,7],
                   format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x-1])

with col3:
    air_time   = st.slider('Air Time (min)', 30, 700, 150)
    distance   = st.slider('Distance (miles)', 50, 5000, 800)
    is_holiday = st.checkbox('Holiday?')

if st.button('🚀 Predict Delay', use_container_width=True):
    try:
        airline_code = mappings['airlines'].index(airline)
        origin_code  = mappings['origins'].index(origin)
        dest_code    = mappings['dests'].index(dest)
        quarter      = (month - 1) // 3 + 1

        sm_runtime = boto3.client('sagemaker-runtime', region_name=REGION)
        payload    = f'2018,{quarter},{month},{day},{day_of_week},{airline_code},{origin_code},{dest_code},{air_time},{distance},{int(is_holiday)}'

        response = sm_runtime.invoke_endpoint(
            EndpointName='flight-delay-endpoint',
            ContentType='text/csv',
            Body=payload
        )
        score = float(response['Body'].read().decode('utf-8').strip())

        st.markdown('---')
        if score > 0.5:
            st.error(f'⚠️ **LIKELY DELAYED** — {score:.1%} probability of delay')
        else:
            st.success(f'✅ **LIKELY ON TIME** — {score:.1%} probability of delay')
        st.progress(score)
        st.caption(f'Threshold: 0.5 | Score: {score:.4f}')
    except Exception as e:
        st.warning(f'Endpoint not available: {e}')
        st.info('Deploy the SageMaker endpoint to enable live predictions.')

st.header('🌤️ Live Airport Weather')
weather_df = get_weather()

if not weather_df.empty:
    st.dataframe(weather_df, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(weather_df.groupby('airport')['PRCP'].mean().reset_index(),
                     x='airport', y='PRCP', title='Avg Precipitation by Airport')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(weather_df.groupby('airport')['TMAX'].mean().reset_index(),
                     x='airport', y='TMAX', title='Avg Max Temp by Airport')
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info('Weather data loading...')

st.markdown('---')
st.caption('Built by Aditi Shelke | AWS SageMaker + Lambda + RDS + Streamlit')

# ── AI Chatbot ────────────────────────────────────────
st.markdown('---')
st.header('🤖 AI Flight Delay Assistant')
st.caption('Powered by Amazon Bedrock Nova Micro')

from chatbot import ask_bedrock

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg['role']):
        st.write(msg['content'])

# Chat input
if prompt := st.chat_input('Ask about flight delays... e.g. "Which airports have most delays?"'):
    st.session_state.chat_history.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.write(prompt)

    with st.chat_message('assistant'):
        with st.spinner('Thinking...'):
            raw = ask_bedrock(prompt, st.session_state.chat_history[:-1])
            import re
            response = re.sub(r"<thinking>.*?</thinking>", "", raw, flags=re.DOTALL).strip()
        st.write(response)
    st.session_state.chat_history.append({'role': 'assistant', 'content': response})
