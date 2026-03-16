import os
import streamlit as st
import boto3
import pymysql
import pandas as pd
import plotly.express as px
import json
import re
import numpy as np
from datetime import datetime

REGION   = 'us-east-1'
BUCKET   = 'flight-delay-aditi-2026'
RDS_HOST = 'flight-delay-db.ca522oasyz0w.us-east-1.rds.amazonaws.com'
RDS_USER = 'admin'
RDS_PASS = os.environ.get("RDS_PASS", "your-password-here")
RDS_DB   = 'flightdelaydb'
ENDPOINT = 'flight-delay-endpoint'

s3 = boto3.client('s3', region_name=REGION)

st.set_page_config(page_title='Flight Delay Predictor', page_icon='✈️', layout='wide')

@st.cache_data
def load_mappings():
    obj = s3.get_object(Bucket=BUCKET, Key='sagemaker/mappings/category_mappings.json')
    return json.loads(obj['Body'].read())

@st.cache_data
def load_metrics():
    try:
        obj = s3.get_object(Bucket=BUCKET, Key='sagemaker/metrics/v3_latest.json')
        m = json.loads(obj['Body'].read())
        return {
            'roc_auc': m.get('v3_auc', 'N/A'),
            'accuracy': m.get('v3_accuracy', 'N/A'),
            'recall': m.get('v3_recall', 'N/A'),
            'f1_score': m.get('v3_f1', 'N/A'),
            'precision': m.get('v3_precision', 'N/A'),
            'version': 'v3',
            'num_features': m.get('num_features', 27),
        }
    except:
        return {'roc_auc': 'N/A', 'accuracy': 'N/A', 'recall': 'N/A', 'f1_score': 'N/A', 'version': 'v1', 'num_features': 11}

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

def get_airport_weather_for_prediction(airport_code):
    """Pull latest weather for a specific airport from RDS for model input"""
    try:
        conn = pymysql.connect(host=RDS_HOST, user=RDS_USER,
                               password=RDS_PASS, database=RDS_DB, connect_timeout=5)
        df = pd.read_sql(f"""
            SELECT AWND, PRCP, SNOW, TMAX, TMIN
            FROM weather_data
            WHERE airport = '{airport_code}'
            ORDER BY fetch_date DESC LIMIT 1
        """, conn)
        conn.close()
        if not df.empty:
            return {
                'wind': float(df['AWND'].iloc[0] or 0),
                'precip': float(df['PRCP'].iloc[0] or 0),
                'snow': float(df['SNOW'].iloc[0] or 0),
                'tmax': float(df['TMAX'].iloc[0] or 0),
                'tmin': float(df['TMIN'].iloc[0] or 0),
            }
    except:
        pass
    return {'wind': 9.5, 'precip': 0.0, 'snow': 0.0, 'tmax': 68.0, 'tmin': 51.0}

mappings = load_mappings()
metrics  = load_metrics()

HOLIDAYS = {
    (1,1),(1,2),(1,15),(2,19),(5,28),
    (7,3),(7,4),(7,5),(9,3),
    (11,21),(11,22),(11,23),(11,25),(11,26),
    (12,24),(12,25),(12,26),(12,31)
}

st.sidebar.header('📊 Model Performance')
for label, key in [('ROC AUC','roc_auc'),('Accuracy','accuracy'),('Recall','recall'),('F1 Score','f1_score')]:
    val = metrics.get(key, 'N/A')
    st.sidebar.metric(label, f'{val:.4f}' if isinstance(val, float) else val)
st.sidebar.markdown('---')
st.sidebar.markdown('**Trained on:** 7M+ flight records')
st.sidebar.markdown('**Data:** BTS 2018 + NOAA Weather')
st.sidebar.markdown('**Model:** XGBoost v3 with HPO')
st.sidebar.markdown(f'**Features:** {metrics.get("num_features", 27)}')

st.title('✈️ Real-Time Flight Delay Prediction')
st.markdown('**Built with AWS SageMaker + XGBoost + Live Weather + Delay Propagation**')

st.header('🔮 Predict Flight Delay')
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader('Flight Info')
    airline = st.selectbox('Airline', sorted(mappings['airlines']))
    origin  = st.selectbox('Origin Airport', sorted(mappings['origins']))
    dest    = st.selectbox('Destination Airport', sorted(mappings['dests']))

with col2:
    st.subheader('Schedule')
    month       = st.slider('Month', 1, 12, datetime.now().month)
    day         = st.slider('Day of Month', 1, 31, min(datetime.now().day, 28))
    day_of_week = st.selectbox('Day of Week', [1,2,3,4,5,6,7],
                   format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x-1])
    dep_hour    = st.slider('Departure Hour (0-23)', 0, 23, 10)

with col3:
    st.subheader('Flight Details')
    air_time   = st.slider('Air Time (min)', 30, 700, 150)
    distance   = st.slider('Distance (miles)', 50, 5000, 800)

with st.expander('⚙️ Operational Context (Advanced)', expanded=False):
    st.caption('Delay propagation factors — how delays cascade between flights on the same aircraft.')
    op1, op2, op3 = st.columns(3)
    with op1:
        prev_arr_delay = st.number_input('Previous Flight Delay (min)',
            min_value=-30, max_value=300, value=0)
        prev_was_delayed = 1 if prev_arr_delay > 15 else 0
    with op2:
        turnaround = st.number_input('Turnaround Time (min)',
            min_value=0, max_value=720, value=60)
        tight_turnaround = 1 if turnaround < 45 else 0
    with op3:
        tail_daily_delay = st.number_input('Aircraft Delay Today (min)',
            min_value=0, max_value=500, value=0)
        is_first_flight = 1 if (prev_arr_delay == 0 and turnaround == 60 and tail_daily_delay == 0) else 0

if st.button('🚀 Predict Delay', use_container_width=True):
    try:
        airline_code = mappings['airlines'].index(airline)
        origin_code  = mappings['origins'].index(origin)
        dest_code    = mappings['dests'].index(dest)
        quarter      = (month - 1) // 3 + 1
        is_holiday   = 1 if (month, day) in HOLIDAYS else 0
        is_peak      = 1 if (6 <= dep_hour <= 10) or (16 <= dep_hour <= 20) else 0
        is_early     = 1 if dep_hour < 7 else 0
        route_delay_rate   = 0.19
        airline_delay_rate = 0.19

        # Pull LIVE weather from RDS
        wx = get_airport_weather_for_prediction(origin)

        # 27-feature payload matching v3 training order
        features = [
            2018, quarter, month, day, day_of_week,
            airline_code, origin_code, dest_code, air_time, distance,
            is_holiday, dep_hour, is_peak, is_early,
            prev_arr_delay, prev_was_delayed, turnaround,
            tight_turnaround, tail_daily_delay, is_first_flight,
            route_delay_rate, airline_delay_rate,
            wx['wind'], wx['precip'], wx['snow'], wx['tmax'], wx['tmin']
        ]
        payload = ','.join(str(f) for f in features)

        sm_runtime = boto3.client('sagemaker-runtime', region_name=REGION)
        response = sm_runtime.invoke_endpoint(
            EndpointName=ENDPOINT, ContentType='text/csv', Body=payload
        )
        score = float(response['Body'].read().decode('utf-8').strip())

        st.markdown('---')
        if score > 0.7:
            st.error(f'🔴 **HIGH DELAY RISK** — {score:.1%} probability of delay (>15 min)')
        elif score > 0.5:
            st.warning(f'🟡 **MODERATE DELAY RISK** — {score:.1%} probability of delay (>15 min)')
        elif score > 0.3:
            st.info(f'🟢 **LOW DELAY RISK** — {score:.1%} probability of delay (>15 min)')
        else:
            st.success(f'✅ **LIKELY ON TIME** — {score:.1%} probability of delay (>15 min)')
        st.progress(min(score, 1.0))

        with st.expander('📋 Prediction Details', expanded=True):
            d1, d2, d3 = st.columns(3)
            with d1:
                st.markdown(f'**Flight:** {airline} {origin} → {dest}')
                st.markdown(f'**Schedule:** Month {month}, Day {day}')
                st.markdown(f'**{["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][day_of_week-1]}** at {dep_hour}:00')
            with d2:
                st.markdown(f'**Prev Delay:** {prev_arr_delay} min {"⚠️" if prev_arr_delay > 15 else "✅"}')
                st.markdown(f'**Turnaround:** {turnaround} min {"⚠️" if tight_turnaround else "✅"}')
                st.markdown(f'**Holiday:** {"Yes" if is_holiday else "No"} | **Peak:** {"Yes" if is_peak else "No"}')
            with d3:
                st.markdown(f'**🌤️ Live Weather at {origin}:**')
                st.markdown(f'Wind: {wx["wind"]:.1f} mph | Precip: {wx["precip"]:.2f} in')
                st.markdown(f'Snow: {wx["snow"]:.1f} in | Temp: {wx["tmin"]:.0f}–{wx["tmax"]:.0f}°F')
        st.caption(f'Score: {score:.4f} | Model: XGBoost v3 (27 features, weather-aware)')
    except Exception as e:
        st.warning(f'⚠️ Endpoint not available: {e}')
        st.info('Deploy the SageMaker endpoint to enable live predictions.')

st.header('🌤️ Live Airport Weather')
weather_df = get_weather()
if not weather_df.empty:
    st.dataframe(weather_df, use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(weather_df.groupby('airport')['PRCP'].mean().reset_index(),
                     x='airport', y='PRCP', title='Avg Precipitation by Airport',
                     color_discrete_sequence=['#4A90D9'])
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(weather_df.groupby('airport')['TMAX'].mean().reset_index(),
                     x='airport', y='TMAX', title='Avg Max Temp by Airport',
                     color_discrete_sequence=['#E8725C'])
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info('Weather data loading...')

st.markdown('---')
st.caption('Built by Aditi Shelke | AWS SageMaker + Lambda + RDS + Bedrock + Streamlit')

st.markdown('---')
st.header('🤖 AI Flight Delay Assistant')
st.caption('Powered by Amazon Bedrock Nova Micro')

from chatbot import ask_bedrock

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg['role']):
        st.write(msg['content'])

if prompt := st.chat_input('Ask about flight delays... e.g. "Which airports have most delays?"'):
    st.session_state.chat_history.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.write(prompt)
    with st.chat_message('assistant'):
        with st.spinner('Thinking...'):
            raw = ask_bedrock(prompt, st.session_state.chat_history[:-1])
            response = re.sub(r"<thinking>.*?</thinking>", "", raw, flags=re.DOTALL).strip()
        st.write(response)
    st.session_state.chat_history.append({'role': 'assistant', 'content': response})
