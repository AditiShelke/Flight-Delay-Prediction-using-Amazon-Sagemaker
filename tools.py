import pymysql
import boto3
import json

RDS_HOST = 'flight-delay-db.ca522oasyz0w.us-east-1.rds.amazonaws.com'
RDS_USER = 'admin'
RDS_PASS = 'FlightDelay2026!'
RDS_DB   = 'flightdelaydb'

def get_weather_for_airport(airport_code):
    """Get latest weather data for a specific airport from RDS"""
    try:
        conn = pymysql.connect(host=RDS_HOST, user=RDS_USER,
                               password=RDS_PASS, database=RDS_DB, connect_timeout=5)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT airport, fetch_date, AWND, PRCP, SNOW, TMAX, TMIN
            FROM weather_data
            WHERE airport = %s
            ORDER BY fetch_date DESC
            LIMIT 7
        """, (airport_code.upper(),))
        rows = cursor.fetchall()
        conn.close()
        if not rows:
            return f"No weather data found for {airport_code}"
        result = f"Weather data for {airport_code}:\n"
        for row in rows:
            result += f"  Date: {row[1]}, Wind: {row[2]}mph, Precip: {row[3]}in, Snow: {row[4]}in, MaxTemp: {row[5]}°C, MinTemp: {row[6]}°C\n"
        return result
    except Exception as e:
        return f"Error fetching weather: {e}"

def get_all_airport_weather():
    """Get today's weather for all airports"""
    try:
        conn = pymysql.connect(host=RDS_HOST, user=RDS_USER,
                               password=RDS_PASS, database=RDS_DB, connect_timeout=5)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT airport, fetch_date, AWND, PRCP, SNOW, TMAX, TMIN
            FROM weather_data
            ORDER BY fetch_date DESC
            LIMIT 10
        """)
        rows = cursor.fetchall()
        conn.close()
        if not rows:
            return "No weather data available"
        result = "Current weather across all tracked airports:\n"
        for row in rows:
            result += f"  {row[0]}: Wind={row[2]}mph, Precip={row[3]}in, Snow={row[4]}in, High={row[5]}°C, Low={row[6]}°C\n"
        return result
    except Exception as e:
        return f"Error: {e}"

def get_model_metrics():
    """Get latest model performance metrics from S3"""
    try:
        s3 = boto3.client('s3', region_name='us-east-1')
        obj = s3.get_object(Bucket='flight-delay-aditi-2026', Key='sagemaker/metrics/latest.json')
        metrics = json.loads(obj['Body'].read())
        return f"Model metrics: ROC AUC={metrics['roc_auc']}, Accuracy={metrics['accuracy']}, Recall={metrics['recall']}, F1={metrics['f1_score']}, Trained: {metrics['timestamp']}"
    except Exception as e:
        return f"Error: {e}"
