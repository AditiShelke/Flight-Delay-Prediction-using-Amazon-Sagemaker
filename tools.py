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

import urllib.request

OWM_KEY = 'a4aab45a3caef73055dbc342adbf5412'

AIRPORT_COORDS = {
    'JFK': (40.6413, -73.7781),
    'LAX': (33.9425, -118.4081),
    'ORD': (41.9742, -87.9073),
    'ATL': (33.6407, -84.4277),
    'DFW': (32.8998, -97.0403),
    'DEN': (39.8561, -104.6737),
    'SFO': (37.6213, -122.3790),
    'SEA': (47.4502, -122.3088),
    'MIA': (25.7959, -80.2870),
    'BOS': (42.3656, -71.0096)
}

def get_weather_forecast(airport_code, target_date):
    """Get weather forecast for a specific airport and future date (up to 7 days)"""
    try:
        airport = airport_code.upper()
        if airport not in AIRPORT_COORDS:
            return f"Airport {airport} not supported. Use: {', '.join(AIRPORT_COORDS.keys())}"
        
        lat, lon = AIRPORT_COORDS[airport]
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OWM_KEY}&units=metric&cnt=40"
        
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read())
        
        # Find forecast closest to target date
        target = str(target_date)
        best = None
        for item in data['list']:
            if target in item['dt_txt']:
                best = item
                break
        
        if not best:
            # Just get the closest available
            best = data['list'][0]
        
        weather = best['weather'][0]['description']
        temp    = best['main']['temp']
        wind    = best['wind']['speed'] * 2.237  # m/s to mph
        rain    = best.get('rain', {}).get('3h', 0)
        snow    = best.get('snow', {}).get('3h', 0)
        dt      = best['dt_txt']

        result = f"Forecast for {airport} on {dt}:\n"
        result += f"  Conditions: {weather}\n"
        result += f"  Temperature: {temp:.1f}°C\n"
        result += f"  Wind: {wind:.1f}mph\n"
        result += f"  Rain: {rain:.2f}mm\n"
        result += f"  Snow: {snow:.2f}mm\n"
        
        # Delay risk assessment
        risk = "LOW"
        if wind > 25 or rain > 5 or snow > 1:
            risk = "HIGH"
        elif wind > 15 or rain > 2:
            risk = "MODERATE"
        result += f"  Delay Risk from Weather: {risk}\n"
        return result
    except Exception as e:
        return f"Forecast error: {e}"
