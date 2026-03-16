# Real-Time Flight Delay Prediction Platform

![AWS](https://img.shields.io/badge/AWS-SageMaker-orange) ![Python](https://img.shields.io/badge/Python-3.9-blue) ![XGBoost](https://img.shields.io/badge/Model-XGBoost-green) ![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)

A production-grade MLOps platform that predicts flight delays in real-time using machine learning, live weather data, and a conversational AI assistant.

# What I built:
* Trained XGBoost models on 7M+ flight records (BTS 2018 data), achieving ROC AUC 0.875 through feature engineering inspired by recent academic research ( Zhou (George Mason, 2025) ), on delay propagation and aircraft rotation chains
* Implemented hyperparameter tuning across 5 jobs using SageMaker Automatic Tuning, handling class imbalance (19% delay rate) with dynamic scale_pos_weight
* Built an automated pipeline: AWS Lambda (live weather ingestion) → Amazon RDS → SageMaker (retraining) → EC2 (Streamlit dashboard)
* Integrated Amazon Bedrock Nova Micro to create a conversational AI assistant that answers natural language queries using live RDS data 

## Architecture
```
Live Weather API → AWS Lambda → Amazon RDS (MySQL)
                                      ↓
BTS Flight Data → Amazon S3 → SageMaker XGBoost → Prediction Endpoint
                                      ↓
                        Streamlit Dashboard (EC2)
                                      ↓
                    Amazon Bedrock Nova Micro (AI Chatbot)
```

## 📊 Model Performance based on version 1 of model (11 Features)

| Metric | Score |
|--------|-------|
| ROC AUC | **0.732** |
| Accuracy | 0.687 |
| Recall | 0.639 |
| F1 Score | 0.439 |

- Trained on **7M+ flight records** (BTS 2018 On-Time Performance Data)
- Hyperparameter tuning across 5 jobs via SageMaker Automatic Tuning
- Handles class imbalance (19% delay rate) with dynamic scale_pos_weight
- Improved model ROC AUC from 0.732 → 0.875 (in version 2) Features 22 by engineering delay propagation features (previous flight delay, turnaround time, tail number rotation chains) inspired by Zhou et al. 2025

## Features

- **Real-time predictions** — input airline, route, date → get delay probability
- **Live weather integration** — Lambda fetches daily weather for 10 major airports
- **AI chatbot** — Bedrock Nova Micro answers questions using real RDS data as context
- **Automated pipeline** — Lambda → S3 → SageMaker → EC2 dashboard

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Storage | Amazon S3, Amazon RDS (MySQL) |
| ML Training | Amazon SageMaker, XGBoost |
| Serving | SageMaker Endpoint, EC2 |
| Data Pipeline | AWS Lambda, EventBridge |
| Dashboard | Streamlit, Plotly |
| AI Assistant | Amazon Bedrock (Nova Micro) |
| Language | Python 3.9 |

## Project Structure
```
├── app.py          # Streamlit dashboard
├── chatbot.py      # Bedrock AI assistant with tool use
├── tools.py        # RDS + S3 data query tools for AI
└── README.md
```
## Weather Forecast Integration

- Integrated **OpenWeatherMap API** for 7-day weather forecasts
- AI chatbot automatically detects airport + date from natural language
- Provides specific delay probability % based on forecast conditions
- Falls back to historical pattern estimates when forecast unavailable

Example queries:
- *"Will my JFK flight be delayed on March 15?"*
- *"What's the delay risk for LAX tomorrow?"*
- *"Compare delay risk between ORD and ATL this Friday"*


## Key ML Decisions

- **Target**: ArrDel15 (15+ min late arrival) — binary classification
- **Dropped leakage features**: DepDelay, CarrierDelay, WeatherDelay (known only after flight)
- **Features**: Year, Quarter, Month, DayofMonth, DayOfWeek, Airline, Origin, Dest, AirTime, Distance, is_holiday
- **Best hyperparameters**: max_depth=9, eta=0.132, gamma=1.95, subsample=0.727, num_round=200

## Screenshots


## Author

**Aditi Shelke** — [LinkedIn](https://linkedin.com/in/aditi-shelke) | [GitHub](https://github.com/AditiShelke)
