# Flight-Delay-Prediction-using-ML
### Overview
This project aims to improve the customer experience for flights that were delayed due to weather conditions for the busiest airports in the US. To achieve this, we have built a machine learning (ML) model using XGBoost to predict whether a flight will be delayed due to weather conditions. The model is trained on historical flight data collected by the Office of Airline Information, Bureau of Transportation Statistics (BTS), covering the years 2013 to 2018.

### Dataset
The dataset used for this project contains scheduled and actual departure and arrival times reported by certified US air carriers that account for at least 1 percent of domestic scheduled passenger revenues. The dataset includes the following information for each flight:

Date and time of departure and arrival.
Origin and destination airports.
Airline operating the flight.
Flight distance.
Delay status of the flight.

### Steps
#### We use the XGBoost algorithm, a powerful gradient boosting technique, to train a binary classification model to predict flight delays due to weather. The model hyperparameters are tuned using SageMaker's HyperparameterTuner to optimize the performance.

# Evaluation 
Model Evaluation: The trained model is evaluated on a test dataset, and various performance metrics, such as accuracy, precision, recall, F1 score, ROC curve, and confusion matrix, are calculated and displayed.

# Results
The trained XGBoost model shows promising results in predicting flight delays due to weather. The model's performance is evaluated using various metrics, demonstrating its ability to identify weather-related delays accurately.

# Contributing
We welcome contributions to this project! If you find any issues or have ideas for improvement, please feel free to open an issue or submit a pull request.
