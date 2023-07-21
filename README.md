# Flight-Delay-Prediction-using-Amazon SageMaker
### Overview
Flight delays can lead to significant disruptions and inconvenience for both airlines and passengers. This project aims to develop a machine learning model that can accurately predict flight delays based on historical data and weather conditions.
The model is trained on historical flight data collected by the Office of Airline Information, Bureau of Transportation Statistics (BTS), covering the years 2013 to 2018.
The project uses Amazon SageMaker,XGBoost to train and deploy machine learning models for flight delay prediction.

### Dataset
The data used for this project includes historical flight information, weather data, and airline information contains scheduled and actual departure and arrival times reported by certified US air carriers.
##### The dataset includes the following information for each flight:
1. Date and time of departure and arrival.
2. Origin and destination airports.
3. Airline operating the flight.

The dataset contains the following columns:
* Year: The year of the flight.
* Quarter: The quarter of the year when the flight occurred.
* Month: The month of the flight.
* DayofMonth: The day of the month when the flight occurred.
* DayOfWeek: The day of the week when the flight occurred.
* FlightDate: The date of the flight.
* Reporting_Airline: The airline reporting the flight delay.
* Origin: The origin airport code.
* OriginState: The state where the origin airport is located.
* Dest: The destination airport code.
* is_delay: Whether the flight was delayed (1) or not (0) due to weather conditions.
* AirTime: The flight time (in minutes).
* is_holiday: Whether the flight date is a holiday (1) or not (0).
*....

### Feature Engineering
The data was preprocessed and feature engineering techniques were applied to enhance the predictive power of the model. The following additional features were added:

1. Holidays: An indicator variable to mark flight dates that coincide with public holidays.
2. Weather Conditions: Weather data, including wind speed, precipitation, snowfall, and temperature, were merged with the flight data based on airport codes.

### Model Development
#### We use the XGBoost algorithm, a powerful gradient boosting technique, to train a binary classification model to predict flight delays due to weather. The model hyperparameters are tuned using SageMaker's HyperparameterTuner to optimize the performance.

#### Usage
To replicate the project and train the models, follow these steps:
1. Prepare the data: The flight and weather data should be preprocessed and merged to create the dataset used for training.
2. Feature engineering: Add the additional features, such as holidays and weather conditions, to enhance the model's predictive power.
3. Model development: Train an initial XGBoost model on the dataset.
4. Hyperparameter optimization: Perform hyperparameter tuning to find the best hyperparameters for the XGBoost model.
5. Evaluation: Evaluate the performance of the models using various metrics and visualizations.
   
# Evaluation 
Model Evaluation: The trained model is evaluated on a test dataset, and various performance metrics, such as accuracy, precision, recall, F1 score, ROC curve, and confusion matrix, are calculated and displayed.

# Results
The trained XGBoost model shows promising results in predicting flight delays due to weather. The model's performance is evaluated using various metrics, demonstrating its ability to identify weather-related delays accurately.

# Contributing
We welcome contributions to this project! If you find any issues or have ideas for improvement, please feel free to open an issue or submit a pull request.
