# Water-Quality-Prediction-System

Water Quality Prediction
This project aims to predict the water quality based on various parameters such as temperature, pH, and light intensity using machine learning algorithms.

Project Description
The project includes the following steps:

Data collection and preprocessing
Exploratory Data Analysis (EDA) and visualization
Model selection and evaluation
Model optimization and hyperparameter tuning
Model deployment and prediction
The main Jupyter notebook file for the project is Water Quality.ipynb, which contains the complete code implementation for the above steps.

The Water prediction.csv file contains the dataset used for training and testing the prediction model.

The model.sav file contains the implementation of Random Forest algorithms used for training and predicting water quality.

The app.py file contains the implementation for the Streamlit web application.

Requirements
The project requires the following Python packages:

Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Scipy
To install these packages, run the following command in your terminal or command prompt:

Copy code
pip install -r requirements.txt
Dataset
The dataset used in this project is Water prediction.csv, which contains features such as temperature, pH, and turbidity of water samples, as well as the corresponding water quality index (WQI) values.

Code Implementation
The main code implementation for the project is in the Water Quality.ipynb file, which contains the following sections:

Importing necessary libraries
Loading the dataset and exploring the data
Data preprocessing, cleaning, and feature engineering
Exploratory Data Analysis (EDA) and visualization
Pie chart
Count plot
Distribution plot
KDE plot
Data correlation
Pandas Profile report
Model selection and evaluation
Random Forest
Model evaluation
Model optimization and hyperparameter tuning
Random Forest
Model deployment and prediction

The model.sav file contains the following machine learning models for the project:

RandomForestClassifier: A random forest classifier for predicting water quality

The app.py file contains the following code implementation for the Streamlit web application:

Importing necessary libraries
Loading the trained model and utility functions
Building the web application using Streamlit
Header and description
User input for temperature, pH, turbidity and other parameters
Prediction of water quality using the trained model
Displaying the predicted water quality

Conclusion
This project demonstrates the prediction of water quality based on various parameters such as temperature, pH, and chlorine levels using machine learning algorithms. The project includes data preprocessing, EDA, model training, and prediction, and provides utility functions for data visualization and analysis.
