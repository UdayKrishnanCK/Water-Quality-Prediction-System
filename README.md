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

The data.csv file contains the dataset used for training and testing the prediction model.

The utilities.py file contains utility functions for data preprocessing, EDA, and visualization.

The models.py file contains the implementation of Random Forest and Gaussian Naive Bayes algorithms used for training and predicting water quality.

The config.py file defines the paths and settings for the project.

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
The dataset used in this project is data.csv, which contains features such as temperature, pH, and light intensity of water samples, as well as the corresponding water quality index (WQI) values.

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
Gaussian Naive Bayes
Model evaluation
Model optimization and hyperparameter tuning
Random Forest
Gaussian Naive Bayes
Model deployment and prediction
The utilities.py file contains the following utility functions for the project:

remove_outliers: Removes outliers from a dataset based on a given column
visualize_correlation: Visualizes the correlation between the features in the dataset
plot_distribution: Plots the distribution of a given column
plot_count: Plots the count of values in a given column
plot_pie: Plots a pie chart of the values in a given column
plot_kde: Plots a KDE plot of the values in a given column
plot_data: Plots the data in a given column
plot_heatmap: Plots a heatmap of the correlation between the features in the dataset
The models.py file contains the following machine learning models for the project:

RandomForestClassifier: A random forest classifier for predicting water quality
GaussianNB: A Gaussian naive Bayes classifier for predicting water quality
The config.py file defines the following paths and settings for the project:

DATA_PATH: Path to the dataset file
MODELS_PATH: Path to the models directory
Conclusion
This project demonstrates the prediction of water quality based on various parameters such as temperature, pH, and light intensity using machine learning algorithms. The project includes data preprocessing, EDA, model training, and prediction, and provides utility functions for data visualization and analysis.
