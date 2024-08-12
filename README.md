Data Analysis Project README
Overview
This project is part of my data analyst portfolio, focusing on building a machine learning model to predict car prices based on data scraped from MercadoLibre. The project includes web scraping, data cleaning, exploratory data analysis, and predictive modeling.

Project Structure
Data Collection: Web scraping car listings from MercadoLibre to create a dataset.
Data Cleaning: Removing outliers and cleaning the data for better analysis.
Exploratory Data Analysis (EDA): Analyzing the distribution of variables and their relationships.
Modeling: Implementing various regression models to predict car prices.
Evaluation: Evaluating the performance of different models using metrics like MSE, MAE, and R².
Requirements
Python 3.x
Required Libraries:
requests
beautifulsoup4
pandas
matplotlib
seaborn
scikit-learn
xgboost
scipy
Files
main.py: The main script that contains all functions and executes the scraping, cleaning, EDA, and modeling tasks.
README.md: Documentation of the project (this file).
Usage
Web Scraping:

The script scrapes car listings from MercadoLibre.
It fetches the dollar exchange rate from DolarHoy and converts car prices from USD to ARS.
Data Cleaning:

Removes listings with 0 km.
Handles outliers in price, year, and km.
EDA:

Visualizes distributions of price, year, and km.
Examines the relationship between variables.
Shows correlation metrics.
Modeling:

Divides the dataset into training and testing sets.
Implements models including Linear Regression, Decision Tree, Random Forest, SVR, KNN, ElasticNet, and XGBoost.
Uses RandomizedSearchCV for hyperparameter tuning.
Model Evaluation:

Evaluates models using MSE, MAE, and R² metrics.
Compares the performance of different models.
Example Output
Execution Time:
Scraping time, data cleaning time, and model training time are printed to the console.
Model Performance:
The best model and its parameters are displayed.
Evaluation metrics for each model are printed.
Future Improvements
Implement additional data preprocessing techniques.
Explore other machine learning models or deep learning approaches.
Integrate the model into a web application for real-time car price predictions.
License
This project is open-source and available under the MIT License.
