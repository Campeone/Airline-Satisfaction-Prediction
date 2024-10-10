# Airline Satisfaction Prediction

## Overview
This project aims to build a classification model that predicts airline passenger satisfaction based on various features such as flight distance, departure and arrival time convenience, in-flight services, seat comfort, and more. The model helps airlines enhance customer experience by identifying key factors affecting satisfaction.

The analysis leverages machine learning techniques to classify passengers as either satisfied or dissatisfied with their flight experience.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Modeling](#modeling)
6. [Results](#results)
7. [Usage](#usage)
8. [Conclusion](#conclusion)
9. [Technologies Used](#technologies-used)
10. [Contact](#contact)

## Project Structure
```
├── data/               # Dataset folder
├── notebooks/          # Jupyter notebooks for analysis
├── models/             # Saved machine learning models
├── images/             # Images used in the README or reports
├── README.md           # Project documentation
└── requirements.txt    # Dependencies for the project
```

## Installation
To run this project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/airline-satisfaction-prediction.git
cd airline-satisfaction-analysis
```

Create a virtual environment and activate it:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

Install the dependencies:
```bash
pip install -r requirements.txt
```

## Dataset
The dataset used for this analysis contains features that describe various aspects of a passenger's flight experience. Key features include:

- **Flight Distance**
- **Departure Delay**
- **Arrival Delay**
- **Seat Comfort**
- **In-flight Service**
- **Age**
- **Class** (Business, Economy, Economy Plus)

- **Target**: Satisfaction (Satisfied / Dissatisfied)

**Source**: [https://www.kaggle.com/datasets/sjleshrac/airlines-customer-satisfaction]

## Exploratory Data Analysis (EDA)
- Univariate analysis was performed on individual features to understand their distribution.
- Bivariate analysis was used to investigate the relationship between features and the target variable.
- Data preprocessing involved handling missing values, encoding categorical variables, and feature scaling.

Visualizations include:
- Correlation heatmaps
- Feature importance
- Satisfaction distribution

## Modeling
The following machine learning models were evaluated:
1. **Logistic Regression**
2. **Random Forest**
3. **Gradient Boosting**
4. **Ada Boost**
5. **Naive Bayes**
6. **Decison Tree**

Each model was evaluated using metrics such as:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

**Model Selection**: AUCThe model with the highest performance metrics was selected, and hyperparameter tuning was done using grid search or random search.

## Results
- **Best Model**: [e.g., Random Forest Classifier]
- **Accuracy**: [insert value]
- **Precision**: [insert value]
- **Recall**: [insert value]
- **F1-Score**: [insert value]
- **AUC-ROC**: [insert value]

Visualization of results:
- Confusion Matrix
- ROC Curve

## Usage
To make predictions on new data, use the trained model by running the following command:
```bash
python src/predict.py --input data/new_passenger_data.csv
```

Example:
```
| Passenger ID | Age | Flight Distance | Seat Comfort | Predicted Satisfaction |
|--------------|-----|-----------------|--------------|------------------------|
| 001          | 35  | 1500            | 4.5          | Satisfied               |
| 002          | 42  | 900             | 3.0          | Dissatisfied            |
```

## Conclusion
The classification model provides insights into the factors contributing to airline passenger satisfaction. By focusing on key elements such as in-flight service, seat comfort, and delays, airlines can improve customer satisfaction levels and potentially increase customer loyalty.

Future work:
- Testing with larger datasets
- Implementing deep learning techniques
- Including more features (e.g., customer reviews, flight duration)

## Technologies Used
- **Programming Language**: Python
- **Data Analysis**: Pandas, NumPy
- **Data Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn, XGBoost
- **Development Environment**: Jupyter Notebook
- **Version Control**: Git

## Contact
For more information or collaboration opportunities, feel free to contact me:

- **Name**: [Your Name]
- **Email**: [Your Email]
- **Portfolio**: [Link to your portfolio]

---

This README provides a clear, structured overview of your project. You can modify it based on the exact content and findings from your notebook. Would you like to add any additional sections or details to the README?
