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
cd airline-satisfaction-prediction
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
6. **Decision Tree**

Each model was evaluated using metrics such as:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

**Model Selection**: AUC-ROC was conducted on the model. Random Forest Classifier demonstrates the best performance among the multiple classifiers as it is the closest to the top left corner. It has an AUC of 0.99


## Results
- **Best Model**: Random Forest Classifier
- **Accuracy**: 0.94
- **Precision**: 0.96
- **Recall**: 0.94
- **F1-Score**: 0.94
- **AUC-ROC**: 0.99

Visualization of results:
- Confusion Matrix
- ROC Curve

## Conclusion
The classification model provides insights into the factors contributing to airline passenger satisfaction. By focusing on key elements such as in-flight service, seat comfort, and delays, airlines can improve customer satisfaction levels and potentially increase customer loyalty.

Future work:
- Testing with larger datasets
- Implementing deep learning techniques

## Technologies Used
- **Programming Language**: Python
- **Data Analysis**: Pandas, NumPy
- **Data Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn, XGBoost
- **Development Environment**: Google Colab
- **Version Control**: Git

## Contact
For more information or collaboration opportunities, feel free to contact me:

- **Name**: Ojo Timilehin
- **Email**: Ojotimilehin01@gmail.com
- **Portfolio**: https://ojotimilehin01.wixsite.com/ojotimi 
