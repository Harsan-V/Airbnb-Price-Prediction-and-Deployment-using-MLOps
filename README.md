# Airbnb Price Prediction and Deployment using MLOps

## Overview

This project implements an end-to-end MLOps pipeline for predicting Airbnb listing prices across major Indian cities using machine learning.

The system collects Airbnb listing data, performs feature engineering, trains predictive models, tracks model performance, and deploys the best model through a FastAPI-based REST API.

The application also includes prediction monitoring, logging, model health checks, and performance reporting to support production-ready machine learning workflows.

## I used the Airbnb data

<img width="1469" height="834" alt="Screenshot 2026-06-15 at 10 45 53 PM" src="https://github.com/user-attachments/assets/a3ce86f9-e103-4c2d-a212-babeab2c1f48" />


## Key Features

- Airbnb listing price prediction
- Multi-city Indian Airbnb dataset
- FastAPI deployment
- Model performance tracking
- Prediction monitoring and logging
- Automated input validation
- Health check endpoints
- Model metrics reporting
- Production-ready API architecture

## Supported Cities

- Chennai
- Bangalore
- Hyderabad
- Mumbai
- New Delhi

## Technology Stack

### Data Processing
- Python
- Pandas
- NumPy

### Machine Learning
- Scikit-Learn
- Random Forest Regressor
- Ridge Regressor

### MLOps
- DVC
- GitHub Actions

### Deployment
- FastAPI

### Monitoring
- Prediction Logging
- Model Health Checks
- Performance Reporting

## API Endpoints

| Endpoint | Description |
|-----------|-------------|
| `/health` | Service health check |
| `/predict` | Predict Airbnb listing price |
| `/features` | View model features |
| `/model/metrics` | View model metrics |
| `/monitoring/summary` | Prediction monitoring summary |
| `/monitoring/logs` | Recent prediction logs |
| `/monitoring/training-report` | Training monitoring report |

## Project Workflow

Data Collection → Data Validation → Feature Engineering → Model Training → Model Evaluation → Model Deployment → Monitoring → Continuous Improvement

## Output
<img width="1470" height="956" alt="Screenshot 2026-06-16 at 12 37 16 PM" src="https://github.com/user-attachments/assets/30ad3114-432a-4f37-ab27-bb31ad59c412" />
<img width="1470" height="956" alt="Screenshot 2026-06-16 at 12 37 42 PM" src="https://github.com/user-attachments/assets/c143f24b-a88f-4a4e-ad26-05f9582891a8" />
<img width="1470" height="956" alt="Screenshot 2026-06-16 at 12 38 47 PM" src="https://github.com/user-attachments/assets/36691da3-956f-4a2f-a1d8-395465f42ea2" />
<img width="1470" height="956" alt="Screenshot 2026-06-16 at 12 39 06 PM" src="https://github.com/user-attachments/assets/bd163dd0-f804-47b9-a4ca-51d00ed80a52" />





