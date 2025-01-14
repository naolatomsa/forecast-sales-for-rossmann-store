# Rossmann Store Sales - Task 1: Customer Purchasing Behavior

## Overview
This repository contains the exploratory data analysis (EDA) for Rossmann store sales, focusing on customer purchasing behavior. Task 1 aims to uncover insights into how factors such as promotions, holidays, and competition influence sales.

## Objectives
- Analyze and visualize customer behavior trends.
- Investigate the impact of promotions, holidays, and competitor activity on sales.
- Provide actionable insights for business decision-making.

## Deliverables
- **Notebook**: Detailed EDA with visualizations and insights.
- **Python Script**: Modular and reusable functions for EDA tasks.
- **Logging**: Traceable steps for reproducibility.


# Task 2: Prediction of Store Sales

    Preprocessing:
        Convert non-numeric columns, handle missing values, and generate features (e.g., weekdays, holidays).
        Scale data using StandardScaler.

    Model Building:
        Use RandomForest Regressor within an sklearn pipeline for regression modeling.

    Loss Function:
        Select an appropriate loss function for regression (e.g., Mean Squared Error).

    Post Prediction Analysis:
        Analyze feature importance and estimate confidence intervals.

    Model Serialization:
        Save models with timestamps for tracking predictions.

    Deep Learning Model:
        Build a simple LSTM model for time-series forecasting using TensorFlow/PyTorch.

# Task 3: Model Serving API Call

    Create REST API:
        Use FastAPI to serve the trained model for real-time predictions.

    Define API Endpoints:
        Create endpoints to handle input data and return predictions.

    Handle Requests:
        Preprocess data, make predictions using the model, and return results.

    Deployment:
        Deploy the API to a server or cloud platform for live use.

## How to Use
1. Clone the repository:
   \`\`\`bash
   git clone https://github.com/yourusername/rossmann-store-sales.git
   \`\`\`
2. Install dependencies:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`
