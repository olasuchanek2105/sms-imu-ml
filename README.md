# Space Motion Sickness Risk Prediction (IMU-based ML Pipeline)

This project presents a modular machine learning pipeline for predicting the risk of Space Motion Sickness (SMS) based on inertial measurement unit (IMU) data.
The project was developed as part of an engineering thesis in Biomedical Engineering and focuses on signal preprocessing, feature extraction, classical ML models, and model evaluation.

## Problem Description
Space Motion Sickness (SMS) is a common issue during spaceflight and simulator-based motion exposure.  
The goal of this project is to assess the risk of SMS using head movement data acquired from IMU sensors and to evaluate the effectiveness of classical machine learning models for this task.

## Project Structure
data/                  # IMU datasets (not included / anonymized)
preprocessing/         # Signal preprocessing (filtering, segmentation)
feature_extraction/    # Feature extraction from IMU signals
models/                # ML model definitions
evaluation/            # Model evaluation and metrics
experiments/           # Experimental setups and comparisons
utils/                 # Shared utility functions
scripts/               # Helper scripts
train.py               # Main training pipeline
requirements.txt       # Python dependencies


## Methods & Technologies
- **Programming language:** Python  
- **Libraries:** NumPy, Pandas, SciPy, scikit-learn  
- **Signal processing:** filtering, windowing, feature extraction  
- **Machine learning:** Random Forest, SVM, Logistic Regression  
- **Evaluation:** cross-validation, classification metrics  

## ML Pipeline Overview
1. Data preprocessing (filtering, segmentation into time windows)
2. Feature extraction from IMU signals
3. Model training using classical ML algorithms
4. Model evaluation and comparison
5. Experimental analysis


## How to Run
```bash
pip install -r requirements.txt
python train.py


