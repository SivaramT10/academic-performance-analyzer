# academic-performance-analyzer


## Overview
A data-driven tool that analyzes student academic records and predicts
performance risk using statistical analysis and basic machine learning.

## Problem Statement
Early identification of academically at-risk students can help institutions
take corrective actions and improve learning outcomes.

## Features
- Academic data preprocessing
- Performance risk prediction
- Explainable machine learning model
- Text-based analysis report

## Technologies Used
- Python
- Pandas
- Scikit-learn

## Approach
Student attendance and assessment scores are used as input features.
A logistic regression model is trained to predict whether a student is
academically safe or at risk.

## Dataset
The dataset used in this project is sourced from publicly available
academic performance datasets for educational purposes.

Due to size and licensing considerations, the dataset is not included
in this repository.

Example sources:
- UCI Machine Learning Repository
- Kaggle (public academic datasets)

Users can replace the dataset path in the code with any compatible
academic performance dataset following the expected format.

## Expected Dataset Format
The dataset should contain the following columns:
- attendance
- midterm
- assignment
- final
- result (1 = Safe, 0 = At Risk)

## Output
The program generates a `report.txt` file containing:
- Model accuracy
- Risk prediction summary for test samples

## Limitations
- Small or synthetic datasets
- Academic indicators only
- Not intended for real-world deployment without validation

## Status
Self-learning / academic project
