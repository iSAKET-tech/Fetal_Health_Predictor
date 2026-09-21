# Fetal Health Predictor

A machine learning-based application for classifying fetal health into **Normal, Suspect, and Pathological** categories using cardiotocography (CTG) data.

The project covers the complete machine learning workflow, including data exploration, preprocessing, model development, evaluation, feature importance analysis, cross-validation, hyperparameter tuning, and deployment through a Streamlit web application.

> **Disclaimer:** This project is developed for educational and research purposes. It is not a clinically validated diagnostic system and should not be used for medical decision-making.

---

## Project Overview

Cardiotocography (CTG) is used to monitor fetal heart rate and uterine activity. The resulting measurements contain patterns that can be used to classify fetal health conditions.

This project uses CTG-related features to build a machine learning classification model capable of predicting one of three fetal health categories:

- **Normal**
- **Suspect**
- **Pathological**

The final model is integrated into a Streamlit application where users can enter CTG parameters and obtain a predicted fetal health class along with model probability outputs.

---

## Objectives

The main objectives of this project are:

- Analyze and understand CTG data
- Perform exploratory data analysis
- Validate and preprocess the dataset
- Analyze class imbalance
- Build a baseline classification model
- Train a Random Forest classifier
- Compare model performance
- Analyze feature importance
- Experiment with feature reduction
- Perform stratified cross-validation
- Tune Random Forest hyperparameters
- Select a final model
- Build an interactive Streamlit prediction application

---

## Dataset

The project uses a fetal health classification dataset containing cardiotocography measurements.

### Dataset Statistics

| Property | Value |
|---|---:|
| Total Records | 2,126 |
| Input Features | 21 |
| Target Feature | `fetal_health` |
| Missing Values | 0 |
| Duplicate Rows | 0 |
| Data Type | Numerical |

---

## Target Classes

The target variable is `fetal_health`.

| Label | Category |
|---:|---|
| 1 | Normal |
| 2 | Suspect |
| 3 | Pathological |

### Class Distribution

| Class | Number of Records |
|---|---:|
| Normal | 1,655 |
| Suspect | 295 |
| Pathological | 176 |

The dataset is imbalanced, with the Normal class representing the majority of observations.

Because of this imbalance, model evaluation does not rely only on accuracy. Precision, recall, F1-score, macro F1, and confusion matrices are also considered.

---

# Features

The model uses 21 CTG-related features.

```text
baseline value
accelerations
fetal_movement
uterine_contractions
light_decelerations
severe_decelerations
prolongued_decelerations
abnormal_short_term_variability
mean_value_of_short_term_variability
percentage_of_time_with_abnormal_long_term_variability
mean_value_of_long_term_variability
histogram_width
histogram_min
histogram_max
histogram_number_of_peaks
histogram_number_of_zeroes
histogram_mode
histogram_mean
histogram_median
histogram_variance
histogram_tendency
