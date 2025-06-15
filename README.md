# Spot the Scam - Fraud Job Detection

## Project Overview

**Spot the Scam** is a Streamlit-based web application that detects fraudulent job postings using machine learning. The app is designed to help users identify scam job listings by analyzing text features and leveraging advanced resampling techniques to address severe class imbalance (95% real, 5% fake jobs). It provides interactive analytics, single and bulk prediction, and visualizations to support decision-making.

---

## Key Features

- **Automatic Data Cleaning**: Cleans and combines multiple text fields for robust feature extraction.
- **Class Imbalance Handling**: Uses SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset, improving detection of rare fraudulent jobs.
- **Model Training & Tuning**: Trains a Random Forest classifier with hyperparameter tuning (number of trees adjustable via sidebar).
- **Performance Metrics**: Displays F1 score and visual analytics for model evaluation.
- **Interactive Visualizations**:
  - Top 10 most suspicious job listings.
  - Histogram of predicted fraud probabilities.
  - Pie chart of predicted real vs. fake jobs.
- **Single & Bulk Prediction**:
  - Predict fraud probability for a single job posting via form.
  - Upload a CSV for batch predictions (no 'fraudulent' column required).
- **User-Friendly Interface**: Built with Streamlit for easy interaction and visualization.

---

## Technologies Used

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend**: Python
- **Libraries**:
  - `pandas`, `numpy` for data handling
  - `scikit-learn` for machine learning and evaluation
  - `imblearn` for SMOTE oversampling
  - `matplotlib` for plotting

---

## Setup Instructions

1. **Download the project files.**
2. **Install required libraries** in Command Prompt:
   ```
   pip install streamlit pandas numpy matplotlib scikit-learn imbalanced-learn
   ```
3. **Change directory** to the project folder (example):
   ```
   cd C:\Marvin\College\IITG_DS_COURSE\25_June_Hackathon
   ```
4. **Run the app** using Streamlit:
   ```
   streamlit run app.py
   ```
5. **Interact with the app** in your browser:
   - View analytics and suspicious listings.
   - Predict single or bulk job postings.
   - Upload your own CSV for batch predictions.

---

## Dataset

- The app expects a dataset named `fake_job_postings.csv` in the project directory.
- Required columns: `title`, `description`, `company_profile`, `requirements`, `benefits`, `location`, `fraudulent`.

---

## Why It Matters

Fraudulent job postings are a real threat to job seekers. With severe class imbalance in real-world data, traditional models often fail to detect rare fraud cases. This project uses SMOTE and class-weighted learning to ensure fair, accurate detection—helping users and organizations spot scams before they cause harm.

---

## Credits

Developed by Team Data Disciples for IITG_DS_COURSE Hackathon, June 2025.

---

**Tip:** For best results, ensure your input data matches the expected format. The app automatically handles missing values and text cleaning.