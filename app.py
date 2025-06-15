import base64
import io
import time
import re
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE

# --- Naukri Webscraper Function ---
from datetime import datetime
from bs4 import BeautifulSoup
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def scrape_naukri_job(job_url):
    st.info(f"🔍 Scraping: {job_url}")
    options = uc.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    driver = uc.Chrome(options=options)
    driver.get(job_url)

    try:
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
    except Exception as e:
        driver.save_screenshot("debug.png")
        driver.quit()
        st.error("⚠️ Could not load job page. Possibly blocked or slow connection.")
        return None

    time.sleep(5)  # wait for dynamic content

    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    page_text = soup.get_text(separator="\n")

    def extract_between(text, start_label, end_label=None):
        try:
            start_idx = text.index(start_label) + len(start_label)
            if end_label:
                end_idx = text.index(end_label, start_idx)
                return text[start_idx:end_idx].strip()
            return text[start_idx:].strip()
        except ValueError:
            return "N/A"

    # Map webscraped fields to your model's expected columns
    data = {
        "title": extract_between(page_text, "Sales Officer", "\nHDB"),  # crude fallback
        "company_profile": extract_between(page_text, "About company", "Awarded by").replace("\n", ", "),
        "description": extract_between(page_text, "Job description", "Roles and Responsibilities"),
        "requirements": extract_between(page_text, "Key Skills", "About company").replace("\n", ", "),
        "benefits": extract_between(page_text, "Benefits", "\n"),  # This may not exist, fallback to N/A
        "location": extract_between(page_text, "P.A.", "\nPosted").replace(",", " "),
    }
    # Fallbacks for missing fields
    for k in data:
        if not data[k] or data[k] == "N/A":
            data[k] = ""
    return data

# --- Streamlit App ---
st.set_page_config(page_title="Fraud Job Detector", layout="wide")
st.title("Spot the Scam - Fraud Job Detection")

# Load and preprocess training data from file
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv(r"C:\Marvin\College\IITG_DS_COURSE\25_June_Hackathon\fake_job_postings.csv")
    def clean_text(text):
        if pd.isna(text): return ""
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        return re.sub(r'\s+', ' ', text).strip()
    text_columns = ['title', 'description', 'company_profile', 'requirements', 'benefits', 'location']
    for col in text_columns:
        df[col] = df[col].astype(str).apply(clean_text)
    df['text'] = df[text_columns].agg(' '.join, axis=1)
    df = df.dropna(subset=['text', 'fraudulent'])
    return df

# Model training function
@st.cache_resource
def train_model(df, n_estimators):
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['fraudulent'], test_size=0.2, stratify=df['fraudulent'], random_state=42
    )
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_train_vec, y_train)
    model = RandomForestClassifier(n_estimators=n_estimators, class_weight='balanced', random_state=42)
    model.fit(X_resampled, y_resampled)
    probs = model.predict_proba(X_test_vec)[:, 1]
    preds = (probs > 0.3).astype(int)
    f1 = f1_score(y_test, preds)
    return model, vectorizer, f1, probs, y_test, X_test

# Hyperparameter tuning slider
n_estimators = st.sidebar.slider("Number of Trees (n_estimators)", 100, 500, step=50, value=200)

# Load data and train model
with st.spinner("Training model..."):
    df = load_and_preprocess_data()
    model, vectorizer, f1, probs, y_test, X_test = train_model(df, n_estimators)

st.success(f"Model Ready - Current F1 Score: {f1:.2f}")

# Results Table
results = X_test.reset_index().copy()
results['fraudulent'] = y_test.reset_index(drop=True)
results['probability'] = probs
results['prediction'] = (probs > 0.5).astype(int)
top_suspicious = results.sort_values('probability', ascending=False).head(10)

st.subheader("Top 10 Most Suspicious Listings")
st.dataframe(top_suspicious)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Histogram of Fraud Probabilities")
    fig, ax = plt.subplots()
    ax.hist(probs, bins=20, color='skyblue', edgecolor='black')
    ax.set_title("Histogram of Fraud Probabilities")
    ax.set_xlabel("Probability")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

with col2:
    st.subheader("🥧 Pie Chart: Predicted Fake vs Real Jobs")
    pred_counts = results['prediction'].value_counts().sort_index()
    fig2, ax2 = plt.subplots()
    ax2.pie(pred_counts, labels=['Real (0)', 'Fake (1)'], autopct='%1.1f%%', colors=['#8fd9b6','#ff9999'])
    ax2.set_title("Predicted Job Types")
    st.pyplot(fig2)

# User Input Section
st.subheader("Predict a New Job Posting")
with st.form("predict_form"):
    title = st.text_input("Job Title")
    description = st.text_area("Job Description")
    company_profile = st.text_area("Company Profile")
    requirements = st.text_area("Requirements")
    benefits = st.text_area("Benefits")
    location = st.text_input("Location")
    submitted = st.form_submit_button("Predict")

if submitted:
    new_text = ' '.join([title, description, company_profile, requirements, benefits, location])
    new_text_clean = re.sub(r'[^a-z\s]', '', new_text.lower())
    vec = vectorizer.transform([new_text_clean])
    prob = model.predict_proba(vec)[0][1]
    prediction = int(prob > 0.5)
    label = "Fake" if prediction == 1 else "Real"
    st.markdown(f"### 🔍 Prediction: **{label}** ({prob:.2f} Probability)")

# --- Naukri URL Prediction Section ---
st.subheader("🌐 Predict Directly from a Naukri Job URL")
with st.form("naukri_url_form"):
    naukri_url = st.text_input("Paste Naukri Job URL")
    url_submitted = st.form_submit_button("Scrape & Predict")

if url_submitted and naukri_url:
    with st.spinner("Scraping job details from Naukri..."):
        scraped = scrape_naukri_job(naukri_url)
    if scraped:
        # Show scraped fields
        st.write("**Scraped Job Fields:**")
        st.json(scraped)
        # Prepare text for model
        text_fields = [
            scraped.get("title", ""),
            scraped.get("description", ""),
            scraped.get("company_profile", ""),
            scraped.get("requirements", ""),
            scraped.get("benefits", ""),
            scraped.get("location", "")
        ]
        new_text = ' '.join(text_fields)
        new_text_clean = re.sub(r'[^a-z\s]', '', new_text.lower())
        vec = vectorizer.transform([new_text_clean])
        prob = model.predict_proba(vec)[0][1]
        prediction = int(prob > 0.5)
        label = "Fake" if prediction == 1 else "Real"
        st.markdown(f"### 🔍 Prediction: **{label}** ({prob:.2f} Probability)")
    else:
        st.error("Failed to scrape the job posting. Please check the URL or try again later.")

# Batch Prediction Section
st.subheader("📄 Upload CSV for Bulk Prediction")
csv_file = st.file_uploader("Upload CSV containing multiple job listings (no 'fraudulent' column required)", type=['csv'])

if csv_file is not None:
    try:
        bulk_df = pd.read_csv(csv_file)
        expected_cols = ['title', 'description', 'company_profile', 'requirements', 'benefits', 'location']
        missing_cols = [col for col in expected_cols if col not in bulk_df.columns]
        if missing_cols:
            st.error(f"Missing columns in uploaded file: {', '.join(missing_cols)}")
        else:
            def clean_text(text):
                if pd.isna(text): return ""
                text = text.lower()
                text = re.sub(r'http\S+', '', text)
                text = re.sub(r'[^a-z\s]', '', text)
                return re.sub(r'\s+', ' ', text).strip()
            for col in expected_cols:
                bulk_df[col] = bulk_df[col].astype(str).apply(clean_text)
            bulk_df['text'] = bulk_df[expected_cols].agg(' '.join, axis=1)
            vecs = vectorizer.transform(bulk_df['text'])
            probs_bulk = model.predict_proba(vecs)[:, 1]
            preds_bulk = (probs_bulk > 0.5).astype(int)
            bulk_df['Prediction'] = ['Fake' if p else 'Real' for p in preds_bulk]
            bulk_df['Probability'] = probs_bulk
            st.success("✅ Bulk predictions complete")
            st.dataframe(bulk_df[['title', 'Prediction', 'Probability']])
    except Exception as e:
        st.error(f"Failed to process the file: {e}")