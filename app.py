import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Tourism Analytics", layout="wide")

st.title("🌍 Tourism Experience Analytics Dashboard")

# LOAD MODELS
@st.cache_resource
def load_models():
    models = {}
    models["linear_reg"] = joblib.load("models/linear_regression_model.joblib")
    models["xgb_reg"] = joblib.load("models/xgboost_model.joblib")
    models["logistic_clf"] = joblib.load("models/logistic_regression_model.joblib")
    models["xgb_clf"] = joblib.load("models/xgboost_classification_model.joblib")
    models["predicted_ratings"] = joblib.load("models/predicted_ratings.pkl")
    models["cosine_sim"] = joblib.load("models/cosine_sim.pkl")
    models["content_df"] = joblib.load("models/content_df.pkl")
    models["indices"] = joblib.load("models/indices.pkl")
    return models

models = load_models()

# SIDEBAR NAVIGATION
menu = st.sidebar.selectbox(
    "Select Module",
    [
        "Home",
        "Regression - Rating Prediction",
        "Classification - Visit Mode",
        "Collaborative Recommender",
        "Content-Based Recommender"
    ]
)

# HOME
if menu == "Home":
    st.write("""
    ### Welcome to Tourism Experience Analytics
    
    This system includes:
    - ⭐ Attraction Rating Prediction
    - 🧠 Visit Mode Classification
    - 🤝 Collaborative Filtering Recommender
    - 📍 Content-Based Recommender
    """)

# REGRESSION MODULE
elif menu == "Regression - Rating Prediction":
    
    st.header("⭐ Predict Attraction Rating")
    
    model_choice = st.selectbox(
        "Choose Regression Model",
        ["Linear Regression", "XGBoost Regressor"]
    )
    
    # --- INPUT FEATURES ---
    continent = st.text_input("Continent")
    region = st.text_input("Region")
    country = st.text_input("Country")
    city = st.text_input("City")
    visit_year = st.number_input("Visit Year", 2000, 2030)
    visit_month = st.slider("Visit Month", 1, 12)
    attraction_type = st.text_input("Attraction Type")
    
    if st.button("Predict Rating"):
        
        input_df = pd.DataFrame([{
            "Continent": continent,
            "Region": region,
            "Country": country,
            "CityName": city,
            "VisitYear": visit_year,
            "VisitMonth": visit_month,
            "AttractionType": attraction_type
        }])
        
        if model_choice == "Linear Regression":
            prediction = models["linear_reg"].predict(input_df)
        else:
            prediction = models["xgb_reg"].predict(input_df)
        
        st.success(f"Predicted Rating: {round(prediction[0], 2)}")

# CLASSIFICATION MODULE
elif menu == "Classification - Visit Mode":
    
    st.header("👨‍👩‍👧 Predict Visit Mode")
    
    model_choice = st.selectbox(
        "Choose Classification Model",
        ["Logistic Regression", "XGBoost Classifier"]
    )
    
    continent = st.text_input("Continent")
    region = st.text_input("Region")
    country = st.text_input("Country")
    city = st.text_input("City")
    visit_year = st.number_input("Visit Year", 2000, 2030)
    visit_month = st.slider("Visit Month", 1, 12)
    attraction_type = st.text_input("Attraction Type")
    
    if st.button("Predict Visit Mode"):
        
        input_df = pd.DataFrame([{
            "Continent": continent,
            "Region": region,
            "Country": country,
            "CityName": city,
            "VisitYear": visit_year,
            "VisitMonth": visit_month,
            "AttractionType": attraction_type
        }])
        
        if model_choice == "Logistic Regression":
            prediction = models["logistic_clf"].predict(input_df)
        else:
            prediction = models["xgb_clf"].predict(input_df)
        
        st.success(f"Predicted Visit Mode: {prediction[0]}")

# COLLABORATIVE FILTERING
elif menu == "Collaborative Recommender":
    
    st.header("🤝 Collaborative Filtering Recommendations")
    
    user_id = st.number_input("Enter User ID", min_value=1)
    
    if st.button("Recommend"):
        
        try:
            predictions = models["predicted_ratings"].loc[user_id]
            top_items = predictions.sort_values(ascending=False).head(5)
            # Convert to DataFrame
            top_df = top_items.reset_index()
            top_df.columns = ["AttractionId", "PredictedRating"]
            
            # Merge with attraction names
            results = top_df.merge(
                models["content_df"][["AttractionId", "Attraction"]],
                on="AttractionId",
                how="left"
            )
            
            st.dataframe(results[["Attraction", "PredictedRating"]])
        except:
            st.error("User ID not found.")

# CONTENT-BASED FILTERING
elif menu == "Content-Based Recommender":
    
    st.header("📍 Similar Attraction Recommendations")
    
    attraction_id = st.number_input("Enter Attraction ID")
    
    if st.button("Recommend Similar Attractions"):
        
        try:
            idx = models["indices"][attraction_id]
            sim_scores = list(enumerate(models["cosine_sim"][idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
            
            attraction_indices = [i[0] for i in sim_scores]
            
            results = models["content_df"].iloc[attraction_indices][
                ["AttractionId","Attraction"]
            ]
            
            st.write(results)
        
        except:
            st.error("Attraction ID not found.")