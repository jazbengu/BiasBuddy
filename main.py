
# app.py
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Load Idol Data
@st.cache
def load_data():
    return pd.read_csv("idols_data.csv")  # Ensure idols_data.csv exists with relevant fields

# Content-Based Filtering
def recommend_bias(user_traits, idols_data):
    # Scaling numerical features
    scaler = StandardScaler()
    features = ["age", "popularity", "vocal_skill", "dance_skill", "rap_skill"]
    scaled_data = scaler.fit_transform(idols_data[features])
    scaled_user = scaler.transform([user_traits])
    
    # Compute similarity
    similarities = cosine_similarity(scaled_user, scaled_data)
    idols_data["Similarity"] = similarities[0]
    return idols_data.sort_values(by="Similarity", ascending=False).head(5)

# Streamlit UI
st.title("BiasBuddy 🎤")
st.subheader("Find Your Ultimate K-pop Bias!")
st.write("Input your preferences, and we'll recommend your perfect match from the K-pop world!")

# User Inputs
st.sidebar.header("Your Preferences")
user_age = st.sidebar.slider("Your Age", min_value=10, max_value=50, value=20)
user_popularity = st.sidebar.slider("Preference for Popularity (1-10)", 1, 10, 5)
vocal_pref = st.sidebar.slider("Vocal Skill Preference (1-10)", 1, 10, 5)
dance_pref = st.sidebar.slider("Dance Skill Preference (1-10)", 1, 10, 5)
rap_pref = st.sidebar.slider("Rap Skill Preference (1-10)", 1, 10, 5)

if st.sidebar.button("Find My Bias ✨"):
    # Load data
    idols_data = load_data()
    
    # Get user traits
    user_traits = [user_age, user_popularity, vocal_pref, dance_pref, rap_pref]
    
    # Recommend biases
    recommendations = recommend_bias(user_traits, idols_data)
    
    # Display results
    st.subheader("Your Top K-pop Biases")
    for idx, row in recommendations.iterrows():
        st.write(f"**{row['name']}** from **{row['group']}** - Similarity: {row['Similarity']:.2f}")
        st.image(row['image_url'], width=150)  # Ensure the dataset has image URLs

st.sidebar.markdown("---")
st.sidebar.info("Built with ❤️ for K-pop fans!")

