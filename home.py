import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import os

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="AI Sales Predictor", page_icon="🤖", layout="centered")

# =========================================================
# PAGE STATE MANAGER
# =========================================================
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to(page_name):
    st.session_state.page = page_name

# =========================================================
# COMMON BACKGROUND CSS
# =========================================================
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://wallpapers.com/images/hd/black-blur-background-ai0plsbfayz8go0c.jpg");
    background-size: cover;
    background-position: center;
}
[data-testid="stAppViewContainer"]::before {
    content: "";
    background: rgba(0,0,0,0.55); 
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    backdrop-filter: blur(6px);
    z-index: -1;
}
.glass-box {
    background: rgba(255,255,255,0.17);
    padding: 38px;
    border-radius: 20px;
    backdrop-filter: blur(0px) !important;
    box-shadow: 0 8px 40px rgba(0,0,0,0.4);
    width: 520px;
    margin-top: 120px;
    text-align: center;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# =========================================================
# LOAD MODEL (same for all pages)
# =========================================================
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(__file__)
    MODEL_PATH = os.path.join(BASE_DIR, "multiple_reg_model.pkl")
    model = pickle.load(open(MODEL_PATH, "rb"))
    return model

model = load_model()

# =========================================================
# PAGE 1 → HOME PAGE
# =========================================================
if st.session_state.page == "home":

    st.markdown("<h1 style='color:white;font-size:45px;'>🤖 AI Sales Predictor</h1>", unsafe_allow_html=True)

    st.write("""
    ### 📌 What this model does?
    ✔ Predicts sales based on marketing budget  
    ✔ Uses Machine Learning  
    ✔ Gives accurate forecast instantly ⚡  

    ### 📌 Why use it?
    ✔ Helps in smarter business decisions  
    ✔ Allocates marketing budget better  
    ✔ Boosts profit 💰  

    ### 📌 Easy to use  
    Just enter TV, Radio, Newspaper values → Get prediction 📊  
    """)

    if st.button("🚀 Get Started"):
        go_to("appp")

    if st.button("📠 To Know Model"):
        go_to("about")

# =========================================================
# PAGE 2 → APP (Predictor)
# =========================================================
elif st.session_state.page == "appp":

    st.markdown("<h2 class='glow-title'>📊 Sales Predictor</h2>", unsafe_allow_html=True)

    tv = st.number_input("TV Budget (₹)", min_value=0.0, step=0.1)
    radio = st.number_input("Radio Budget (₹)", min_value=0.0, step=0.1)
    newspaper = st.number_input("Newspaper Budget (₹)", min_value=0.0, step=0.1)

    if st.button("🔮 Predict Sales"):
        input_data = np.array([[tv, radio, newspaper]])
        result = float(model.predict(input_data)[0])
        st.success(f"✨ Predicted Sales: **{result:.2f} units**")

    if st.button("🏡 Go Home"):
        go_to("home")

    if st.button("📘 About Model"):
        go_to("about")

# =========================================================
# PAGE 3 → ABOUT MODEL (full content)
# =========================================================
elif st.session_state.page == "about":

    st.markdown('<div class="glass-box"><h1><center>⚡ Developed by <b>Karan Jadhav</b></h1></center>', unsafe_allow_html=True)

    st.title("📘 About the Sales Prediction Model")

    st.write("""
    ## 👋 Overview  
    This application uses **Multiple Linear Regression (MLR)** to predict **Sales** based on:

    - 📺 TV  
    - 📻 Radio  
    - 📰 Newspaper  

    ---
    ## 🧠 What is Multiple Linear Regression?

    Formula:  
    **Sales = a + b1(TV) + b2(Radio) + b3(Newspaper)**  

    ---
    ## 🔍 How It Works
    1️⃣ Learns from data  
    2️⃣ Finds best-fit line  
    3️⃣ Predicts future sales  

    ---
    ## 🏆 Developer  
    **Karan Jadhav — Data Science Enthusiast**
    """)

    st.title("📈 Model Evaluation & Insights")

    df = pd.read_csv("Advertising Budget and Sales.csv", index_col=0)

    st.write("### 🔹 Correlation Heatmap")
    plt.figure(figsize=(8,5))
    sns.heatmap(df.corr(), annot=True)
    st.pyplot(plt)

    st.write("### 🔹 Sample True vs Predicted Plot")
    X = df.drop("Sales ($)", axis=1)
    y = df["Sales ($)"]
    y_pred = model.predict(X)

    plt.figure(figsize=(6,4))
    plt.scatter(y, y_pred)
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    st.pyplot(plt)

    if st.button("🏡 Go Home"):
        go_to("home")

    if st.button("🚀 Predict Sales"):
        go_to("appp")

# =========================================================
# END
# =========================================================
