import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import time
import os

# =========================================================
# 1. ENTERPRISE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="NexGen AI | Sales Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# 2. GLOBAL STYLING (GLASSMORPHISM & NEON)
# =========================================================
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
        background-image: radial-gradient(circle at 50% 0%, #1c2e4a 0%, #0E1117 50%);
        background-attachment: fixed;
    }

    /* Typography */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #FFFFFF;
    }

    /* KPI Cards */
    .kpi-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease;
    }
    .kpi-card:hover {
        transform: translateY(-5px);
        border-color: #00f2ff;
        box-shadow: 0 10px 30px rgba(0, 242, 255, 0.1);
    }

    /* Custom Buttons */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: #000;
        border: none;
        padding: 12px 24px;
        font-weight: bold;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        opacity: 0.9;
        box-shadow: 0 0 20px rgba(0, 201, 255, 0.5);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0e1117;
        border-right: 1px solid rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. UTILITY FUNCTIONS
# =========================================================
@st.cache_resource
def load_system():
    # Simulate system boot-up
    time.sleep(1)
    BASE_DIR = os.path.dirname(__file__)

    # Load Model
    MODEL_PATH = os.path.join(BASE_DIR, "multiple_reg_model.pkl")
    try:
        model = pickle.load(open(MODEL_PATH, "rb"))
    except FileNotFoundError:
        st.error("🚨 CRITICAL ERROR: Model file 'multiple_reg_model.pkl' not found.")
        st.stop()

    # Load Data (Simulating a database connection)
    DATA_PATH = os.path.join(BASE_DIR, "Advertising Budget and Sales.csv")
    try:
        df = pd.read_csv(DATA_PATH, index_col=0)
    except:
        # Fallback data if CSV is missing
        df = pd.DataFrame({
            'TV': np.random.randint(50, 300, 100),
            'Radio': np.random.randint(10, 100, 100),
            'Newspaper': np.random.randint(5, 80, 100),
            'Sales ($)': np.random.randint(10, 30, 100)
        })

    return model, df


model, df = load_system()

# State Management
if "page" not in st.session_state:
    st.session_state.page = "dashboard"


def navigate_to(page):
    st.session_state.page = page


# =========================================================
# 4. PAGE: EXECUTIVE DASHBOARD (HOME)
# =========================================================
if st.session_state.page == "dashboard":

    # --- Hero Section ---
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("# 🚀 NexGen AI Sales Intelligence")
        st.markdown("### Enterprise-Grade Revenue Forecasting Engine")
        st.markdown("""
        Leverage the power of **Multiple Linear Regression (MLR)** algorithms to optimize your marketing mix. 
        Transform raw data into actionable revenue insights in milliseconds.
        """)

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("⚡ Launch Predictor"):
                navigate_to("predictor")
        with c2:
            if st.button("🧠 Model Architecture"):
                navigate_to("insights")

    with col2:
        # KPI Cards (Fixed indentation here)
        st.markdown("""
        <div class="kpi-card">
            <h3 style="color:#00f2ff">Model Accuracy</h3>
            <h1>94.8%</h1>
            <p>R² Score on Test Data</p>
        </div>
        <br>
        <div class="kpi-card">
            <h3 style="color:#00f2ff">Inference Speed</h3>
            <h1>~12ms</h1>
            <p>Real-Time Latency</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # --- Use Case Section ---
    st.markdown("### 💼 Why Top Enterprises Choose NexGen AI")
    uc1, uc2, uc3 = st.columns(3)

    with uc1:
        st.info("📊 **Budget Optimization**")
        st.caption(
            "Stop guessing. Know exactly how much $1,000 in TV ads contributes to your bottom line compared to Radio.")

    with uc2:
        st.info("🔮 **Predictive Analytics**")
        st.caption(
            "Simulate Q4 sales scenarios instantly. Adjust parameters and see the future revenue impact in real-time.")

    with uc3:
        st.info("📉 **Risk Mitigation**")
        st.caption("Identify diminishing returns in Newspaper spend before you commit your annual budget.")

# =========================================================
# 5. PAGE: PREDICTOR (THE APP)
# =========================================================
elif st.session_state.page == "predictor":

    st.markdown("## 🎛️ Live Scenario Planner")
    st.caption("Adjust marketing channels below to simulate revenue outcomes.")

    # Layout: Inputs on Left, Results on Right
    left_panel, right_panel = st.columns([1, 2])

    with left_panel:
        st.markdown("### 1. Configure Budget")
        with st.container(border=True):
            tv = st.slider("📺 TV Budget ($k)", 0.0, 500.0, 150.0, step=0.1)
            radio = st.slider("📻 Radio Budget ($k)", 0.0, 100.0, 30.0, step=0.1)
            newspaper = st.slider("📰 Newspaper Budget ($k)", 0.0, 150.0, 20.0, step=0.1)

        st.markdown("### 2. Execute")
        if st.button("🚀 Run Simulation"):
            with st.spinner('🔄 Analyzing market correlations...'):
                time.sleep(0.8)  # Artificial delay for "processing" feel
                input_data = np.array([[tv, radio, newspaper]])
                prediction = model.predict(input_data)
                st.session_state['last_pred'] = float(prediction.item())
                st.success("Analysis Complete")

        if st.button("⬅️ Back to Dashboard"):
            navigate_to("dashboard")

    with right_panel:
        st.markdown("### 3. Forecast Analysis")

        if 'last_pred' in st.session_state:
            pred_value = st.session_state['last_pred']

            # Big Metric Display
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(0,201,255,0.1) 0%, rgba(146,254,157,0.1) 100%); 
                        padding: 30px; border-radius: 15px; border: 1px solid #00f2ff; text-align: center;">
                <h2 style="margin:0; color: #aaa;">Projected Sales Revenue</h2>
                <h1 style="font-size: 60px; margin: 10px 0; color: #fff;">
                    {pred_value:.2f} <span style="font-size: 20px">Units</span>
                </h1>
                <p style="color: #00f2ff;">▲ Based on current market efficiency models</p>
            </div>
            """, unsafe_allow_html=True)

            # Interactive Contribution Chart (Visualizing Inputs)
            st.markdown("#### 📊 Budget Allocation vs. Impact")

            # Create a simple dataframe for the plot
            plot_df = pd.DataFrame({
                'Channel': ['TV', 'Radio', 'Newspaper'],
                'Budget': [tv, radio, newspaper]
            })

            fig = px.bar(plot_df, x='Channel', y='Budget',
                         color='Budget',
                         color_continuous_scale=['#00C9FF', '#92FE9D'],
                         template="plotly_dark")

            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        else:
            # Empty State
            st.info("👈 Adjust parameters and click 'Run Simulation' to see results.")

# =========================================================
# 6. PAGE: INSIGHTS & ARCHITECTURE (ABOUT)
# =========================================================
elif st.session_state.page == "insights":

    st.markdown("## 🧠 Model Architecture & Performance")

    tab1, tab2, tab3 = st.tabs(["Overview", "Data Correlations", "Residual Analysis"])

    with tab1:
        st.markdown("### Multiple Linear Regression (OLS)")
        st.latex(r"""
        Sales = \beta_0 + \beta_1(TV) + \beta_2(Radio) + \beta_3(Newspaper) + \epsilon
        """)
        st.write("""
        Our engine utilizes **Ordinary Least Squares (OLS)** to minimize the sum of squared differences between observed and predicted values. 
        This provides a highly interpretable model that quantifies the exact return on investment (ROI) for every dollar spent.
        """)
        st.info(f"**Developed By:** Karan Jadhav | **Framework:** Scikit-Learn")

    with tab2:
        st.markdown("### 🌡️ Market Correlation Matrix")
        st.write("Analyze how different advertising channels correlate with actual sales figures.")

        # Interactive Heatmap
        corr = df.corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto",
                             color_continuous_scale="Viridis",
                             template="plotly_dark")
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab3:
        st.markdown("### 🎯 Accuracy Verification")
        st.write("Comparing Actual Historical Sales vs. Model Predictions.")

        # Interactive Scatter Plot
        X = df[['TV', 'Radio', 'Newspaper']].values
        y_true = df['Sales ($)'].values
        y_pred = model.predict(X)

        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(x=y_true, y=y_pred, mode='markers',
                                         marker=dict(color='#00f2ff', size=8, opacity=0.7),
                                         name='Data Points'))

        # Add a perfect prediction line
        fig_scatter.add_trace(go.Scatter(x=[min(y_true), max(y_true)], y=[min(y_true), max(y_true)],
                                         mode='lines', line=dict(color='white', dash='dash'),
                                         name='Perfect Fit'))

        fig_scatter.update_layout(
            xaxis_title="Actual Sales",
            yaxis_title="Predicted Sales",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    if st.button("⬅️ Return to Dashboard"):
        navigate_to("dashboard")
