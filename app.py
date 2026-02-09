import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, BaggingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

st.set_page_config(page_title="Auto Sector ML Pro", layout="wide")

# --- 1. DATA LOADING & CLEANING (Notebook Logic) ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Autocomp - Companies list (CA-1) (1).csv")
        df.columns = df.columns.str.strip()
        
        def clean_val(x):
            if pd.isna(x): return None
            s = re.sub(r'[^\d.]', '', str(x).split('(')[0])
            if s.count('.') > 1:
                p = s.split('.')
                s = p[0] + "." + "".join(p[1:])
            try: return float(s)
            except: return None

        df['Revenue'] = df['Revenue'].apply(clean_val)
        df['EBITDA'] = df['EBITDA (IN MILLIONS)'].apply(clean_val)
        df = df.dropna(subset=['Revenue', 'EBITDA'])
        df['Margin %'] = (df['EBITDA'] / df['Revenue'] * 100).round(2)
        
        # Performance Class for Classification (Notebook Logic)
        df['Performance_Class'] = (df['Margin %'] > df['Margin %'].median()).astype(int)
        return df
    except Exception as e:
        st.error(f"CSV Error: {e}")
        return pd.DataFrame()

df_all = load_data()

if df_all.empty:
    st.error("Data load nahi ho paya. File check karein.")
else:
    # --- SIDEBAR ---
    st.sidebar.title("🎮 ML Controls")
    
    st.sidebar.subheader("📈 Growth Calculator")
    growth_pct = st.sidebar.slider("Expected Growth %", 0, 100, 10)
    
    st.sidebar.divider()
    
    all_states = sorted(df_all['Location'].unique())
    selected_states = st.sidebar.multiselect("Select States", options=all_states, default=all_states[:1])
    state_df = df_all[df_all['Location'].isin(selected_states)]
    
    selected_comps = st.sidebar.multiselect("Select Companies", options=sorted(state_df['Name of the company'].unique()), default=sorted(state_df['Name of the company'].unique())[:10])
    df_final = state_df[state_df['Name of the company'].isin(selected_comps)]

    st.title("🚗 Auto Sector Advanced ML Dashboard")

    if len(df_final) < 3:
        st.warning("👈 Kam se kam 3-5 companies select karein algorithms ke liye!")
    else:
        # CALCULATIONS
        current_rev = df_final['Revenue'].sum()
        projected_rev = current_rev * (1 + growth_pct/100)

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Companies", len(df_final))
        m2.metric("Total Revenue", f"₹{current_rev:.1f}M")
        m3.metric(f"Projected (+{growth_pct}%)", f"₹{projected_rev:.1f}M", delta=f"{growth_pct}%")
        m4.metric("Avg Margin", f"{df_final['Margin %'].mean():.1f}%")

        # TABS
        t1, t2, t3, t4 = st.tabs(["📊 Charts", "🧬 Clustering", "🤖 ML Models", "📝 Details"])

        with t1:
            st.subheader("Financial Performance")
            fig_bar = px.bar(df_final, x="Name of the company", y=["Revenue", "EBITDA"], barmode="group", hover_data=["Margin %"])
            st.plotly_chart(fig_bar, use_container_width=True)

        with t2:
            st.subheader("Advanced Clustering")
            c_algo = st.selectbox("Algorithm", ["K-Means", "Agglomerative"])
            k_val = st.slider("Clusters", 2, 5, 3)
            
            X_scaled = StandardScaler().fit_transform(df_final[['Revenue', 'EBITDA']])
            if c_algo == "K-Means":
                model = KMeans(n_clusters=k_val, random_state=42, n_init=10).fit(X_scaled)
            else:
                model = AgglomerativeClustering(n_clusters=k_val).fit(X_scaled)
            
            df_final['Cluster'] = model.labels_.astype(str)
            fig_km = px.scatter(df_final, x="Revenue", y="EBITDA", color="Cluster", hover_name="Name of the company", size="Margin %")
            st.plotly_chart(fig_km, use_container_width=True)

        with t3:
            st.subheader("ML Predictions (from Notebook)")
            model_task = st.radio("Task", ["Regression (Predict EBITDA)", "Classification (Margin Quality)"])
            
            X = df_final[['Revenue']]
            X_scaled = StandardScaler().fit_transform(X)
            
            if model_task == "Regression (Predict EBITDA)":
                m_type = st.selectbox("Model", ["Linear Regression", "Random Forest Regressor"])
                if m_type == "Linear Regression":
                    model = LinearRegression().fit(X, df_final['EBITDA'])
                else:
                    model = RandomForestRegressor(n_estimators=100).fit(X, df_final['EBITDA'])
                
                df_final['Pred'] = model.predict(X)
                fig_ml = px.scatter(df_final, x="Revenue", y="EBITDA", hover_name="Name of the company")
                fig_ml.add_traces(go.Scatter(x=df_final['Revenue'], y=df_final['Pred'], name="Model Prediction", line=dict(color='red')))
                st.plotly_chart(fig_ml, use_container_width=True)

            else:
                m_type = st.selectbox("Model", ["SVM Classifier", "MLP Neural Network", "Bagging Classifier", "Random Forest"])
                y = df_final['Performance_Class']
                
                if m_type == "SVM Classifier":
                    clf = SVC().fit(X_scaled, y)
                elif m_type == "MLP Neural Network":
                    clf = MLPClassifier(max_iter=1000).fit(X_scaled, y)
                elif m_type == "Random Forest":
                    clf = RandomForestClassifier().fit(X_scaled, y)
                else:
                    clf = BaggingClassifier().fit(X_scaled, y)
                
                df_final['Class_Pred'] = clf.predict(X_scaled).astype(str)
                fig_cl = px.scatter(df_final, x="Revenue", y="EBITDA", color="Class_Pred", 
                                    title="Classification (1: High Margin, 0: Low Margin)",
                                    hover_name="Name of the company")
                st.plotly_chart(fig_cl, use_container_width=True)

        with t4:
            st.subheader("Company Details")
            st.dataframe(df_final[['Name of the company', 'Location', 'Revenue', 'EBITDA', 'Margin %']])