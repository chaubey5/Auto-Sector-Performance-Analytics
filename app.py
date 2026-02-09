import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

st.set_page_config(page_title="Auto Analytics Pro", layout="wide")

# --- 1. DATA LOADING & CLEANING ---
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
        return df
    except Exception as e:
        st.error(f"CSV Error: {e}")
        return pd.DataFrame()

df_all = load_data()

# --- 2. MAIN APP LOGIC ---
if df_all.empty:
    st.error("Data load nahi ho paya. Please check your CSV file.")
else:
    # --- SIDEBAR & CALCULATOR ---
    st.sidebar.title("🎮 Controls & Calculator")
    
    # Revenue Growth Calculator
    st.sidebar.subheader("📈 Revenue Growth Calculator")
    growth_pct = st.sidebar.slider("Expected Growth %", 0, 100, 10)
    
    st.sidebar.divider()
    
    # Filters
    all_states = sorted(df_all['Location'].unique())
    selected_states = st.sidebar.multiselect("1. Select States", options=all_states, default=all_states[:1])
    
    state_df = df_all[df_all['Location'].isin(selected_states)]
    all_comps = sorted(state_df['Name of the company'].unique())
    selected_comps = st.sidebar.multiselect("2. Select Companies", options=all_comps, default=all_comps[:8])

    df_final = state_df[state_df['Name of the company'].isin(selected_comps)]

    st.title("🚗 Auto Sector Interactive Dashboard")

    if len(df_final) < 2:
        st.warning("👈 Sidebar se kam se kam 2-3 companies select karein!")
    else:
        # CALCULATIONS
        current_rev = df_final['Revenue'].sum()
        projected_rev = current_rev * (1 + growth_pct/100)

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Companies", len(df_final))
        col2.metric("Current Total Revenue", f"₹{current_rev:.1f}M")
        col3.metric(f"Projected Revenue (+{growth_pct}%)", f"₹{projected_rev:.1f}M", delta=f"{growth_pct}%")
        col4.metric("Avg Margin", f"{df_final['Margin %'].mean():.1f}%")

        # Tabs
        t1, t2, t3, t4 = st.tabs(["📊 Charts", "🧬 Clustering", "📈 Predictions", "📝 Details"])

        with t1:
            st.subheader("Financial Visualization (Hover for details)")
            fig_bar = px.bar(df_final, x="Name of the company", y=["Revenue", "EBITDA"],
                             barmode="group", title="Revenue vs EBITDA",
                             hover_data=["Location", "Margin %"])
            st.plotly_chart(fig_bar, use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                fig_pie = px.pie(df_final, values='Revenue', names='Name of the company', title="Revenue Share", hole=0.4)
                st.plotly_chart(fig_pie, use_container_width=True)
            with col_b:
                fig_hist = px.histogram(df_final, x="Margin %", title="Margin Distribution", color_discrete_sequence=['orange'])
                st.plotly_chart(fig_hist, use_container_width=True)

        with t2:
            st.subheader("Market Segmentation (K-Means)")
            num_s = len(df_final)
            if num_s > 2:
                k_val = st.slider("Select Clusters", 2, min(num_s, 5), 2)
                X_scaled = StandardScaler().fit_transform(df_final[['Revenue', 'EBITDA']])
                km = KMeans(n_clusters=k_val, random_state=42, n_init=10).fit(X_scaled)
                df_final['Cluster'] = km.labels_.astype(str)

                fig_km = px.scatter(df_final, x="Revenue", y="EBITDA", color="Cluster", 
                                    hover_name="Name of the company", title="Company Grouping")
                st.plotly_chart(fig_km, use_container_width=True)
            else:
                st.info("Clustering ke liye zyada companies select karein.")

        with t3:
            st.subheader("Interactive Predictive Models")
            # Linear Regression
            lr = LinearRegression().fit(df_final[['Revenue']], df_final['EBITDA'])
            df_final['LR_Line'] = lr.predict(df_final[['Revenue']])
            
            fig_lr = px.scatter(df_final, x="Revenue", y="EBITDA", hover_name="Name of the company", title="Linear Regression")
            fig_lr.add_traces(go.Scatter(x=df_final['Revenue'], y=df_final['LR_Line'], name="Best Fit Line", line=dict(color='red')))
            st.plotly_chart(fig_lr, use_container_width=True)

        with t4:
            st.subheader("Company Descriptions")
            for _, row in df_final.iterrows():
                with st.expander(f"🏢 {row['Name of the company']}"):
                    st.write(f"**Location:** {row['Location']}")
                    st.write(f"**Products:** {row['Products']}")
                    st.info(f"EBITDA Margin: {row['Margin %']}%")