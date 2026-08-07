import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------------------------------
# 0. Page Configuration & Styling
# ----------------------------------------------------
st.set_page_config(
    page_title="GrowthFlow Executive Analytics Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 GrowthFlow Executive Analytics Dashboard")
st.markdown("Cohort Retention, Customer Segmentation, Churn Timing & Revenue Trajectory")

# ----------------------------------------------------
# 1. Data Ingestion & Transformation Pipeline
# ----------------------------------------------------
@st.cache_data
def load_dashboard_data():
    try:
        df = pd.read_csv("model_df.csv")
    except FileNotFoundError:
        # Fallback if master_model_df or model_df exists in memory
        if 'master_model_df' in globals():
            df = master_model_df.copy()
        elif 'model_df' in globals():
            df = model_df.copy()
        else:
            st.error("Data file 'model_df.csv' not found! Please export the dataset to root.")
            st.stop()

    # Bin total_months_active into strict 24-Month Cohorts
    tenure_bins = [0, 3, 6, 12, 18, 24]
    tenure_labels = ['0–3 Mos', '4–6 Mos', '7–12 Mos', '13–18 Mos', '19–24 Mos']
    
    # Check for correct column name variation
    tenure_col = 'total_months_active' if 'total_months_active' in df.columns else 'months_since_subscription'
    
    df['tenure_cohort'] = pd.cut(
        df[tenure_col],
        bins=tenure_bins,
        labels=tenure_labels,
        include_lowest=True,
    )
    return df

raw_df = load_dashboard_data()

# ----------------------------------------------------
# 2. Interactive Sidebar Controls
# ----------------------------------------------------
st.sidebar.header("🔍 Global Dashboard Filters")

# Filter A: Plan Type
all_plans = list(raw_df['latest_plan_type'].dropna().unique()) if 'latest_plan_type' in raw_df.columns else []
selected_plans = st.sidebar.multiselect(
    "Filter by Plan Type:",
    options=all_plans,
    default=all_plans
)

# Filter B: Industry
all_industries = list(raw_df['industry'].dropna().unique()) if 'industry' in raw_df.columns else []
selected_industries = st.sidebar.multiselect(
    "Filter by Industry:",
    options=all_industries,
    default=all_industries
)

# Apply Filters
df_dash = raw_df.copy()

if selected_plans:
    df_dash = df_dash[df_dash['latest_plan_type'].isin(selected_plans)]

if selected_industries:
    df_dash = df_dash[df_dash['industry'].isin(selected_industries)]

# ----------------------------------------------------
# 3. High-Level Executive KPI Cards
# ----------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

total_cust = len(df_dash)
churn_rate = (df_dash['is_churned'].mean() * 100) if total_cust > 0 else 0.0
avg_revenue = df_dash['avg_monthly_revenue'].mean() if total_cust > 0 else 0.0
avg_decay = df_dash['session_duration_decay_ratio'].mean() if total_cust > 0 else 0.0

col1.metric("Total Active Customers", f"{total_cust:,}")
col2.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
col3.metric("Avg Monthly Revenue (ARPU)", f"${avg_revenue:,.2f}")
col4.metric("Avg Session Decay Ratio", f"{avg_decay:.2f}")

st.divider()

# ----------------------------------------------------
# 4. Interactive Visualizations Grid: Row 1
# ----------------------------------------------------
c1, c2 = st.columns(2)

with c1:
    st.subheader("🗓️ Cohort Retention Heatmap (%)")
    if not df_dash.empty:
        matrix = df_dash.groupby(['tenure_cohort', 'latest_plan_type'], observed=False)['is_churned'].mean().unstack() * 100
        retention_matrix = (100 - matrix).round(1)
        
        fig_heatmap = px.imshow(
            retention_matrix,
            text_auto='.1f',
            color_continuous_scale='Blues',
            labels=dict(x="Plan Type", y="Tenure Cohort", color="Retention (%)"),
            aspect="auto"
        )
        fig_heatmap.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("No data available for selected filter options.")

with c2:
    st.subheader("💰 Monthly Revenue Density by Cohort")
    if not df_dash.empty:
        rev_df = df_dash.groupby(['tenure_cohort', 'latest_plan_type'], observed=False)['avg_monthly_revenue'].mean().reset_index()
        fig_rev = px.bar(
            rev_df,
            x='tenure_cohort',
            y='avg_monthly_revenue',
            color='latest_plan_type',
            barmode='group',
            labels={'avg_monthly_revenue': 'Avg Revenue ($)', 'tenure_cohort': 'Tenure Cohort', 'latest_plan_type': 'Plan Type'},
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig_rev.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_rev, use_container_width=True)
    else:
        st.info("No data available for selected filter options.")

# ----------------------------------------------------
# 5. Interactive Visualizations Grid: Row 2
# ----------------------------------------------------
c3, c4 = st.columns(2)

with c3:
    st.subheader("🏢 Customer Distribution (Industry vs Plan)")
    if not df_dash.empty and 'industry' in df_dash.columns:
        fig_sunburst = px.sunburst(
            df_dash,
            path=['industry', 'latest_plan_type'],
            values='avg_monthly_revenue',
            color='latest_plan_type',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_sunburst.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_sunburst, use_container_width=True)
    else:
        st.info("No industry data available.")

with c4:
    st.subheader("📉 Activity Decay Trajectory")
    if not df_dash.empty:
        decay_df = df_dash.groupby('tenure_cohort', observed=False)['session_duration_decay_ratio'].mean().reset_index()
        fig_decay = px.line(
            decay_df,
            x='tenure_cohort',
            y='session_duration_decay_ratio',
            markers=True,
            labels={'session_duration_decay_ratio': 'Session Decay Ratio', 'tenure_cohort': 'Tenure Cohort'}
        )
        fig_decay.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Baseline Activity (1.0)")
        fig_decay.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_decay, use_container_width=True)
    else:
        st.info("No data available for selected filter options.")

# ----------------------------------------------------
# 6. Raw Data Download Option
# ----------------------------------------------------
with st.expander("📥 Inspect & Download Filtered Dashboard Dataset"):
    st.dataframe(df_dash, use_container_width=True)
    csv_data = df_dash.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered CSV",
        data=csv_data,
        file_name="filtered_growthflow_analytics.csv",
        mime="text/csv"
    )