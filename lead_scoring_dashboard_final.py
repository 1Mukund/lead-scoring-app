import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Dynamic Lead Scoring System", layout="wide")
st.title("📊 Dynamic, Math-Based Lead Scoring System")

uploaded_file = st.file_uploader("Upload your Leads Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('-', '_')
    df.fillna(0, inplace=True)

    # Feature Columns
    feature_cols = [
        'CumulativeTime', 'Number_of_Page_Visited', 'Unqiue_Visits',
        'WhatsappInbound', 'WhatsappOutbound',
        'daysSinceLastWebActivity', 'daysSinceLastInbound', 'daysSinceLastOutbound',
        'HighValuePageViews', 'DownloadedFilesCount'
    ]

    # Normalize Features
    scaler = MinMaxScaler()
    feature_data = scaler.fit_transform(df[feature_cols])
    normalized_features = pd.DataFrame(feature_data, columns=feature_cols)

    # Calculate Dynamic Feature Weights based on Correlation
    corrs = pd.DataFrame()
    corrs['feature'] = feature_cols
    corrs['correlation_to_inbound'] = [df[f].corr(df['WhatsappInbound']) for f in feature_cols]
    corrs['abs_corr'] = corrs['correlation_to_inbound'].abs()
    corrs['weight'] = corrs['abs_corr'] / corrs['abs_corr'].sum()

    feature_weights = corrs.set_index('feature')['weight'].to_dict()

    # Dynamic Scoring
    df['lead_score'] = normalized_features.dot(pd.Series(feature_weights))

    # Dynamic Bucketing based on Percentile
    df['score_percentile'] = df['lead_score'].rank(pct=True) * 100
    df['score_percentile'] = df['score_percentile'].clip(lower=0, upper=100)

    bucket_labels = ["Dormant", "Cold", "Curious", "Warm", "Engaged", "Hot"]
    df['lead_bucket'] = pd.cut(
        df['score_percentile'],
        bins=[-0.01, 30, 50, 75, 90, 100],
        labels=bucket_labels,
        include_lowest=True
    )

    # Show Scoring Formula and Weights
    with st.expander("📋 Scoring Formula and Feature Weights"):
        st.dataframe(corrs[['feature', 'weight']])
        formula = "Lead Score = " + " + ".join([f"{w:.2f}×{f}" for f, w in feature_weights.items()])
        st.write(formula)

    # Contribution Table
    contribution_matrix = normalized_features.mul(pd.Series(feature_weights), axis=1)
    st.subheader("🔍 Feature Contributions Per Lead")
    st.dataframe(contribution_matrix)

    # Visualizations
    st.subheader("📈 Lead Score Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['lead_score'], bins=20, ax=ax1)
    st.pyplot(fig1)

    st.subheader("📊 Lead Buckets Distribution")
    fig2, ax2 = plt.subplots()
    sns.countplot(x='lead_bucket', data=df, order=bucket_labels, ax=ax2)
    st.pyplot(fig2)

    # Download Scored Leads
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    buffer.seek(0)
    st.download_button("Download Leads with Scores", buffer, "scored_leads.xlsx")

else:
    st.info("Please upload an Excel file to begin scoring.")
