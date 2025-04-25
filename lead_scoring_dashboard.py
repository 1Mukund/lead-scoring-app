
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Dynamic Lead Scoring", layout="wide")

st.title("📊 Dynamic Lead Scoring & Messaging System")
st.write("Upload your lead Excel file to begin dynamic scoring based on behavioral analysis.")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Fill missing values
    df.fillna(0, inplace=True)

    # Feature columns
    feature_cols = [
        'CumulativeTime', 'Number_of_Page_Visited', 'Unqiue_Visits',
        'WhatsappInbound', 'WhatsappOutbound',
        'daysSinceLastWebActivity', 'daysSinceLastInbound', 'daysSinceLastOutbound',
        'HighValuePageViews', 'DownloadedFilesCount'
    ]

    # Normalize features
    scaler = MinMaxScaler()
    feature_data = scaler.fit_transform(df[feature_cols])
    feature_df = pd.DataFrame(feature_data, columns=feature_cols)

    # Correlation-based weights (auto-learned from dataset)
    corrs = pd.DataFrame()
    corrs['feature'] = feature_cols
    corrs['correlation_to_inbound'] = [df[f].corr(df['WhatsappInbound']) for f in feature_cols]
    corrs['abs_corr'] = corrs['correlation_to_inbound'].abs()
    corrs['weight'] = corrs['abs_corr'] / corrs['abs_corr'].sum()

    # Display feature importance
    st.subheader("📈 Feature Importance (Auto-Learned)")
    st.dataframe(corrs[['feature', 'weight']].sort_values(by='weight', ascending=False))

    # Calculate score
    weights = corrs.set_index('feature')['weight'].to_dict()
    df['lead_score'] = feature_df.dot(pd.Series(weights))

    # Bucket assignment based on percentile
    df['score_percentile'] = df['lead_score'].rank(pct=True) * 100

    def bucketize(p):
        if p >= 90:
            return 'Hot'
        elif p >= 75:
            return 'Engaged'
        elif p >= 50:
            return 'Warm'
        elif p >= 30:
            return 'Curious'
        elif p > 0:
            return 'Cold'
        else:
            return 'Dormant'

    df['lead_bucket'] = df['score_percentile'].apply(bucketize)

    # Suggested messages
    def suggest_message(bucket):
        return {
            'Hot': "You're close to finalizing — let's schedule your site visit!",
            'Engaged': "Would you like to explore our EMI or pricing walkthrough?",
            'Warm': "Here’s a brochure & walkthrough to help your decision.",
            'Curious': "Explore what makes our projects different in 2-min video.",
            'Cold': "We’re here if you have any questions or want to revisit.",
            'Dormant': "Limited time offer for you if you’re still interested."
        }.get(bucket, "")

    df['recommended_message'] = df['lead_bucket'].apply(suggest_message)

    # Display results
    st.subheader("📊 Scored Leads with Buckets and Messages")
    st.dataframe(df[['LeadId', 'lead_score', 'lead_bucket', 'recommended_message']])

    # Charts
    st.subheader("📊 Lead Score Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['lead_score'], bins=20, ax=ax1)
    st.pyplot(fig1)

    st.subheader("📊 Bucket Distribution")
    fig2, ax2 = plt.subplots()
    sns.countplot(x='lead_bucket', data=df, order=['Hot', 'Engaged', 'Warm', 'Curious', 'Cold', 'Dormant'], ax=ax2)
    st.pyplot(fig2)

    # Download updated Excel
    st.subheader("📥 Download Updated Leads File")
    st.download_button("Download Excel", df.to_excel(index=False), file_name="scored_leads.xlsx")
else:
    st.info("Please upload an Excel file with the required lead data fields.")
