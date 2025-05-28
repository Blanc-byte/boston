import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Load dataset (update this to your actual CSV path if needed)
df = pd.read_csv("boston.csv")  # Make sure your dataset is named properly

st.title("🏠 Boston Housing Dataset Visualization")
st.markdown("This app shows visual insights into housing prices using various charts.")

# Show Data
st.subheader("📄 Raw Data")
st.write(df.head())

# 1. Heatmap - Correlation between features
st.subheader("1️⃣ Heatmap of Feature Correlations")
fig1, ax1 = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax1)
st.pyplot(fig1)
st.caption("This heatmap shows correlations between variables. Note the strong negative correlation between LSTAT and MEDV.")

# 2. Histogram - Distribution of MEDV
st.subheader("2️⃣ Distribution of House Prices (MEDV)")
fig2, ax2 = plt.subplots()
sns.histplot(df["MEDV"], bins=30, kde=True, color='skyblue', ax=ax2)
st.pyplot(fig2)
st.caption("The histogram shows how home prices (MEDV) are distributed, with a peak around $20,000–$25,000.")

# 3. Boxplot - LSTAT vs MEDV
st.subheader("3️⃣ Boxplot of % Lower Status Population (LSTAT) vs House Prices")
fig3, ax3 = plt.subplots()
sns.boxplot(x=pd.qcut(df["LSTAT"], 4), y="MEDV", data=df, ax=ax3)
st.pyplot(fig3)
st.caption("This boxplot shows how home prices decrease as the percentage of lower-status residents increases.")

# 4. Scatter Plot - RM vs MEDV
st.subheader("4️⃣ Rooms vs House Price")
fig4 = px.scatter(df, x="RM", y="MEDV", trendline="ols", color="LSTAT")
st.plotly_chart(fig4)
st.caption("More rooms generally correlate with higher home prices. Color shows % of lower-status population.")

# 5. Pairplot (sampled)
st.subheader("5️⃣ Pairplot of Selected Features")
sample_df = df[["RM", "LSTAT", "PTRATIO", "MEDV"]].sample(200, random_state=42)
sns_plot = sns.pairplot(sample_df)
st.pyplot(sns_plot)
st.caption("Pairplot shows interactions among RM, LSTAT, PTRATIO, and MEDV. Helps reveal linear trends and clustering.")

# 6. Bar chart - Average MEDV by RAD index
st.subheader("6️⃣ Average House Price by Accessibility to Radial Highways (RAD)")
avg_rad = df.groupby("RAD")["MEDV"].mean().reset_index()
fig6 = px.bar(avg_rad, x="RAD", y="MEDV", color="MEDV", title="Average MEDV by RAD")
st.plotly_chart(fig6)
st.caption("This bar chart shows the average house price by highway accessibility index (RAD).")

# Footer
st.markdown("---")
st.markdown("🔍 **Dataset Source:** Boston Housing Dataset from UCI Repository or sklearn")

