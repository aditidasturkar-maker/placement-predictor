import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Placement Dashboard", layout="wide")

# -----------------------------
# PASTEL CSS 🎨
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #fceff9, #e0f7fa);
}
h1, h2, h3 {
    color: #6a5acd;
}
.stButton>button {
    background-color: #ffb6c1;
    color: black;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("placement_data.csv")

df.drop("College_ID", axis=1, inplace=True)

le = LabelEncoder()
df["Internship_Experience"] = le.fit_transform(df["Internship_Experience"])
df["Placement"] = le.fit_transform(df["Placement"])

X = df.drop("Placement", axis=1)
y = df["Placement"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier()
model.fit(X_scaled, y)

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Dashboard", "🔮 Prediction"])

# -----------------------------
# HOME PAGE
# -----------------------------
if page == "🏠 Home":
    st.title("🎓 Student Placement Prediction System")
    st.write("This system predicts whether a student will be placed or not using Machine Learning.")
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135755.png", width=200)

# -----------------------------
# DASHBOARD PAGE
# -----------------------------
elif page == "📊 Dashboard":
    st.title("📊 Data Analysis Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("CGPA Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["CGPA"], bins=20, ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Communication Skills")
        fig, ax = plt.subplots()
        sns.histplot(df["Communication_Skills"], bins=20, ax=ax)
        st.pyplot(fig)

    st.subheader("Internship vs Placement")
    fig, ax = plt.subplots()
    sns.countplot(x="Internship_Experience", hue="Placement", data=df, ax=ax)
    st.pyplot(fig)

# -----------------------------
# PREDICTION PAGE
# -----------------------------
elif page == "🔮 Prediction":

    st.title("🔮 Placement Prediction")

    col1, col2 = st.columns(2)

    user_data = []
    columns = list(X.columns)

    for i, col in enumerate(columns):

        if i % 2 == 0:
            with col1:
                if col == "Internship_Experience":
                    val = st.selectbox("Internship Experience", ["No", "Yes"])
                    user_data.append(1 if val == "Yes" else 0)
                else:
                    val = st.number_input(col)
                    user_data.append(val)
        else:
            with col2:
                if col == "Internship_Experience":
                    val = st.selectbox("Internship Experience", ["No", "Yes"])
                    user_data.append(1 if val == "Yes" else 0)
                else:
                    val = st.number_input(col)
                    user_data.append(val)

    st.markdown("---")

    if st.button("🎯 Predict"):

        user_array = np.array([user_data])
        user_scaled = scaler.transform(user_array)

        pred = model.predict(user_scaled)
        prob = model.predict_proba(user_scaled)

        if pred[0] == 1:
            st.success("🎉 Student is Likely to be PLACED")
        else:
            st.error("❌ Student is NOT Likely to be Placed")

        st.info(f"📊 Placement Probability: {round(prob[0][1]*100, 2)}%")