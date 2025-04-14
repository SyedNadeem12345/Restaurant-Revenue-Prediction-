import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.set_page_config(page_title="🍽️ Restaurant Revenue Prediction", layout="centered")

st.title("🍽️ Restaurant Revenue Prediction App")
st.write("Upload your dataset and predict restaurant revenue based on orders and other numerical inputs.")

# Upload file
uploaded_file = st.file_uploader("Upload your revenue_prediction.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("File uploaded successfully!")

    # Data Preprocessing
    df = df.drop(columns=["Id", "Name", "Franchise", "Category", "City", "No_Of_Item"], errors="ignore")

    # Check if at least 2 columns remain
    if df.shape[1] < 2:
        st.error("Not enough numerical data left after dropping columns.")
    else:
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Model
        regressor = LinearRegression()
        regressor.fit(x_train, y_train)

        y_pred = regressor.predict(x_test)
        r2 = r2_score(y_test, y_pred)

        # Evaluation
        st.subheader("📊 Model Evaluation")
        st.write(f"**R² Score:** {r2:.2f}")

        # Visualization
        st.subheader("📉 Revenue vs Orders Plot")
        fig, ax = plt.subplots()
        ax.scatter(x_train[:, 0], y_train, color="blue", label="Training Data")
        ax.plot(x_train[:, 0], regressor.predict(x_train), color="green", label="Regression Line")
        ax.set_xlabel("Orders (or first feature)")
        ax.set_ylabel("Revenue")
        ax.legend()
        st.pyplot(fig)

        # Make new prediction
        st.subheader("🔮 Predict New Revenue")
        feature_count = x.shape[1]
        input_vals = []

        for i in range(feature_count):
            val = st.number_input(f"Enter value for feature {i+1} (same order as your dataset)", step=1.0)
            input_vals.append(val)

        if st.button("Predict Revenue"):
            prediction = regressor.predict([input_vals])
            st.success(f"Predicted Revenue: ₹{prediction[0]:,.2f}")
