import streamlit as st
import joblib
import numpy as np


model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")


def main():

    st.set_page_config(
        page_title="Mall Customer Segmentation"
    )

    st.title("Mall Customer Segmentation")
    st.write("Predict the customer segment using the trained K-Means model.")

    gender = st.selectbox(
        "Select Gender",
        ("Male", "Female")
    )

    gender = 1 if gender == "Male" else 0

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=25
    )

    income = st.number_input(
        "Annual Income (k$)",
        min_value=0,
        max_value=150,
        value=40
    )

    spending = st.slider(
        "Spending Score (1-100)",
        1,
        100,
        60
    )

    if st.button("Predict Cluster"):

        data = np.array([[gender, age, income, spending]])

        scaled_data = scaler.transform(data)

        prediction = model.predict(scaled_data)

        st.success(f"Customer belongs to Cluster {prediction[0]}")

        if prediction[0] == 0:
            st.info("Segment 0")
        elif prediction[0] == 1:
            st.info("Segment 1")
        elif prediction[0] == 2:
            st.info("Segment 2")
        elif prediction[0] == 3:
            st.info("Segment 3")
        elif prediction[0] == 4:
            st.info("Segment 4")


if __name__ == "__main__":
    main()