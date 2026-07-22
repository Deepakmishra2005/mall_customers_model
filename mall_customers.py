import streamlit as st
import joblib
import numpy as np

def main():

    html_temp = """
    <h1 style="text-align:center;">Mall Customer Segmentation</h1>
    """

    st.markdown(html_temp, unsafe_allow_html=True)
    st.write("Enter the customer details to predict the customer segment.")

    model = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler(1).pkl")

    age = st.number_input("Age", min_value=18, max_value=100, value=30)

    annual_income = st.number_input(
        "Annual Income (k$)",
        min_value=0,
        max_value=200,
        value=50
    )

    spending_score = st.slider(
        "Spending Score (1-100)",
        min_value=1,
        max_value=100,
        value=50
    )

    if st.button("Predict Cluster"):

        data = np.array([[age, annual_income, spending_score]])

        scaled_data = scaler.transform(data)

        cluster = model.predict(scaled_data)

        st.success(f"Customer belongs to Cluster {cluster[0]}")

        if cluster[0] == 0:
            st.info("Segment: Standard Customers")
        elif cluster[0] == 1:
            st.info("Segment: High Income - High Spending")
        elif cluster[0] == 2:
            st.info("Segment: Low Income - High Spending")
        elif cluster[0] == 3:
            st.info("Segment: High Income - Low Spending")
        elif cluster[0] == 4:
            st.info("Segment: Low Income - Low Spending")

if __name__ == "__main__":
    main()