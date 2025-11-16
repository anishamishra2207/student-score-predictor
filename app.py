# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from io import BytesIO
from PIL import Image
import base64
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# --- Files ---
MODEL_FILE = "house_model.pkl"
SCALER_FILE = "scaler.pkl"
DATA_FILE = "housing.csv"
HISTORY_FILE = "history.csv"

# --- Ensure artifacts exist ---
if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
    st.error("Model or scaler not found. Run `python model.py` first to train and save the model.")
    st.stop()

# --- Load model & scaler ---
model = pickle.load(open(MODEL_FILE, "rb"))
scaler = pickle.load(open(SCALER_FILE, "rb"))

# --- Page config ---
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

# --- Dark theme CSS (simple) ---
dark_css = """
<style>
body { background-color: #0e1117; color: #e6eef8; }
.stButton>button { background-color: #1f6feb; color: white; }
.stDownloadButton>button { background-color: #1f6feb; color: white; }
.stSlider>div>div>input { color: #e6eef8; }
[data-testid="stSidebar"] { background-color: #0b1220; color: #e6eef8; }
</style>
"""
st.markdown(dark_css, unsafe_allow_html=True)

# --- Helper functions ---
def predict_single(area, bedrooms, age):
    X = np.array([[area, bedrooms, age]])
    Xs = scaler.transform(X)
    pred = model.predict(Xs)[0]
    return float(pred)

def predict_batch(df_input):
    X = df_input[["Area", "Bedrooms", "Age"]].values
    Xs = scaler.transform(X)
    preds = model.predict(Xs)
    return preds

def to_excel_bytes(df):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="predictions")
        writer.save()
    return buffer.getvalue()

def get_download_link_df(df, filename="predictions.csv"):
    return st.download_button(label="Download CSV", data=df.to_csv(index=False).encode('utf-8'),
                              file_name=filename, mime="text/csv")

# --- Sidebar navigation ---
st.sidebar.title("🏠 House Predictor")
page = st.sidebar.radio("Navigate", ["Home", "Bulk Predict", "Dashboard", "Upload Image", "About"])

# --- Load dataset (if present) ---
if os.path.exists(DATA_FILE):
    df_all = pd.read_csv(DATA_FILE)
else:
    df_all = pd.DataFrame(columns=["Area", "Bedrooms", "Age", "Price"])

# --- Ensure history file exists ---
if not os.path.exists(HISTORY_FILE):
    pd.DataFrame(columns=["Area", "Bedrooms", "Age", "PredictedPrice"]).to_csv(HISTORY_FILE, index=False)

# -----------------------
# Page: Home
# -----------------------
if page == "Home":
    st.title("🏠 House Price Prediction — Single")
    st.markdown("Use the sliders to provide house details. The model will predict price in ₹ (same units as training).")

    col1, col2 = st.columns([2, 1])

    with col1:
        area = st.number_input("Area (sq ft)", min_value=500, max_value=5000, value=1500, step=10)
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3, step=1)
        age = st.number_input("Age (years)", min_value=0, max_value=50, value=5, step=1)

        pred = predict_single(area, bedrooms, age)
        st.success(f"🏡 Estimated Price: ₹ {pred:,.2f}")

        if st.button("Save to History 💾"):
            # append to history
            history = pd.read_csv(HISTORY_FILE)
            new_row = {"Area": area, "Bedrooms": bedrooms, "Age": age, "PredictedPrice": pred}
            history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
            history.to_csv(HISTORY_FILE, index=False)
            st.write("Saved to prediction history.")

    with col2:
        st.subheader("Model Metrics (on full dataset)")
        if not df_all.empty:
            X_all = df_all[["Area", "Bedrooms", "Age"]].values
            y_all = df_all["Price"].values
            preds_all = predict_batch(df_all)
            r2 = r2_score(y_all, preds_all)
            mse = mean_squared_error(y_all, preds_all)
            st.metric("R² Score", f"{r2:.3f}")
            st.metric("MSE", f"{mse:.2f}")
        else:
            st.info("No training dataset found (housing.csv). Metrics unavailable.")

    st.markdown("---")
    st.subheader("Prediction History (recent)")
    hist = pd.read_csv(HISTORY_FILE)
    if hist.empty:
        st.info("No predictions yet.")
    else:
        st.dataframe(hist.sort_index(ascending=False).head(10))
        get_download_link_df(hist, filename="prediction_history.csv")
        if st.button("🔄 Reset History"):
            pd.DataFrame(columns=["Area", "Bedrooms", "Age", "PredictedPrice"]).to_csv(HISTORY_FILE, index=False)
            st.success("History reset!")

# -----------------------
# Page: Bulk Predict
# -----------------------
elif page == "Bulk Predict":
    st.title("📥 Bulk Prediction")
    st.markdown("Upload a CSV containing columns: **Area, Bedrooms, Age**. The app will predict price for each row.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    sample_button = st.button("Show Sample CSV")

    if sample_button:
        sample = pd.DataFrame({
            "Area": [1200, 1500, 900],
            "Bedrooms": [2, 3, 2],
            "Age": [10, 5, 20]
        })
        st.dataframe(sample)

        st.download_button("Download sample CSV", data=sample.to_csv(index=False).encode('utf-8'),
                           file_name="sample_bulk.csv", mime="text/csv")

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        needed = {"Area", "Bedrooms", "Age"}
        if not needed.issubset(set(input_df.columns)):
            st.error(f"CSV must contain columns: {needed}")
            st.stop()

        st.write("Preview uploaded data:")
        st.dataframe(input_df.head())

        if st.button("Run Bulk Predict"):
            preds = predict_batch(input_df)
            input_df["PredictedPrice"] = preds
            st.success("Prediction complete!")
            st.dataframe(input_df)

            # Save to history
            history = pd.read_csv(HISTORY_FILE)
            hist_new = input_df[["Area", "Bedrooms", "Age", "PredictedPrice"]]
            history = pd.concat([history, hist_new], ignore_index=True)
            history.to_csv(HISTORY_FILE, index=False)

            # Download options
            get_download_link_df(input_df, filename="bulk_predictions.csv")
            excel_bytes = to_excel_bytes(input_df)
            st.download_button("Download Excel", data=excel_bytes, file_name="bulk_predictions.xlsx")

# -----------------------
# Page: Dashboard
# -----------------------
elif page == "Dashboard":
    st.title("📊 Dashboard")
    st.markdown("Visuals for the original dataset and prediction history.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Training Data Overview")
        if not df_all.empty:
            st.dataframe(df_all.describe())
            st.markdown("**Scatter: Area vs Price**")
            fig, ax = plt.subplots()
            ax.scatter(df_all["Area"], df_all["Price"], alpha=0.6)
            ax.set_xlabel("Area (sq ft)")
            ax.set_ylabel("Price")
            st.pyplot(fig)
        else:
            st.info("No training dataset (housing.csv) found.")

    with col2:
        st.subheader("Prediction History Charts")
        hist = pd.read_csv(HISTORY_FILE)
        if not hist.empty:
            st.markdown("**Predicted Price Distribution**")
            fig2, ax2 = plt.subplots()
            ax2.hist(hist["PredictedPrice"], bins=15)
            ax2.set_xlabel("Predicted Price")
            ax2.set_ylabel("Count")
            st.pyplot(fig2)

            st.markdown("**Area vs Predicted Price (History)**")
            fig3, ax3 = plt.subplots()
            ax3.scatter(hist["Area"], hist["PredictedPrice"], alpha=0.7)
            ax3.set_xlabel("Area")
            ax3.set_ylabel("Predicted Price")
            st.pyplot(fig3)

            st.markdown("Download history:")
            get_download_link_df(hist, filename="prediction_history.csv")
        else:
            st.info("No predictions made yet. Use Home or Bulk Predict to add entries.")

# -----------------------
# Page: Upload Image
# -----------------------
elif page == "Upload Image":
    st.title("🖼️ Upload House Image (Display only)")
    st.markdown("Upload a house image to display here. **Note:** This app does not use images for prediction — the model predicts from numerical features only.")

    uploaded_image = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        img = Image.open(uploaded_image)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        st.success("Image loaded. You can still use sliders / CSV for predictions.")

# -----------------------
# Page: About
# -----------------------
elif page == "About":
    st.title("ℹ️ About & Instructions")
    st.markdown("""
    **Project:** House Price Prediction (Linear Regression)  
    **Model trained with:** `model.py` on `housing.csv` (Area, Bedrooms, Age -> Price)  
    **How to use:**  
    1. If not yet trained, run `python model.py` to create `house_model.pkl` and `scaler.pkl`.  
    2. Run the app: `streamlit run app.py`.  
    3. Use *Home* for single predictions, *Bulk Predict* to upload many rows, *Dashboard* to view charts.

    **Notes & Tips**
    - Bulk CSV must contain columns: `Area`, `Bedrooms`, `Age`.
    - Prediction history is saved to `history.csv`.
    - You can reset history from the Home page.
    """)

    if not df_all.empty:
        st.markdown("### Dataset preview (first rows)")
        st.dataframe(df_all.head())

    st.markdown("### Requirements")
    st.code("pip install streamlit scikit-learn pandas numpy pillow xlsxwriter")

