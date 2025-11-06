# # dashboard.py (Streamlit app)
# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt

# st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
# st.title("🔍 Real-Time Fraud Detection Dashboard")

# # Load data
# log_path = "results/logged_predictions.csv"
# try:
#     df = pd.read_csv(log_path)
#     st.success(f"Loaded {len(df)} transactions from log.")
# except FileNotFoundError:
#     st.error("No prediction log found. Run predictions first.")
#     st.stop()

# # Summary stats
# fraud_count = df['is_fraud'].sum()
# non_fraud_count = len(df) - fraud_count
# st.metric("Total Transactions", len(df))
# st.metric("Detected as Fraud", fraud_count)
# st.metric("Detected as Non-Fraud", non_fraud_count)

# # Pie chart
# st.subheader("📊 Fraud vs Non-Fraud")
# fig1, ax1 = plt.subplots()
# ax1.pie([non_fraud_count, fraud_count], labels=["Non-Fraud", "Fraud"], autopct='%1.1f%%', startangle=90, colors=["lightgreen", "salmon"])
# ax1.axis("equal")
# st.pyplot(fig1)

# # Score histogram
# st.subheader("📈 Distribution of Fraud Scores")
# fig2, ax2 = plt.subplots()
# ax2.hist(df['score'], bins=30, color='skyblue', edgecolor='black')
# ax2.axvline(x=0.5, color='red', linestyle='--', label='Threshold = 0.5')
# ax2.set_xlabel("Fraud Score")
# ax2.set_ylabel("Frequency")
# ax2.legend()
# st.pyplot(fig2)

# # View recent predictions
# st.subheader("🧾 Recent Predictions")
# st.dataframe(df.tail(10))



#==================================

# dashboard.py (Enhanced Streamlit App with Timeline Graphs + Live Accuracy Section)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import classification_report
import requests

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("🔍 Real-Time Fraud Detection Dashboard")

# Load data
log_path = "results/logged_predictions.csv"
try:
    df = pd.read_csv(log_path)
    st.success(f"Loaded {len(df)} transactions from log.")
except FileNotFoundError:
    st.error("No prediction log found. Run predictions first.")
    st.stop()

# Add timestamp if missing
if "timestamp" not in df.columns:
    df["timestamp"] = pd.to_datetime("now")
else:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')

# Sidebar filters
st.sidebar.header("🔎 Filters")
is_fraud_filter = st.sidebar.selectbox("Show only:", ["All", "Fraud", "Non-Fraud"])
score_range = st.sidebar.slider("Fraud score range:", 0.0, 1.0, (0.0, 1.0))

# ✅ === Added Live Accuracy Metrics Section ===
st.sidebar.header("Model Performance")

def get_model_accuracy():
    try:
        response = requests.get("http://127.0.0.1:8000/live_accuracy")
        return response.json()
    except:
        return {"error": "Backend is not running or accuracy endpoint missing"}

if st.sidebar.button("🔄 Check Live Accuracy"):
    accuracy_data = get_model_accuracy()
    if "error" not in accuracy_data:
        st.sidebar.metric("Total Predictions", accuracy_data["total_predictions"])
        st.sidebar.metric("Fraud Rate", f"{accuracy_data['fraud_rate']}%")
        st.sidebar.metric("Avg Confidence", f"{accuracy_data['average_confidence']:.3f}")
    else:
        st.sidebar.error(accuracy_data["error"])
# ✅ === End Accuracy Block ===

# Apply filters
if is_fraud_filter == "Fraud":
    df = df[df["is_fraud"] == True]
elif is_fraud_filter == "Non-Fraud":
    df = df[df["is_fraud"] == False]

df = df[(df["score"] >= score_range[0]) & (df["score"] <= score_range[1])]

# Download CSV
st.sidebar.download_button("⬇️ Download Filtered Log", data=df.to_csv(index=False), file_name="filtered_predictions.csv")

# Metrics
st.metric("Total Transactions", len(df))
st.metric("Detected as Fraud", df['is_fraud'].sum())
st.metric("Detected as Non-Fraud", len(df) - df['is_fraud'].sum())

# ✅ Accuracy plot placeholder (optional future expansion)
st.subheader("📊 Model Accuracy Over Time")
st.write("Accuracy trend visualization coming soon... (based on backend logs)")

# Pie chart
st.subheader("📊 Fraud vs Non-Fraud")
fig1, ax1 = plt.subplots()
ax1.pie([len(df) - df['is_fraud'].sum(), df['is_fraud'].sum()], 
        labels=["Non-Fraud", "Fraud"], autopct='%1.1f%%', startangle=90, 
        colors=["lightgreen", "salmon"])
ax1.axis("equal")
st.pyplot(fig1)

# Score histogram
st.subheader("📈 Distribution of Fraud Scores")
fig2, ax2 = plt.subplots()
ax2.hist(df['score'], bins=30, color='skyblue', edgecolor='black')
ax2.axvline(x=0.5, color='red', linestyle='--', label='Threshold = 0.5')
ax2.set_xlabel("Fraud Score")
ax2.set_ylabel("Frequency")
ax2.legend()
st.pyplot(fig2)

# 📅 Timeline chart of fraud over time
st.subheader("📆 Fraud Over Time")
if "timestamp" in df.columns:
    df.set_index("timestamp", inplace=True)
    timeline = df.resample("H")["is_fraud"].sum()
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    timeline.plot(ax=ax3, kind='line', color='red', marker='o')
    ax3.set_title("Number of Fraudulent Transactions Over Time (Hourly)")
    ax3.set_ylabel("Fraud Count")
    ax3.set_xlabel("Timestamp")
    st.pyplot(fig3)
    df.reset_index(inplace=True)

# Export PNG suggestion
st.info("📌 To export dashboard: Ctrl+P → Save as PDF OR use GoFullPage Chrome extension")

# Classification metrics if ground truth exists
if "Class" in df.columns:
    report = classification_report(df['Class'], df['is_fraud'], output_dict=True)
    st.subheader("📏 Classification Report")
    st.dataframe(pd.DataFrame(report).transpose())

# Transaction simulator
st.subheader("🧪 Simulate a Transaction")
with st.form("simulate_form"):
    input_data = {}
    for col in df.columns:
        if col not in ["is_fraud", "score", "timestamp", "Class"]:
            input_data[col] = st.number_input(col, value=float(df[col].mean()))
    submitted = st.form_submit_button("Predict")
    if submitted:
        response = requests.post("http://127.0.0.1:8000/predict", json=input_data)
        if response.status_code == 200:
            st.success(f"Prediction: {response.json()}")
        else:
            st.error("Prediction failed.")

# Recent predictions
st.subheader("🧾 Recent Predictions")
def highlight_fraud(row):
    return ['background-color: salmon' if row["is_fraud"] else '' for _ in row]

st.dataframe(df.tail(20).style.apply(highlight_fraud, axis=1))
