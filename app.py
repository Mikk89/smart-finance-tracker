import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import requests
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Machine Learning Core
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Smart Finance Tracker & Predictor", layout="wide")

DB_FILE = "finance_tracker.db"

# ==========================================
# ENTERPRISE SQLITE DATABASE ENGINE
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            category TEXT NOT NULL,
            amount REAL NOT NULL,
            notes TEXT
        )
    ''')
    conn.commit()
    
    # Auto-seed database with 90 days of structural data if empty
    cursor.execute("SELECT COUNT(*) FROM transactions")
    if cursor.fetchone()[0] < 10:
        start_date = datetime.now() - timedelta(days=90)
        base_data = []
        for i in range(90):
            current_date = start_date + timedelta(days=i)
            is_weekend = current_date.weekday() >= 5
            date_str = current_date.strftime("%Y-%m-%d")
            
            if current_date.day == 1:
                base_data.append((date_str, "Rent", 500.00, "Monthly Fixed Rent"))
                base_data.append((date_str, "Utilities", 120.00, "Electric & Water Bill"))
                
            food_cost = np.random.uniform(20, 50) + (25 if is_weekend else 0)
            base_data.append((date_str, "Food", round(food_cost, 2), "Daily meals"))
            
            if np.random.rand() > 0.4:
                trans_cost = np.random.uniform(10, 30)
                base_data.append((date_str, "Transportation", round(trans_cost, 2), "Fuel/Commute"))
                
            if is_weekend and np.random.rand() > 0.3:
                ent_cost = np.random.uniform(40, 120)
                base_data.append((date_str, "Entertainment", round(ent_cost, 2), "Weekend leisure"))
                
        cursor.executemany("INSERT INTO transactions (date, category, amount, notes) VALUES (?, ?, ?, ?)", base_data)
        conn.commit()
    conn.close()

init_db()

def fetch_ledger():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM transactions ORDER BY date ASC", conn)
    conn.close()
    df["Date"] = pd.to_datetime(df["date"])
    df["Amount"] = df["amount"]
    df["Category"] = df["category"]
    df["Notes"] = df["notes"]
    return df

df = fetch_ledger()

# ==========================================
# LIVE CURRENCY API GATEWAY
# ==========================================
@st.cache_data(ttl=3600)
def fetch_live_exchange_rate():
    try:
        response = requests.get("https://open.er-api.com/v6/latest/USD", timeout=5)
        if response.status_code == 200:
            return response.json()["rates"].get("THB", 36.50)
    except:
        pass
    return 36.50

thb_rate = fetch_live_exchange_rate()

# ==========================================
# USER INTERFACE
# ==========================================
st.title("💰 Smart Finance Tracker & Predictor")
st.caption("Computer Engineering System | SQLite Engine + Live Exchange Rate Streaming + 4-Way ML Showdown")

tab1, tab2, tab3 = st.tabs(["📥 Ledger & Database Engine", "📊 Analytics & Insights", "🔮 Model Showdown"])

# ------------------------------------------
# TAB 1: LEDGER & CRUD
# ------------------------------------------
with tab1:
    st.subheader("Data Ingestion & SQL Database Management")
    
    # Ingestion Form
    with st.form("sql_ingestion_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            in_date = st.date_input("Date:", datetime.now())
            in_cat = st.selectbox("Category:", ["Food", "Rent", "Transportation", "Entertainment", "Utilities", "Other"])
        with c2:
            raw_amount = st.number_input("Amount:", min_value=0.01, step=5.00, value=15.00)
            currency = st.selectbox("Currency Unit:", ["USD ($)", "THB (฿)"])
        with c3:
            in_notes = st.text_input("Notes / Metadata:")
            st.markdown("<br>", unsafe_allow_html=True)
            submit = st.form_submit_button("Commit to SQL Database", use_container_width=True)

    if submit:
        final_amount = raw_amount * thb_rate if currency == "USD ($)" else raw_amount
        note_suffix = f" (Converted from ${raw_amount:.2f} USD @ {thb_rate:.2f})" if currency == "USD ($)" else ""
        
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO transactions (date, category, amount, notes) VALUES (?, ?, ?, ?)",
                       (str(in_date), in_cat, round(final_amount, 2), in_notes + note_suffix))
        conn.commit()
        conn.close()
        
        st.success("Transaction committed successfully to SQLite database!")
        st.rerun()

    st.markdown("---")

    # SQL Deletion Stage
    st.subheader("Delete Record")
    c_del1, c_del2 = st.columns([3, 1])
    with c_del1:
        default_val = int(df["id"].max()) if not df.empty else 1
        target_id = st.number_input("Enter Target SQL Transaction ID to Delete:", min_value=1, step=1, value=default_val)
    with c_del2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("❌ Execute Delete Query", use_container_width=True):
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM transactions WHERE id = ?", (target_id,))
            conn.commit()
            conn.close()
            st.success(f"Row ID {target_id} deleted successfully!")
            st.rerun()

    st.subheader("Raw Database Table (`transactions`)")
    st.dataframe(df[["id", "date", "category", "amount", "notes"]].style.format({"amount": "฿{:,.2f}"}), use_container_width=True)

# ------------------------------------------
# TAB 2: ANALYTICS
# ------------------------------------------
with tab2:
    st.subheader("Exploratory Data Analysis")
    if not df.empty:
        c_a1, c_a2 = st.columns(2)
        with c_a1:
            fig_pie = px.pie(df, values="Amount", names="Category", title="Category Outflow Distribution", hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
        with c_a2:
            daily_df = df.groupby("Date")["Amount"].sum().reset_index()
            fig_line = px.line(daily_df, x="Date", y="Amount", title="Daily Cash Outflow Velocity (THB)")
            st.plotly_chart(fig_line, use_container_width=True)

# ------------------------------------------
# TAB 3: MACHINE LEARNING SHOWDOWN
# ------------------------------------------
with tab3:
    st.subheader("🔮 4-Way Model Showdown & Multi-Step Forecasting")
    
    ml_base = df.groupby("Date")["Amount"].sum().reset_index()
    ml_base["DayOfMonth"] = ml_base["Date"].dt.day
    ml_base["DayOfWeek"] = ml_base["Date"].dt.weekday
    ml_base["IsWeekend"] = ml_base["DayOfWeek"].apply(lambda x: 1 if x >= 5 else 0)
    
    # Feature Engineering: 7-Day Rolling Average
    ml_base["Rolling_7D_Avg"] = ml_base["Amount"].rolling(window=7, min_periods=1).mean()
    
    X = ml_base[["DayOfMonth", "DayOfWeek", "IsWeekend", "Rolling_7D_Avg"]]
    y = ml_base["Amount"]
    
    if len(ml_base) >= 10:
        models = {
            "Multiple Linear Regression (MLR)": LinearRegression(),
            "Support Vector Regression (SVR)": SVR(kernel="rbf", C=100, gamma=0.1),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        accuracies = {}
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        for name, model in models.items():
            if "Support Vector" in name:
                model.fit(X_scaled, y)
                score = r2_score(y, model.predict(X_scaled))
            else:
                model.fit(X, y)
                score = r2_score(y, model.predict(X))
            accuracies[name] = max(0.0, score)
        
        st.markdown("### 🏆 Model Performance Leaderboard ($R^2$ Variance Score)")
        m_cols = st.columns(4)
        for idx, (name, score) in enumerate(accuracies.items()):
            m_cols[idx].metric(label=name, value=f"{score*100:.2f}%", delta="Champion" if score == max(accuracies.values()) else "Baseline")
            
        st.markdown("---")
        st.markdown("### 📅 Multi-Step 7-Day Future Forecast Horizon")
        
        future_dates = [datetime.now() + timedelta(days=i) for i in range(1, 8)]
        future_df = pd.DataFrame({"Date": future_dates})
        future_df["DayOfMonth"] = future_df["Date"].dt.day
        future_df["DayOfWeek"] = future_df["Date"].dt.weekday
        future_df["IsWeekend"] = future_df["DayOfWeek"].apply(lambda x: 1 if x >= 5 else 0)
        future_df["Rolling_7D_Avg"] = ml_base["Amount"].tail(7).mean()
        
        X_future = future_df[["DayOfMonth", "DayOfWeek", "IsWeekend", "Rolling_7D_Avg"]]
        X_future_scaled = scaler.transform(X_future)
        
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(
            x=ml_base["Date"].tail(7), y=ml_base["Amount"].tail(7),
            name="Actual History", line=dict(color="black", width=3, dash="dot")
        ))
        
        for name, model in models.items():
            preds = model.predict(X_future_scaled) if "Support Vector" in name else model.predict(X_future)
            fig_forecast.add_trace(go.Scatter(x=future_df["Date"], y=preds, name=name, mode="lines+markers"))
            
        fig_forecast.update_layout(
            title="7-Day Multi-Model Prediction Horizon",
            xaxis_title="Calendar Date", yaxis_title="Predicted Cash Outflow (฿)",
            hovermode="x unified"
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
    else:
        st.error("Insufficient historical records to run ML training pipeline.")