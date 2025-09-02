import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Streamlit Config ---
st.set_page_config(page_title="🌾 Crop Yield Prediction", layout="wide")

st.title("🌾 Crop Yield Prediction App")

# --- Load Data ---
df = pd.read_excel("crop yield data sheet.xlsx")

st.sidebar.header("Dataset Info")
st.sidebar.write(f"**Shape:** {df.shape}")
st.sidebar.write(f"**Columns:** {list(df.columns)}")

# --- Data Cleaning ---
df = df[df['Temperatue'] != ':']
df['Temperatue'] = df['Temperatue'].astype(float)
df = df.fillna(df.median(numeric_only=True))

# --- Split ---
X = df.drop('Yeild (Q/acre)', axis=1)
y = df['Yeild (Q/acre)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 EDA", "🤖 Model Training", "📈 Results"])

with tab1:
    st.header("📊 Exploratory Data Analysis")

    st.subheader("Basic Stats")
    st.dataframe(df.describe())

    st.subheader("Rainfall Trend")
    st.line_chart(df["Rain Fall (mm)"])

    st.subheader("Fertilizer Trend")
    st.line_chart(df["Fertilizer"])

    st.subheader("Temperature Trend")
    st.line_chart(df["Temperatue"])

    st.subheader("Nitrogen (N), Phosphorus (P), Potassium (K)")
    st.bar_chart(df[["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"]])

    st.subheader("Yield Distribution")
    st.bar_chart(df["Yeild (Q/acre)"])

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    st.pyplot(fig)

with tab2:
    st.header("🤖 Model Selection & Training")

    model_option = st.selectbox("Select Model", ["Decision Tree Regressor", "Random Forest Regressor"])

    if model_option == "Decision Tree Regressor":
        params = {
            "max_depth": [2, 4, 6, 8],
            "min_samples_split": [2, 4, 6, 8],
            "min_samples_leaf": [2, 4, 6, 8],
            "random_state": [0, 42]
        }
        regressor = DecisionTreeRegressor()
    else:
        params = {
            "n_estimators": [100, 200],
            "max_depth": [2, 4, 6, 8],
            "min_samples_split": [2, 4, 6, 8],
            "min_samples_leaf": [2, 4, 6, 8],
            "random_state": [0, 42]
        }
        regressor = RandomForestRegressor()

    if st.button("Start Training"):
        grid = GridSearchCV(regressor, params, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        st.session_state['model'] = best_model
        st.session_state['y_pred'] = y_pred

        st.success("✅ Model trained successfully!")
        st.write("**Best Params:**", grid.best_params_)
        st.write(f"**Train R2:** {best_model.score(X_train, y_train):.3f}")

with tab3:
    st.header("📈 Results & Feature Importance")

    if 'model' in st.session_state and 'y_pred' in st.session_state:
        best_model = st.session_state['model']
        y_pred = st.session_state['y_pred']

        st.subheader("Prediction vs Actuals")
        comparison_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        st.line_chart(comparison_df.reset_index(drop=True))

        st.subheader("Metrics")
        st.write(f"**R2 Score:** {r2_score(y_test, y_pred):.3f}")
        st.write(f"**MSE:** {mean_squared_error(y_test, y_pred):.3f}")
        st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred):.3f}")

        st.subheader("Feature Importance")
        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": best_model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        st.bar_chart(importance_df.set_index("Feature"))
    else:
        st.info("🔍 Please train a model first in the 'Model Training' tab.")
