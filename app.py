import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="House Price Predictor", layout="wide")

# Custom CSS for refined Dark Mode styling using user's properties
st.markdown("""
<style>
/* Streamlit's default fonts and paddings */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
/* Cards for Metrics */
[data-testid="stMetricValue"] {
    font-size: 2rem;
    color: #e78a53; /* Primary Color */
}
/* Headings */
h1, h2, h3 {
    font-weight: 600 !important;
}
/* Hr lines */
hr {
    border-color: #343434 !important; /* Muted border */
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Data Loading — Ames Housing Dataset (Kaggle)
# Source: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
# -------------------------------------------------------------------
DATASET_URL = "https://raw.githubusercontent.com/wblakecannon/ames/master/data/housing.csv"

# Selected features for the dashboard (mapped from 80+ columns to a curated subset)
SELECTED_FEATURES = [
    'Gr Liv Area',       # Above-grade living area (sqft)
    'Overall Qual',      # Overall material and finish quality (1–10)
    'TotRms AbvGrd',     # Total rooms above grade (excl. bathrooms)
    'Garage Cars',       # Garage capacity (number of cars)
    'Full Bath',         # Full bathrooms above grade
    'Year Built',        # Year of construction
    'Total Bsmt SF',     # Total basement area (sqft)
    'Neighborhood',      # Physical location within Ames
]
TARGET = 'SalePrice'

# Human-friendly labels for display
FEATURE_LABELS = {
    'Gr Liv Area': 'Living Area (sqft)',
    'Overall Qual': 'Overall Quality (1-10)',
    'TotRms AbvGrd': 'Total Rooms',
    'Garage Cars': 'Garage Capacity (Cars)',
    'Full Bath': 'Full Bathrooms',
    'Year Built': 'Year Built',
    'Total Bsmt SF': 'Basement Area (sqft)',
    'Neighborhood': 'Neighborhood',
}

@st.cache_data
def load_data():
    """Load the Ames Housing dataset from GitHub mirror and select curated features."""
    raw = pd.read_csv(DATASET_URL)

    # Keep only selected features + target
    cols_to_keep = SELECTED_FEATURES + [TARGET]
    df = raw[cols_to_keep].copy()

    # Drop rows with any missing values in our selected columns
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Compute a derived feature: Property Age at time of sale (approx)
    # Yr Sold is not in our selection, so we use 2010 as the dataset's median sale year
    df['Property Age'] = 2010 - df['Year Built']

    # Rename for display friendliness
    df.rename(columns=FEATURE_LABELS, inplace=True)
    df.rename(columns={TARGET: 'Sale Price ($)'}, inplace=True)

    return df

df = load_data()

# -------------------------------------------------------------------
# Preprocessing Helper
# -------------------------------------------------------------------
def preprocess(data):
    """One-hot encode categoricals and split X / y."""
    df_encoded = pd.get_dummies(data, columns=['Neighborhood'], drop_first=True)
    X = df_encoded.drop('Sale Price ($)', axis=1)
    y = df_encoded['Sale Price ($)']
    return X, y, df_encoded

X, y, df_encoded = preprocess(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------------------------------------------
# Sidebar Layout
# -------------------------------------------------------------------
st.sidebar.title("House Price AI")
st.sidebar.markdown(
    'Predict **Ames, Iowa** house prices using the '
    '[Kaggle Ames Housing Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).'
)
st.sidebar.markdown('---')
st.sidebar.markdown('### Dataset Info')
st.sidebar.markdown(
    f'- **Source:** Ames Housing (Kaggle)\n'
    f'- **Records:** {len(df):,}\n'
    f'- **Features used:** {len(SELECTED_FEATURES)}\n'
    f'- **Target:** Sale Price'
)
st.sidebar.markdown('---')
st.sidebar.markdown('### Curriculum Mapped')
st.sidebar.markdown('- **Units:** I, II\n- **CLOs:** CLO1, CLO2')
st.sidebar.markdown('---')
st.sidebar.markdown('### Used Tech')
st.sidebar.markdown('- Linear/Polynomial Regression\n- Gradient Descent\n- Cross Validation\n- Over/Under-fitting Analysis')
st.sidebar.markdown('---')
st.sidebar.info('Real-Life Applications: **Zillow**, **MagicBricks**, **Realtor.com**')

# -------------------------------------------------------------------
# Main Content
# -------------------------------------------------------------------
st.title("Advanced House Price Prediction Platform")
st.markdown(
    "A professional End-to-End Machine Learning dashboard leveraging Regression Models "
    "on the **[Ames Housing Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)**."
)

tabs = st.tabs(["Data & Exploration", "Model Training & Concepts", "Simulator"])

# ==========================================
# TAB 1: DATA & EXPLORATION
# ==========================================
with tabs[0]:
    st.header("1. Exploratory Data Analysis & Feature Selection")
    st.write("Explore the **Ames Housing Dataset** — 2,930 residential property sales in Ames, Iowa (2006–2010).")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Properties", f"{len(df):,}")
    col2.metric("Median Price", f"${df['Sale Price ($)'].median():,.0f}")
    col3.metric("Avg Living Area", f"{df['Living Area (sqft)'].mean():,.0f} sqft")
    col4.metric("Max Quality", int(df['Overall Quality (1-10)'].max()))

    st.markdown("### Raw Dataset Snapshot")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### Feature Relationships & Correlation")
    c1, c2 = st.columns(2)
    with c1:
        # Scatter for Living Area vs Sale Price
        fig1 = px.scatter(
            df, x="Living Area (sqft)", y="Sale Price ($)", color="Neighborhood",
            title="Living Area vs Sale Price by Neighborhood",
            color_discrete_sequence=['#e78a53','#5f8787','#fbcb97','#8b5e3c','#a3b18a','#d4a373','#6b705c','#b98b73','#cb997e','#ddbea9'],
            opacity=0.7,
        )
        fig1.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#c1c1c1",
            showlegend=False,
        )
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        # Correlation Heatmap (numeric columns only)
        numeric_cols = df_encoded.select_dtypes(include=[np.number])
        # Pick the original numeric features + target for a clear heatmap
        heatmap_cols = [
            'Living Area (sqft)', 'Overall Quality (1-10)', 'Total Rooms',
            'Garage Capacity (Cars)', 'Full Bathrooms', 'Year Built',
            'Basement Area (sqft)', 'Property Age', 'Sale Price ($)'
        ]
        heatmap_cols = [c for c in heatmap_cols if c in numeric_cols.columns]
        corr = numeric_cols[heatmap_cols].corr()
        fig2 = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            title="Feature Correlation (Feature Selection)",
            color_continuous_scale="Earth",
        )
        fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#c1c1c1")
        st.plotly_chart(fig2, use_container_width=True)

# ==========================================
# TAB 2: MODEL TRAINING & CONCEPTS
# ==========================================
with tabs[1]:
    st.header("2. Regression Algorithms & Evaluation")

    concept_sel = st.selectbox(
        "Select Concept to Explore:",
        [
            "1. Single vs Multivariable Linear Regression",
            "2. Polynomial Regression (Over/Under-fitting)",
            "3. Gradient Descent Optimization",
            "4. Cross-Validation",
        ],
    )

    st.markdown("---")

    if "Single vs Multivariable" in concept_sel:
        st.subheader("Linear Regression (Single Variable vs Multivariable)")
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.markdown("**Single Variable (Living Area Only)**")
            X_sv = df[['Living Area (sqft)']]
            y_sv = df['Sale Price ($)']
            X_tr, X_te, y_tr, y_te = train_test_split(X_sv, y_sv, test_size=0.2, random_state=42)

            lr_single = LinearRegression()
            lr_single.fit(X_tr, y_tr)
            y_pr = lr_single.predict(X_te)

            st.metric("R² Score (Single)", f"{r2_score(y_te, y_pr):.4f}")

            fig_sv = px.scatter(x=X_te['Living Area (sqft)'], y=y_te, opacity=0.5, labels={'x': 'Living Area (sqft)', 'y': 'Sale Price ($)'})
            fig_sv.add_trace(go.Scatter(
                x=X_te['Living Area (sqft)'].sort_values(), y=lr_single.predict(X_te.sort_values(by='Living Area (sqft)')),
                mode='lines', name='Linear Fit', line={"color": "#e78a53", "width": 3}
            ))
            fig_sv.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#c1c1c1")
            st.plotly_chart(fig_sv, use_container_width=True)

        with col_m2:
            st.markdown("**Multivariable (All Features)**")
            lr_multi = LinearRegression()
            lr_multi.fit(X_train, y_train)
            y_pr_m = lr_multi.predict(X_test)

            st.metric("R² Score (Multivariable)", f"{r2_score(y_test, y_pr_m):.4f}")

            # Feature Importance — show top 10 absolute coefficients
            coef_df = pd.DataFrame({'Feature': X.columns, 'Importance': lr_multi.coef_})
            coef_df['Abs'] = coef_df['Importance'].abs()
            coef_df = coef_df.nlargest(10, 'Abs')
            fig_imp = px.bar(
                coef_df, x='Importance', y='Feature', orientation='h',
                title="Top 10 Feature Coefficients", color_discrete_sequence=['#5f8787'],
            )
            fig_imp.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#c1c1c1", title_font_size=14,
            )
            st.plotly_chart(fig_imp, use_container_width=True)

    elif "Polynomial" in concept_sel:
        st.subheader("Polynomial Regression & Overfitting vs Underfitting")
        degree = st.slider("Select Polynomial Degree:", min_value=1, max_value=6, value=2, step=1)

        X_poly_base = df[['Living Area (sqft)']]
        y_poly_base = df['Sale Price ($)']
        X_p_tr, X_p_te, y_p_tr, y_p_te = train_test_split(X_poly_base, y_poly_base, test_size=0.2, random_state=42)

        poly = PolynomialFeatures(degree=degree)
        X_p_tr_poly = poly.fit_transform(X_p_tr)
        X_p_te_poly = poly.transform(X_p_te)

        lr_poly = LinearRegression()
        lr_poly.fit(X_p_tr_poly, y_p_tr)

        # Sort values for smooth curve
        X_smooth = np.linspace(X_poly_base.min().values[0], X_poly_base.max().values[0], 300).reshape(-1, 1)
        y_smooth_pred = lr_poly.predict(poly.transform(X_smooth))

        train_score = r2_score(y_p_tr, lr_poly.predict(X_p_tr_poly))
        test_score = r2_score(y_p_te, lr_poly.predict(X_p_te_poly))

        cc1, cc2 = st.columns(2)
        cc1.metric("Training R² Score", f"{train_score:.4f}")
        cc2.metric("Testing R² Score", f"{test_score:.4f}")

        if degree == 1:
            st.warning("Underfitting: Model is too simple to capture the underlying trend in the Ames dataset.")
        elif degree in [2, 3]:
            st.success("Good Fit: Model properly captures the non-linear relationship between Living Area and Sale Price.")
        else:
            st.error("Overfitting: Model is too complex, fitting noise resulting in wild variations outside data density.")

        fig_poly = px.scatter(
            x=X_p_te['Living Area (sqft)'], y=y_p_te, opacity=0.5,
            title=f"Polynomial Fit (Degree {degree})",
            labels={'x': 'Living Area (sqft)', 'y': 'Sale Price ($)'},
        )
        fig_poly.add_trace(go.Scatter(
            x=X_smooth.flatten(), y=y_smooth_pred, mode='lines',
            name='Poly Curve', line={"color": "#e78a53", "width": 3}
        ))
        fig_poly.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#c1c1c1",
            yaxis_range=[0, df['Sale Price ($)'].max() * 1.3],
        )
        st.plotly_chart(fig_poly, use_container_width=True)

    elif "Gradient Descent" in concept_sel:
        st.subheader("Gradient Descent Optimization")
        st.markdown("Training a Multivariable Linear Regression model using **Stochastic Gradient Descent (SGD)** on real Ames Housing data.")

        epochs_to_simulate = st.slider("Max Iterations (Epochs):", min_value=10, max_value=1000, value=200, step=50)
        learning_rate = st.select_slider("Initial Learning Rate (η₀):", options=[0.0001, 0.001, 0.005, 0.01, 0.05], value=0.01)

        # Scaling BOTH X and y is critical for SGD convergence
        # (Sale prices are ~100k–600k; without y-scaling, gradients explode)
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

        # Use 'invscaling' schedule: η = η₀ / t^0.25 (decays over time, prevents divergence)
        sgd = SGDRegressor(max_iter=epochs_to_simulate, eta0=learning_rate, random_state=42,
                           learning_rate='invscaling', tol=None)
        sgd.fit(X_train_scaled, y_train_scaled)
        y_pred_sgd_scaled = sgd.predict(X_test_scaled)
        y_pred_sgd = scaler_y.inverse_transform(y_pred_sgd_scaled.reshape(-1, 1)).ravel()

        st.metric("Test Set R² Score", f"{r2_score(y_test, y_pred_sgd):.4f}")

        # Manually simulate epochs to plot Cost Function history
        history = []
        sgd_sim = SGDRegressor(max_iter=1, eta0=learning_rate, random_state=42,
                               warm_start=True, learning_rate='invscaling', tol=None)
        for e in range(1, epochs_to_simulate + 1):
            sgd_sim.fit(X_train_scaled, y_train_scaled)
            if e % max(1, epochs_to_simulate // 50) == 0:
                y_pred_e = sgd_sim.predict(X_train_scaled)
                loss = mean_squared_error(y_train_scaled, y_pred_e)
                history.append({'Epoch': e, 'MSE Loss': loss})

        hist_df = pd.DataFrame(history)
        fig_loss = px.line(hist_df, x='Epoch', y='MSE Loss', title="Cost Function Reduction over Epochs", markers=True)
        fig_loss.update_traces(line_color="#e78a53")
        fig_loss.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#c1c1c1")
        st.plotly_chart(fig_loss, use_container_width=True)

    elif "Cross-Validation" in concept_sel:
        st.subheader("Cross-Validation (k-fold)")
        folds = st.slider("Number of Folds (k)", min_value=2, max_value=10, value=5, step=1)

        lr_cv = LinearRegression()
        scores = cross_val_score(lr_cv, X, y, cv=folds, scoring='r2')

        st.markdown(
            f"Evaluated Multivariable Linear Regression across **{folds}** distinct folds of the "
            f"Ames Housing dataset to ensure robustness and avoid overfitting to a specific test set."
        )

        cv_df = pd.DataFrame({'Fold': [f"Fold {i+1}" for i in range(folds)], 'R² Score': scores})

        st.metric("Average CV R² Score", f"{scores.mean():.4f}", f"Std Dev: ±{scores.std():.4f}")

        fig_cv = px.bar(cv_df, x='Fold', y='R² Score', title="R² Score per Fold", color_discrete_sequence=['#5f8787'])
        fig_cv.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#c1c1c1",
            yaxis_range=[0, 1],
        )
        st.plotly_chart(fig_cv, use_container_width=True)

# ==========================================
# TAB 3: SIMULATOR
# ==========================================
with tabs[2]:
    st.header("Real-Time Price Simulation Engine")
    st.write(
        "Input property characteristics below to estimate its market value "
        "using the trained Multivariable Regression Model on the Ames Housing data."
    )

    # Train robust model on all data
    final_model = LinearRegression()
    final_model.fit(X, y)

    # Get unique neighborhoods for the dropdown
    neighborhoods = sorted(df['Neighborhood'].unique().tolist())

    with st.form("prediction_form"):
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            in_area = st.number_input(
                "Living Area (sqft)", min_value=300, max_value=6000, value=1500, step=50,
                help="Above-grade living area in square feet. The average in Ames is ~1,500 sqft."
            )
            in_neighborhood = st.selectbox(
                "Neighborhood", neighborhoods,
                help="Physical location within Ames, Iowa. Different neighborhoods carry different premiums."
            )
            in_bsmt = st.number_input(
                "Basement Area (sqft)", min_value=0, max_value=3000, value=1000, step=50,
                help="Total basement area in square feet. Enter 0 if no basement."
            )
        with sc2:
            in_quality = st.slider(
                "Overall Quality (1-10)", min_value=1, max_value=10, value=5,
                help="Rates the overall material and finish of the house. 10 = Very Excellent."
            )
            in_rooms = st.number_input(
                "Total Rooms (above grade)", min_value=2, max_value=15, value=6, step=1,
                help="Total rooms above grade (does not include bathrooms)."
            )
            in_bath = st.number_input(
                "Full Bathrooms", min_value=0, max_value=5, value=1, step=1,
                help="Number of full bathrooms above grade."
            )
        with sc3:
            in_year = st.number_input(
                "Year Built", min_value=1870, max_value=2025, value=1990, step=1,
                help="Original construction date of the house."
            )
            in_garage = st.number_input(
                "Garage Capacity (Cars)", min_value=0, max_value=5, value=2, step=1,
                help="Size of garage in car capacity."
            )

        submit = st.form_submit_button("Predict Market Price")

    if submit:
        # Prepare input DataFrame matching training columns
        input_data = pd.DataFrame({
            'Living Area (sqft)': [in_area],
            'Overall Quality (1-10)': [in_quality],
            'Total Rooms': [in_rooms],
            'Garage Capacity (Cars)': [in_garage],
            'Full Bathrooms': [in_bath],
            'Year Built': [in_year],
            'Basement Area (sqft)': [in_bsmt],
            'Property Age': [2010 - in_year],
        })

        # One-hot encode neighborhood to match training columns
        for col in X.columns:
            if col.startswith('Neighborhood_'):
                neighborhood_name = col.replace('Neighborhood_', '')
                input_data[col] = 1 if neighborhood_name == in_neighborhood else 0

        # Align columns with training data
        input_data = input_data.reindex(columns=X.columns, fill_value=0)

        pred_price = final_model.predict(input_data)[0]

        st.markdown("---")
        st.markdown("<h3 style='text-align: center; color: #c1c1c1;'>Estimated Market Value</h3>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center; color: #e78a53; font-size: 3rem;'>${pred_price:,.0f}</h1>", unsafe_allow_html=True)
        st.caption("Based on the Ames Housing Dataset regression model. Actual prices may vary.")
