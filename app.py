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
# Data Generation
# -------------------------------------------------------------------
@st.cache_data
def load_data():
    np.random.seed(42)
    N = 800
    area = np.random.randint(800, 5000, N)
    rooms = np.random.randint(1, 7, N)
    locations = np.random.choice(['Downtown', 'Suburbs', 'Rural'], N, p=[0.3, 0.5, 0.2])
    amenities = np.random.randint(0, 10, N)
    age = np.random.randint(0, 100, N)
    parking = np.random.randint(0, 5, N)
    transit_dist = np.random.uniform(0.1, 20.0, N)
    
    # Base formula with non-linear components
    base_price = 50000.0
    price = base_price + (area * 120) + (rooms * 15000) + (amenities * 12000)
    # Additional feature effects
    price -= (age * 600)  # Older homes depreciate
    price += (parking * 7500) # Parking adds value
    price -= (transit_dist * 800) # Farther from transit is slightly cheaper
    
    # Non-linear area effect for Polynomial model to capture later
    price += (area / 100) ** 2 * 60
    
    # Location multipliers
    loc_mult = {'Downtown': 1.6, 'Suburbs': 1.1, 'Rural': 0.8}
    price = price * np.array([loc_mult[l] for l in locations])
    
    # Noise
    noise = np.random.normal(0, 60000, N)
    price += noise
    price = np.maximum(price, 50000) # Minimum cap
    
    df = pd.DataFrame({
        'Area (sqft)': area,
        'Rooms': rooms,
        'Location': locations,
        'Amenities (Count)': amenities,
        'Property Age (Years)': age,
        'Parking Spaces': parking,
        'Distance to Transit (Miles)': np.round(transit_dist, 1),
        'Price ($)': price
    })
    return df

df = load_data()

# -------------------------------------------------------------------
# Preprocessing Helper
# -------------------------------------------------------------------
def preprocess(data):
    # One-hot encoding
    df_encoded = pd.get_dummies(data, columns=['Location'], drop_first=True)
    X = df_encoded.drop('Price ($)', axis=1)
    y = df_encoded['Price ($)']
    return X, y, df_encoded

X, y, df_encoded = preprocess(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------------------------------------------
# Sidebar Layout
# -------------------------------------------------------------------
st.sidebar.title("House Price AI")
st.sidebar.markdown('Predict house prices based on **location**, **area**, **number of rooms**, **amenities**, and more.')
st.sidebar.markdown('---')
st.sidebar.markdown('### Curriculum Mapped')
st.sidebar.markdown('- **Units:** I, II\n- **CLOs:** CLO1, CLO2')
st.sidebar.markdown('---')
st.sidebar.markdown('### Used Tech')
st.sidebar.markdown('- Linear/Polynomial Regression\n- Gradient Descent\n- Cross Validation\n- Over/Under-fitting Analysis')
st.sidebar.markdown('---')
st.sidebar.info('Real-Life Applications: **Zillow**, **MagicBricks**')

# -------------------------------------------------------------------
# Main Content
# -------------------------------------------------------------------
st.title("Advanced House Price Prediction Platform")
st.markdown("A professional End-to-End Machine Learning dashboard leveraging Regression Models.")

tabs = st.tabs(["Data & Exploration", "Model Training & Concepts", "Simulator"])

# ==========================================
# TAB 1: DATA & EXPLORATION
# ==========================================
with tabs[0]:
    st.header("1. Exploratory Data Analysis & Feature Selection")
    st.write("Understand the synthetic dataset simulating real-world housing market trends.")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Properties", len(df))
    col2.metric("Median Price", f"${df['Price ($)'].median():,.0f}")
    col3.metric("Avg Area", f"{df['Area (sqft)'].mean():,.0f} sqft")
    col4.metric("Max Rooms", df['Rooms'].max())
    
    st.markdown("### Raw Dataset Snapshot")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown("### Feature Relationships & Correlation")
    c1, c2 = st.columns(2)
    with c1:
        # Scatter for Area vs Price
        fig1 = px.scatter(df, x="Area (sqft)", y="Price ($)", color="Location", 
                          title="Area vs Price by Location",
                          color_discrete_sequence=['#e78a53', '#5f8787', '#fbcb97'])
        fig1.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#c1c1c1")
        st.plotly_chart(fig1, use_container_width=True)
    
    with c2:
        # Correlation Heatmap
        corr = df_encoded.corr()
        fig2 = px.imshow(corr, text_auto=".2f", aspect="auto", title="Feature Correlation (Feature Selection)", color_continuous_scale="Earth")
        fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#c1c1c1")
        st.plotly_chart(fig2, use_container_width=True)

# ==========================================
# TAB 2: MODEL TRAINING & CONCEPTS
# ==========================================
with tabs[1]:
    st.header("2. Regression Algorithms & Evaluation")
    
    concept_sel = st.selectbox("Select Concept to Explore:", 
                               ["1. Single vs Multivariable Linear Regression", 
                                "2. Polynomial Regression (Over/Under-fitting)", 
                                "3. Gradient Descent Optimization", 
                                "4. Cross-Validation"])
    
    st.markdown("---")
    
    if "Single vs Multivariable" in concept_sel:
        st.subheader("Linear Regression (Single Variable vs Multivariable)")
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.markdown("**Single Variable (Area Only)**")
            X_sv = df[['Area (sqft)']]
            y_sv = df['Price ($)']
            X_tr, X_te, y_tr, y_te = train_test_split(X_sv, y_sv, test_size=0.2, random_state=42)
            
            lr_single = LinearRegression()
            lr_single.fit(X_tr, y_tr)
            y_pr = lr_single.predict(X_te)
            
            st.metric("R² Score (Single)", f"{r2_score(y_te, y_pr):.4f}")
            
            fig_sv = px.scatter(x=X_te['Area (sqft)'], y=y_te, opacity=0.5, labels={'x': 'Area', 'y': 'Price'})
            fig_sv.add_trace(go.Scatter(x=X_te['Area (sqft)'], y=y_pr, mode='lines', name='Linear Fit', line={"color": "#e78a53", "width": 3}))
            fig_sv.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#c1c1c1")
            st.plotly_chart(fig_sv, use_container_width=True)
            
        with col_m2:
            st.markdown("**Multivariable (All Features)**")
            lr_multi = LinearRegression()
            lr_multi.fit(X_train, y_train)
            y_pr_m = lr_multi.predict(X_test)
            
            st.metric("R² Score (Multivariable)", f"{r2_score(y_test, y_pr_m):.4f}")
            
            # Feature Importance
            coef_df = pd.DataFrame({'Feature': X.columns, 'Importance': lr_multi.coef_})
            fig_imp = px.bar(coef_df, x='Importance', y='Feature', orientation='h', title="Feature Coefficients", color_discrete_sequence=['#5f8787'])
            fig_imp.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#c1c1c1", title_font_size=14)
            st.plotly_chart(fig_imp, use_container_width=True)
            
    elif "Polynomial" in concept_sel:
        st.subheader("Polynomial Regression & Overfitting vs Underfitting")
        degree = st.slider("Select Polynomial Degree:", min_value=1, max_value=6, value=2, step=1)
        
        X_poly_base = df[['Area (sqft)']] # Using Area to visually show fitting
        y_poly_base = df['Price ($)']
        X_p_tr, X_p_te, y_p_tr, y_p_te = train_test_split(X_poly_base, y_poly_base, test_size=0.2, random_state=42)
        
        poly = PolynomialFeatures(degree=degree)
        X_p_tr_poly = poly.fit_transform(X_p_tr)
        X_p_te_poly = poly.transform(X_p_te)
        
        lr_poly = LinearRegression()
        lr_poly.fit(X_p_tr_poly, y_p_tr)
        
        # Sort values for smooth curve
        X_smooth = np.linspace(X_poly_base.min(), X_poly_base.max(), 300).reshape(-1, 1)
        y_smooth_pred = lr_poly.predict(poly.transform(pd.DataFrame(X_smooth, columns=['Area (sqft)'])))
        
        train_score = r2_score(y_p_tr, lr_poly.predict(X_p_tr_poly))
        test_score = r2_score(y_p_te, lr_poly.predict(X_p_te_poly))
        
        cc1, cc2 = st.columns(2)
        cc1.metric("Training R² Score", f"{train_score:.4f}")
        cc2.metric("Testing R² Score", f"{test_score:.4f}")
        
        if degree == 1:
            st.warning("Underfitting: Model is too simple to capture the underlying trend mapping Area to Price.")
        elif degree in [2, 3]:
            st.success("Good Fit: Model properly captures the trend without tracking pure noise.")
        else:
            st.error("Overfitting: Model is too complex, fitting noise resulting in wild variations outside data density.")
            
        fig_poly = px.scatter(x=X_p_te['Area (sqft)'], y=y_p_te, opacity=0.5, title=f"Polynomial fit (Degree {degree})", labels={'x': 'Area', 'y': 'Price'})
        fig_poly.add_trace(go.Scatter(x=X_smooth.flatten(), y=y_smooth_pred, mode='lines', name='Poly Curve', line={"color": "#e78a53", "width": 3}))
        fig_poly.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#c1c1c1", yaxis_range=[0, df['Price ($)'].max() * 1.5])
        st.plotly_chart(fig_poly, use_container_width=True)

    elif "Gradient Descent" in concept_sel:
        st.subheader("Gradient Descent Optimization")
        st.markdown("Training a Multivariable Linear Regression model using **Stochastic Gradient Descent (SGD)**.")
        
        epochs_to_simulate = st.slider("Max Iterations (Epochs):", min_value=10, max_value=1000, value=200, step=50)
        learning_rate = st.select_slider("Learning Rate:", options=[0.0001, 0.001, 0.01, 0.1], value=0.01)
        
        # Scaling is critical for SGD
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        sgd = SGDRegressor(max_iter=epochs_to_simulate, eta0=learning_rate, random_state=42, learning_rate='constant')
        sgd.fit(X_train_scaled, y_train)
        y_pred_sgd = sgd.predict(X_test_scaled)
        
        st.metric("Test Set R² Score", f"{r2_score(y_test, y_pred_sgd):.4f}")
        
        # We manually simulate epochs to plot Cost Function history (since sklearn doesn't return loss history natively for SGDRegressor)
        history = []
        sgd_sim = SGDRegressor(max_iter=1, eta0=learning_rate, random_state=42, warm_start=True, learning_rate='constant')
        for e in range(1, epochs_to_simulate+1):
            sgd_sim.fit(X_train_scaled, y_train)
            if e % max(1, epochs_to_simulate//50) == 0:
                y_pred_e = sgd_sim.predict(X_train_scaled)
                loss = mean_squared_error(y_train, y_pred_e)
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
        
        st.markdown(f"Evaluated Multivariable Linear Regression across **{folds}** distinct chunks of data to ensure robustness and avoid overfitting to a specific test set.")
        
        cv_df = pd.DataFrame({'Fold': [f"Fold {i+1}" for i in range(folds)], 'R² Score': scores})
        
        st.metric("Average CV R² Score", f"{scores.mean():.4f}", f"Std Dev: ±{scores.std():.4f}")
        
        fig_cv = px.bar(cv_df, x='Fold', y='R² Score', title="R² Score per Fold", color_discrete_sequence=['#5f8787'])
        fig_cv.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#c1c1c1", yaxis_range=[0, 1])
        st.plotly_chart(fig_cv, use_container_width=True)


# ==========================================
# TAB 3: SIMULATOR
# ==========================================
with tabs[2]:
    st.header("Real-Time Price Simulation Engine")
    st.write("Input property characteristics below to estimate its market value using our trained Multivariable Regression Model.")
    
    # Train robust model on all data
    final_model = LinearRegression()
    final_model.fit(X, y)
    
    with st.form("prediction_form"):
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            in_area = st.number_input(
                "Property Area (sqft)", min_value=500, max_value=8000, value=2500, step=100, 
                help="Total livable interior space. Larger properties generally commend higher market prices."
            )
            in_loc = st.selectbox(
                "Location Type", ['Downtown', 'Suburbs', 'Rural'],
                help="The geographical classification. Downtown typically has a significant premium multiplier."
            )
        with sc2:
            in_rooms = st.number_input(
                "Total Rooms", min_value=1, max_value=10, value=3, step=1,
                help="Count of primary rooms (bedrooms and living rooms). Excludes bathrooms and closets."
            )
            in_amen = st.number_input(
                "Amenities Count", min_value=0, max_value=15, value=5,
                help="Quantifies localized luxury and utility features directly accessible (e.g., swimming pool, gym, smart home features, premium security, clubhouse)."
            )
        with sc3:
            in_age = st.number_input(
                "Property Age (Years)", min_value=0, max_value=150, value=10, step=1,
                help="Age of the property since original construction. Older properties generally see deprecation without significant renovation."
            )
            in_park = st.number_input(
                "Parking Spaces", min_value=0, max_value=5, value=1, step=1,
                help="Number of dedicated parking spots, garages, or driveway space allocations."
            )
            
        in_transit = st.slider(
            "Distance to Public Transit (Miles)", min_value=0.0, max_value=30.0, value=2.0, step=0.1,
            help="Proximity in miles to the nearest major transit hub (train station, bus terminal). Properties closer to transit typically hold higher value."
        )
            
        submit = st.form_submit_button("Predict Market Price")

    if submit:
        # Prepare input df
        input_data = pd.DataFrame({
            'Area (sqft)': [in_area],
            'Rooms': [in_rooms],
            'Amenities (Count)': [in_amen],
            'Property Age (Years)': [in_age],
            'Parking Spaces': [in_park],
            'Distance to Transit (Miles)': [in_transit]
        })
        # One hot encode location logic manually to match training columns
        for col in X.columns:
            if 'Location_' in col:
                input_data[col] = 1 if ('Location_' + in_loc) == col else 0
                
        # align columns
        input_data = input_data.reindex(columns=X.columns, fill_value=0)
        
        pred_price = final_model.predict(input_data)[0]
        
        st.markdown("---")
        st.markdown("<h3 style='text-align: center; color: #c1c1c1;'>Estimated Value</h3>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center; color: #e78a53; font-size: 3rem;'>${pred_price:,.0f}</h1>", unsafe_allow_html=True)
