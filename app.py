import streamlit as st
import pandas as pd
import joblib

# 1. Set up Page Config
st.set_page_config(
    page_title="Indian House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# 2. Load the Saved Model Pipeline
@st.cache_resource
def load_model():
    return joblib.load('housing_model.pkl')

model = load_model()

# 3. Helper Function for Indian Currency Formatting
def format_inr(number):
    if number >= 10000000:
        return f"₹{number / 10000000:.2f} Crore"
    elif number >= 100000:
        return f"₹{number / 100000:.2f} Lakh"
    else:
        return f"₹{number:,.2f}"

# 4. App Title & Subtitle
st.title("🏠 Indian House Price Prediction Model")
st.markdown("Estimate the market valuation of residential properties based on structural and regional configurations.")
st.write("---")

# 5. Create Columns for Layout Structure
col1, col2 = st.columns(2)

with col1:
    st.subheader("Structural Details")
    area = st.number_input("Total Area (in Sq. Ft.)", min_value=500, max_value=20000, value=4000, step=100)
    bedrooms = st.slider("Number of Bedrooms (BHK)", 1, 6, 3)
    bathrooms = st.slider("Number of Bathrooms", 1, 4, 2)
    stories = st.slider("Total Floors / Stories", 1, 4, 2)
    parking = st.slider("Car Parking Spaces", 0, 3, 1)

with col2:
    st.subheader("Amenities & Locality Features")
    mainroad = st.selectbox("Is it on the Main Road?", ["Yes", "No"])
    guestroom = st.selectbox("Does it have a Guestroom?", ["No", "Yes"])
    basement = st.selectbox("Does it have a Basement?", ["No", "Yes"])
    hotwaterheating = st.selectbox("Does it have Hot Water Heating?", ["No", "Yes"])
    airconditioning = st.selectbox("Does it have Air Conditioning (AC)?", ["Yes", "No"])
    prefarea = st.selectbox("Is it in a Preferred/Premium Locality?", ["Yes", "No"])
    furnishingstatus = st.selectbox("Furnishing Status", ["Furnished", "Semi-Furnished", "Unfurnished"])

st.write("---")

# 6. Prediction Logic
if st.button("Calculate Market Valuation", type="primary", use_container_width=True):
    # Construct input dataframe matching exact training columns
    input_data = pd.DataFrame([{
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'stories': stories,
        'mainroad': mainroad.lower(),
        'guestroom': guestroom.lower(),
        'basement': basement.lower(),
        'hotwaterheating': hotwaterheating.lower(),
        'airconditioning': airconditioning.lower(),
        'parking': parking,
        'prefarea': prefarea.lower(),
        'furnishingstatus': furnishingstatus.lower()
    }])
    
    # Generate Prediction
    prediction = model.predict(input_data)[0]
    formatted_price = format_inr(prediction)
    
    # Display Result
    st.success(f"### Estimated Property Valuation: {formatted_price}")
    
    # Provide a contextual metric summary box
    st.info(f"Calculated average value rate: ₹{int(prediction / area):,} per Sq. Ft.")
