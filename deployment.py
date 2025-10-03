
import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
        page_title="Delivery Time",   # Browser tab title
        page_icon="📊",                 # Favicon
        layout="wide",                  # "centered" (default) or "wide"
        initial_sidebar_state="expanded"  # "auto", "expanded", "collapsed"
)

@st.cache_data
def loading_scalers():
    """Load preprocessing artifacts from pickle files.

    This helper reads three pickle files from the current working
    directory and returns the deserialized objects. It wraps the
    operation in a Streamlit spinner and reports an error in the
    Streamlit app if loading fails.

    Returns
    -------
    tuple
        A tuple (categorical_encoder, x_scaler, y_scaler) containing
        the loaded preprocessing objects.

    Raises
    ------
    RuntimeError
        If any of the pickle files cannot be opened or unpickled.
    """

    with st.spinner("Loading preprocessors..."):
        try:
            with open(r"categorical_encoder.pkl", "rb") as f:
                categorical_encoder = pickle.load(f)
            with open(r"x_scaler.pkl", "rb") as f:
                x_scaler = pickle.load(f)
            with open(r"y_scaler.pkl", "rb") as f:
                y_scaler = pickle.load(f)
            with open(r"Gradien_model.pkl", "rb") as f:
                gradient_model = pickle.load(f)
            with open(r"LGBM_model.pkl", "rb") as f:
                LGBM = pickle.load(f)
        except Exception as exc:
            # Surface the error inside Streamlit UI and raise to allow
            # programmatic handling during tests or when imported.
            st.error(f"Preprocessors loading failed: {exc}")
            raise RuntimeError("Failed to load preprocessing artifacts") from exc

    return categorical_encoder, x_scaler, y_scaler, gradient_model, LGBM

categorical_encoder, x_scaler, y_scaler, gradient_model, LGBM = loading_scalers()

st.title("Amazon Delivery Time")

dataframe_col_order=['Agent_Age', 'Agent_Rating', 'Store_Latitude', 'Store_Longitude',
       'Drop_Latitude', 'Drop_Longitude', 'Weather', 'Traffic', 'Vehicle',
       'Area','Category', 'Distance_km', 'Delay',
       'Order_day', 'Pickup_day_part']

label_column=['Weather', 'Traffic', 'Vehicle', 'Area', 'Category', 'Pickup_day_part']


x_numeric_columns=['Agent_Age', 'Agent_Rating', 'Store_Latitude', 'Store_Longitude',
       'Drop_Latitude', 'Drop_Longitude', 'Distance_km', 'Delay', 'Order_day']

y_numeric_column="Delivery_Time"

ml_columns_order=['Weather', 'Traffic', 'Vehicle', 'Area', 'Category', 'Pickup_day_part',
       'Agent_Age', 'Agent_Rating', 'Store_Latitude', 'Store_Longitude',
       'Drop_Latitude', 'Drop_Longitude', 'Distance_km', 'Delay', 'Order_day']

## Form and Input

with st.form("my_form"):
    col1, col2, col3 = st.columns(3)
    # First Columns iNput
    with col1:
        agent_age=st.number_input("Agent Age", min_value=0,max_value=70, value='min')
        agent_rating=st.number_input("Agent Rating", min_value=1,max_value=6, value='min')
        store_latitude=st.number_input("Store Latitude", min_value=0.0,max_value=40.914057, value='min', step=0.05)
        store_longitude=st.number_input("Store Longitude", min_value=0.0,max_value=100.00, value='min', step=0.5)
        drop_latitude=st.number_input("Drop Latitude", min_value=0.0,max_value=40.914057, value='min', step=0.05)
        drop_longitude=st.number_input("Drop Longitude", min_value=0.0,max_value=100.00, value='min', step=0.5)


    with col2:
# Second Columns Input
        weather=st.selectbox("Select Weather",['Sunny', 'Stormy', 'Sandstorms', 'Cloudy', 'Fog', 'Windy'], index=0, accept_new_options=False)
        traffic= st.selectbox("Select Traffic Type", ['High', 'Jam', 'Low', 'Medium'], index=0, accept_new_options=False)
        vehicle= st.selectbox("Select Vehicle Type", ['motorcycle', 'scooter', 'van', 'bicycle'], index=0, accept_new_options=False)
        area= st.selectbox("Select Area Type", ['Urban', 'Metropolitian', 'Semi-Urban', 'Other'], index=0, accept_new_options=False)
        category=st.selectbox("Product Type",['Clothing', 'Electronics', 'Sports', 'Cosmetics', 'Toys', 'Snacks',
       'Shoes', 'Apparel', 'Jewelry', 'Outdoors', 'Grocery', 'Books',
       'Kitchen', 'Home', 'Pet Supplies', 'Skincare'],index=0, accept_new_options=False)
        distance=st.number_input("Distance Between Store and Drop", min_value=0.0,max_value=100.00, value='min', step=0.5)
    
    # third Columns Input
    with col3:
        delay = st.number_input("Time Taken to pickup Order", min_value=0,max_value=1500, value=15, step=5)
        weekdays = {"Monday": 0,"Tuesday": 1,"Wednesday": 2,"Thursday": 3,"Friday": 4,"Saturday": 5, "Sunday": 6 }
        day=weekdays[st.selectbox("Select Order Day",["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], index=0, accept_new_options=False)]
        pickup_part=st.selectbox("Picked up in: ",['Morning', 'Evening', 'Afternoon', 'Night'], index=0, accept_new_options=False)
        data=pd.DataFrame([{
            'Agent_Age':agent_age, 'Agent_Rating':agent_rating, 'Store_Latitude':store_latitude, 'Store_Longitude':store_longitude,
        'Drop_Latitude':drop_latitude, 'Drop_Longitude':drop_longitude, 'Weather':weather, 'Traffic':traffic, 'Vehicle':vehicle,
        'Area':area,'Category':category, 'Distance_km':distance, 'Delay':delay,
        'Order_day':day, 'Pickup_day_part':pickup_part
        }])
    st.form_submit_button('Update values')


st.dataframe(data)

@st.cache_data
def _data_loading(data,model="LightGBM"):
    x_scaled=pd.DataFrame()
    x_scaled[label_column]=categorical_encoder.transform(data[label_column])
    x_scaled[x_numeric_columns]=x_scaler.transform(data[x_numeric_columns])
    x_scaled=x_scaled[ml_columns_order]
    if model=="LightGBM":
        model=LGBM
        y=y_scaler.inverse_transform(pd.DataFrame([model.predict(x_scaled)]))
        data[y_numeric_column]=y
        return data
    else:
        y=y_scaler.inverse_transform(pd.DataFrame([model.predict(x_scaled)]))
        data[y_numeric_column]=y
        return data
    

with st.status("Processing and Predicting in Progress", expanded=True) as status:
    st.write("Previewing and Loading Data for Processing...")
    st.dataframe(_data_loading(data))

st.write("So far Excuted")


