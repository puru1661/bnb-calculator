import streamlit as st
import pandas as pd
import requests
import datetime
import sqlite3
import math
import lightgbm as lgb
import plotly.express as px
import numpy as np
from joblib import dump, load

locs = pd.read_csv('locations.csv')
abb = pd.read_csv('abb.csv')
rental = pd.read_csv('rental.csv')
props = list(set(locs['prop_name']))



st.set_page_config(page_title='Bnb Calculator',  layout='wide', page_icon= '🎈')
col1, col2 = st.columns([1, 8])
with col1:
    st.image("bnb.png",width=100)

with col2:
    st.title("Airbnb Calculator for Dubai")


if 'beds' not in st.session_state:
    st.session_state.beds = 0
if 'vals' not in st.session_state:
    st.session_state.vals = 0
if 'baths' not in st.session_state:
    st.session_state.baths = 1
if 'capacity' not in st.session_state:
    st.session_state.capacity = 1
if 'results' not in st.session_state:
    st.session_state.results = None
if 'rental' not in st.session_state:
    st.session_state.rental = None
if 'occupancy' not in st.session_state:
    st.session_state.occupancy = 20

option = st.selectbox(
    "Select Building",
    props,
)

sel = locs[locs['prop_name']==option].head(1)

lat = sel.lat.values.astype(float)
lng = sel.lng.values.astype(float)


model = load('calculator.joblib')


def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in kilometers
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    a = (math.sin(dLat / 2) * math.sin(dLat / 2) +
         math.sin(dLon / 2) * math.sin(dLon / 2) * math.cos(lat1) * math.cos(lat2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def filter_data(df,lat,lng,radius,beds):


    df['distance'] = df.apply(lambda row: haversine(lat, lng, row['lat'], row['lng']), axis=1)
    df = df[(df['distance']<radius)& (df['beds']==beds)]
    
    return df



def predict_adr_lgb(occ, lat, lng, beds, baths, capacity, model):
    # Prepare the feature vector in the correct order
    features = np.array([[occ, lat, lng, beds, baths, capacity]])
    
    # Predict using the provided LightGBM model
    try:
        prediction = model.predict(features)[0]  # model.predict returns an array of predictions
        return prediction
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

def calculate_revenue(occupancy, adr):
    # Calculating the revenue
    return 365 * occupancy * adr / 100 

def revenue(data, lat, lng, beds, baths, capacity):
    # Calculate occupancy statistics
    occ_stats = {
        'mean_occ': data['occ'].mean(),
        'median_occ': data['occ'].median(),
        'q3_occ': data['occ'].quantile(0.75),
        'p90_occ': data['occ'].quantile(0.90)
    }

    # Collect data for new DataFrame
    stats = ['mean', 'q3', 'p90']
    perf = ['Average','Good','Great']
    results = []

    for stat in stats:
        occ_key = f"{stat}_occ"
        # Use the predict_adr function to get ADR for the given occupancy
        predicted_adr = predict_adr_lgb(occ_stats[occ_key], lat, lng, beds, baths, capacity,model)
        # Calculate revenue using the predicted ADR and the occupancy statistic
        revenue = calculate_revenue(occ_stats[occ_key], predicted_adr)
        results.append({
            
            'Occupancy (%)': occ_stats[occ_key],
            'ADR': predicted_adr,
            'Estimated Revenue ($)': revenue
        })

    
    results_df = pd.DataFrame(results)
    results_df['Performance'] = perf
    # Return the DataFrame
    return results_df


with st.form(key='property_form'):
    # Create a selectbox for the number of beds
    beds = st.selectbox('Select number of beds:', options=[0, 1, 2, 3, 4, 5])

    # Create a selectbox for the number of baths
    baths = st.selectbox('Select number of baths:', options=[1, 2, 3, 4, 5])

    # Create a slider for selecting capacity
    capacity = st.slider('Select capacity:', min_value=1, max_value=12, value=1)

    # Create a submit button
    submit_button = st.form_submit_button("Submit")

if submit_button:

    st.session_state.selected_beds = beds
    st.session_state.selected_baths = baths
    st.session_state.selected_capacity = capacity

    #st.write(f"You selected {beds} beds, {baths} baths, and a capacity of {capacity}.")
    df = filter_data(abb,lat, lng,0.05,beds)
    st.session_state.vals = len(df) 
    rental_df = filter_data(rental,lat,lng,0.05,beds)
    res = revenue(df,lat, lng,beds,baths,capacity)
    st.session_state.results = res
    st.session_state.rental = rental_df

    


if st.session_state.vals >0:
    st.title('Performance Overview')
   
    col1, col2, col3 = st.columns(3)
    performance_containers = {
        'Average': col1,
        'Good': col2,
        'Great': col3
    }

    st.markdown("""
    <style>
    div.stMarkdown {
        border: 1px solid #f0f0f0;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    # Iterate through each row in the DataFrame and display the information
    for index, row in st.session_state.results.iterrows():
        performance = row['Performance']
        occupancy = row['Occupancy (%)']
        adr = row['ADR']
        revenue = row['Estimated Revenue ($)']

        container = performance_containers[performance]
        with container:
            # Using one markdown block to create a single bordered "container"
            markdown_content = f"""
    <div class="border-md">
        <h4>{performance} Performance</h4>
        <p><strong>Expected Revenue:</strong> AED {revenue:,.2f}</p>
        <p><strong>Occupancy:</strong> {occupancy:.2f}%</p>
        <p><strong>ADR:</strong> AED {adr:,.2f}</p>
    </div>
    """
            container.markdown(markdown_content, unsafe_allow_html=True)
            
        # st.markdown(f"""
        # ### {performance} Performance:
        # {performance} occupancy in this building/region is **{occupancy:.2f}%**. 
        # With this performance, you can expect to make **AED {revenue:,.2f}** 
        # at an ADR of **AED {adr:.2f}**.
        # """)


    # st.title('Predicted Revenue based on Occupancy]')
    # value = st.slider(
    # 'Select Occupancy',  # Title of the slider
    # min_value=0,      # Starting value of the slider
    # max_value=90,     # Maximum value of the slider
    # value=20,          # Initial value of the slider
    # step=5             # Step size for the slider
    # )   


    # predicted_adr = predict_adr_lgb(value, lat, lng, beds, baths, capacity,model)
    # rev = calculate_revenue(value, predicted_adr)

    # st.markdown(f"""
    #     With **{value:.2f}%** occupancy, you can expect to make **AED {np.round(rev,0)}**
    #     """)


    #df['capacity'] = df['capacity'].astype(int)
    st.title('Rental Data from PropertyFinder/Bayut')
    st.write("Average Rental ask rate for {} Bed Apartment in {} is **AED {}**".format(beds,option,np.round(st.session_state.rental['price'].mean(),0)))
    fig = px.histogram(st.session_state.rental, x="price",nbins=20)

    st.header("Distribution of Rental Ask rates")
    # Display the figure in Streamlit
    st.markdown("The below chart shows distribution of what landlords are asking. Each column shows the range of rent and number of properties in that range")
    st.plotly_chart(fig)

else:
    st.markdown("## Data not available for the Building/Area")

# footer = """
# <style>
# .footer {
#     position: fixed;
#     left: 0;
#     bottom: 0;
#     width: 100%;
#     background-color: transparent;
#     color: gray;
#     text-align: center;
#     padding: 10px;
#     font-size: 16px;
# }
# </style>
# <div class="footer">
#     <p>Made with ❤️ by <a href="https://yourwebsite.com" target="_blank">Purushottam Deshpande</a></p>
# </div>
# """

# st.markdown(footer, unsafe_allow_html=True)