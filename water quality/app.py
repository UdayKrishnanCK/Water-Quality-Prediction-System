import streamlit as st
import pickle
from PIL import Image

def main():
    st.set_page_config(page_title="Water Quality Prediction", page_icon=":droplet:")
    st.title(":rainbow[WATER QUALITY PREDICTION]")
    image = Image.open("Sensor-water-1-1024x576.jpg")
    st.image(image,width=1000,use_column_width=True)

    model = pickle.load(open('model.sav', 'rb'))
    scaler = pickle.load(open('scaler.sav','rb'))


    features = ['Iron','Nitrate','Chloride','Lead','Zinc','Turbidity','Fluoride','Copper','Odor','Sulfate','Chlorine','Manganese','TotalDissolvedSolids','Source','WaterTemperature']
    input_data = {}
    for feature in features:
        input_data[feature] = st.text_input(feature, "")

    model = pickle.load(open('model.sav', 'rb'))
    scaler = pickle.load(open('scaler.sav', 'rb'))
    pred = st.button('Predict')

    if pred:
        input_features = [float(input_data[feature]) for feature in features]
        predictions = model.predict(scaler.transform([input_features]))
        if predictions == 0:
            st.error("The water is not pure.")
        else:
            st.success("The water is pure.")

    image3 = Image.open("Water-Sampling-scaled.jpg")
    st.image(image3, width=1000, use_column_width=True)


main()