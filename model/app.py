import streamlit as st
import pickle 
import numpy as np
import pandas as pd
import plotly.graph_objects as go
#from main import get_clean_data



def add_sidebar():
    st.sidebar.header('Cell Nuclei Details')

    data = get_clean_data()
    
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]
    
    input_from_user = {}
    for label, key in slider_labels:
        input_from_user[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    return input_from_user


def get_clean_data():
    '''
    This function cleans and loads the data
    '''
    data= pd.read_csv('data.csv')
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    #print(len(data.columns))
    
    return data


def get_scaled_values(input_data):
    data = get_clean_data()
    X = data.drop(['diagnosis'], axis=1)

    scaled_dict = {}

    for key, value in input_data.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value
    
    #print(scaled_dict)
    return scaled_dict
  

def get_radar_chart(input_data):

    input_data = get_scaled_values(input_data)
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                'Smoothness', 'Compactness', 
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']
    
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
            r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
            ],
            theta=categories,
            fill='toself',
            name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
            r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
            input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']
            ],
            theta=categories,
            fill='toself',
            name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
            r=[
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
            ],
            theta=categories,
            fill='toself',
            name='Worst Value'
    ))

    fig.update_layout(
        polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
        showlegend=True
    )
    
    return fig


def get_predictions(input_data): 
    
    with open('model/saved_files.pkl', 'rb') as f: #saved all together
        files = pickle.load(f)
    model=files['model']
    scaler=files['scaler']

    input_array = np.array(list(input_data.values())).reshape(1,-1)

    input_array_scaled= scaler.transform(input_array)

    predictions= model.predict(input_array_scaled)

    st.subheader('Cell Cluster Prediction')
    st.write('The prediction for the cell with the attributes you inputed is:')

    if predictions[0]==0:
        st.write("<span class = 'diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class = 'diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)

    st.write('Probability of being Benign:', model.predict_proba(input_array_scaled)[0][0].round(2))
    st.write('Probability of being Malicious:', model.predict_proba(input_array_scaled)[0][1].round(2))




def main():
    st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon=":female-doctor:",
    layout="wide",
    initial_sidebar_state="expanded"
    )

    # with open('assets/style.css') as f:
    #     st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

    input_from_user = add_sidebar()
    
    with st.container():
        st.title('Breast Cancer Demo for Front End')
        st.write('Hi my fellow front-end devs, This is a demo of how I want the front end to look like, kindly preview this')

    
    col1, col2 = st.columns([4,1])
    with col1:
        radar_chat = get_radar_chart(input_from_user)
        st.plotly_chart(radar_chat)

    with col2:
        get_predictions(input_from_user)



if __name__ == '__main__':
    main()