import json
import librosa
import numpy as np
import tensorflow as tf
import streamlit as st
from warnings import filterwarnings
filterwarnings('ignore')
import requests
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
from google.generativeai import GenerativeModel
import google.generativeai as genai
import torch.nn.functional as F  # For the softmax function
# Configure API key
genai.configure(api_key='AIzaSyCadi6j6_olpoUrlqRH41wFYqEpNcIn3Vo')  
# Store securely, preferably as environment variable
# replace with your own 

def get_gemini_response(prompt):
    model = GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response.text

# New Code for Bird Image Classifier (Hugging Face Model)
# Load model and processor
processor = AutoImageProcessor.from_pretrained("chriamue/bird-species-classifier")
model = AutoModelForImageClassification.from_pretrained("chriamue/bird-species-classifier")


def get_bird_info_and_image(bird_name):
    # Format the bird name for Wikipedia API
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{bird_name.replace(' ', '_')}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        return {
            "title": data.get("title"),
            "description": data.get("extract", "No description available."),
            "image": data.get("thumbnail", {}).get("source", None),
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", None)
        }
    else:
        # Handle errors (e.g., page not found)
        print(f"Error: Could not fetch data for {bird_name}")
        return None


def streamlit_config():

    # page configuration
    st.set_page_config(page_title='Bird Classifier', page_icon="favicon1.ico", layout='centered', initial_sidebar_state='expanded')

    page_background_color = """
    <style>
    /* Set background color for the entire page */
    body {
        background-color: #f0f8ff; /* Light blue */
    }

    /* Remove background color of the header */
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0); /* Transparent header */
    }

    /* Sidebar background color */
    [data-testid="stSidebar"] {
        background-color: #1e1e2f; /* White sidebar */
        # background-color: #66004d; /* purpel */
    }

    /* Title Styling */
    h1 {
        font-family: 'Helvetica', sans-serif;
        font-size: 40px;
        font-weight: bold;
        color: #1e3d58; /* Dark blue color for title */
    }

    /* Customize the team member text in the sidebar */
    .sidebar-team-members p {
        font-family: 'Arial', sans-serif;
        font-size: 16px;
        color: #white; 
        font-weight: light;
    }

    </style>
    """
    
    st.markdown(page_background_color, unsafe_allow_html=True)

    # title and position
    st.markdown(f'<h1 style="text-align: center;">Identify Bird Species from Sound or Image</h1>',
                unsafe_allow_html=True)

    team_members = [
        "Viraj Yadav, 16010121216",
        "Chhavi Gupta, 16010121218",
        "Zarwaan Shroff, 16010121220",
        "Varrshinie Aravindan, 16010121221",
        "Siddhi Raman,  16010121222"
    ]
    add_vertical_space(2)

    st.sidebar.header("")
    st.sidebar.header("Team Members:")
    for member in team_members:
        st.sidebar.markdown(f'<p class="sidebar-team-members">{member}</p>', unsafe_allow_html=True)

# Function to add vertical space
def add_vertical_space(space_count):
    for _ in range(space_count):
        st.markdown("<br>", unsafe_allow_html=True)
# Streamlit Configuration Setup
streamlit_config()



def prediction(audio_file):

    # Load the Prediction JSON File to Predict Target_Label
    with open('prediction.json', mode='r') as f:
        prediction_dict = json.load(f)

    # Extract the Audio_Signal and Sample_Rate from Input Audio
    audio, sample_rate =librosa.load(audio_file)

    # Extract the MFCC Features and Aggrigate
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_features = np.mean(mfccs_features, axis=1)

    # Reshape MFCC features to match the expected input shape for Conv1D both batch & feature dimension
    mfccs_features = np.expand_dims(mfccs_features, axis=0)
    mfccs_features = np.expand_dims(mfccs_features, axis=2)

    # Convert into Tensors
    mfccs_tensors = tf.convert_to_tensor(mfccs_features, dtype=tf.float32)

    # Load the Model and Prediction
    model = tf.keras.models.load_model('model.h5')
    prediction = model.predict(mfccs_tensors)

    # Find the Maximum Probability Value
    target_label = np.argmax(prediction)

    # Find the Target_Label Name using Prediction_dict
    predicted_class = prediction_dict[str(target_label)]
    temp=np.max(prediction)*100

    confidence = round(temp, 2)
    
    add_vertical_space(1)
    st.markdown(f'<h4 style="text-align: center; color: white;">{confidence:.2f}% Match Found</h4>', 
                    unsafe_allow_html=True)

    predicted_class = predicted_class.replace('_sound', '')
    st.markdown(f'<h2 style="text-align: center; color: #99ccff;">Identified as: {predicted_class}</h2>', 
                    unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        div.stButton > button {
            display: block;
            margin: 0 auto;
        }
        </style>
        """, unsafe_allow_html=True)

    # Initialize session state for showing Q&A form
    if 'show_qa_audio' not in st.session_state:
        st.session_state.show_qa_audio = False

    if st.button(f"Learn more about {predicted_class}"):
        _,col2,_  = st.columns([0.1,0.9,0.1])
        with col2:
            bird_data = get_bird_info_and_image(predicted_class)
            add_vertical_space(1)
            if bird_data:
                st.markdown(f"""
                <div style="background-color: #1e1e2f; border: 1px solid #99ccff; padding: 20px; border-radius: 8px; ">
                    <h3>{bird_data['title']}</h3>
                    <p><strong>Description:</strong> {bird_data['description']}</p>
                    <p><a href="{bird_data['url']}">More Info on Wikipedia</a></p>
                    <img src="{bird_data['image']}" alt="Bird Image" style="max-width: 100%; border-radius: 8px;">
                </div>
                """, unsafe_allow_html=True)
            else:
                st.write("Sorry, no information found for this bird.")
            add_vertical_space(2)

    if st.button(f"Ask questions about {predicted_class}"):
        st.session_state.show_qa_audio = True

    # Only show form if Ask button was clicked
    if st.session_state.show_qa_audio:
        _,col2,_  = st.columns([0.1,0.9,0.1])
        with col2:
            add_vertical_space(1)
            with st.form(key='qa_form'):
                st.markdown("#### Ask questions about this bird:")
                question = st.text_input("")
                # question = st.text_input("Enter question")
                submit_button = st.form_submit_button("Ask")
                
                if submit_button and question:
                    with st.spinner("Generating response..."):
                        prompt = f"""                                                      
                                Please answer this question: {question} about the {predicted_class} bird: 
                                
                                Provide a clear and concise answer"""
                                
                        response = get_gemini_response(prompt)
                        st.write("Answer:\n", response)

classification_type = st.radio("Select Classification Type", ["Audio", "Image"], horizontal=True)

if classification_type == "Audio":
    input_audio = st.file_uploader(label='Upload the Audio', type=['mp3', 'wav'])
    if input_audio is not None:
        st.audio(input_audio, format='audio/wav')  # Streamlit supports wav, mp3, and ogg formats
        prediction(input_audio)

else:  # Image classification
    input_image = st.file_uploader(label='Upload the Image', type=['jpg', 'jpeg'])
    if input_image is not None:
        # Process image prediction
        image = Image.open(input_image)
        inputs = processor(image, return_tensors="pt")
        outputs = model(**inputs)

        # Get the predicted class
        predicted_class_idx = outputs.logits.argmax().item()  # Get the index of the highest logit
        predicted_class = model.config.id2label[predicted_class_idx]
        predicted_class = predicted_class.capitalize()

        # Calculate the confidence
        logits = outputs.logits
        softmax_probs = F.softmax(logits, dim=1)  # Apply softmax to get probabilities
        confidence = softmax_probs[0][predicted_class_idx].item()  # Get the probability of the predicted class

        # Convert confidence to percentage
        confidence_percentage = confidence * 100

        st.image(image)
        add_vertical_space(1)
        st.markdown(f'<h4 style="text-align: center; color: white;">{confidence_percentage:.2f}% Match Found</h4>', 
                        unsafe_allow_html=True)        
        st.markdown(f'<h2 style="text-align: center; color: #99ccff;">Identified as: {predicted_class}</h2>', 
                    unsafe_allow_html=True)
        
        _,col2,_  = st.columns([0.1,0.9,0.1])
        with col2:

            st.markdown(
                """
                <style>
                div.stButton > button {
                    display: block;
                    margin: 0 auto;
                }
                </style>
                """, unsafe_allow_html=True)
                    
        # Initialize session state for showing Q&A form
        if 'show_qa_image' not in st.session_state:
            st.session_state.show_qa_image = False

        if st.button(f"Learn more about {predicted_class}"):
            _,col2,_  = st.columns([0.1,0.9,0.1])
            with col2:
                bird_data = get_bird_info_and_image(predicted_class)
                add_vertical_space(1)
                if bird_data:
                    st.markdown(f"""
                    <div style="background-color: #1e1e2f; border: 1px solid #99ccff; padding: 20px; border-radius: 8px; ">
                        <h3>{bird_data['title']}</h3>
                        <p><strong>Description:</strong> {bird_data['description']}</p>
                        <p><a href="{bird_data['url']}">More Info on Wikipedia</a></p>
                        <img src="{bird_data['image']}" alt="Bird Image" style="max-width: 100%; border-radius: 8px;">
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.write("Sorry, no information found for this bird.")
                add_vertical_space(2)

        if st.button(f"Ask questions about {predicted_class}"):
            st.session_state.show_qa_image = True

        # Only show form if Ask button was clicked
        if st.session_state.show_qa_image:
            _,col2,_  = st.columns([0.1,0.9,0.1])
            with col2:
                add_vertical_space(1)
                with st.form(key='qa_form'):
                    st.markdown("#### Ask questions about this bird:")
                    question = st.text_input("")
                    submit_button = st.form_submit_button("Ask")
                    
                    if submit_button and question:
                        with st.spinner("Generating response..."):
                            prompt = f"""                                                      
                                    Please answer this question: {question} about the {predicted_class} bird: 
                                    
                                    Provide a clear and concise answer"""
                                    
                            response = get_gemini_response(prompt)
                            st.write("Answer:\n", response)