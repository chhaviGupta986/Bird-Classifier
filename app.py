import os
import json
import librosa
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
# from streamlit_extras.add_vertical_space import add_vertical_space
from warnings import filterwarnings
filterwarnings('ignore')
import requests
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image
from google.generativeai import configure, GenerativeModel
import google.generativeai as genai

# Configure API key
genai.configure(api_key='AIzaSyCadi6j6_olpoUrlqRH41wFYqEpNcIn3Vo')  # Store securely, preferably as environment variable

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
    st.set_page_config(page_title='Bird Classifier', layout='centered', initial_sidebar_state='expanded')


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
    # # add_vertical_space(4)
    # # Display content in the sidebar
    # st.sidebar.header("TEAM MEMBERS:")
    team_members = [
        "Viraj Yadav, 16010121216",
        "Chhavi Gupta, 16010121218",
        "Zarwaan Shroff, 16010121220",
        "Varrshinie Aravindan, 16010121221",
        "Siddhi Raman,  16010121222"
    ]
    # for member in team_members:
    #     st.sidebar.write(member)
    # for member in team_members:
    #     st.sidebar.markdown(f'<p class="sidebar-team-members">{member}</p>', unsafe_allow_html=True)

    # List of bird species
    # bird_species = [
    #     'Tataupa Tinam', 'Southern Brown Kiwi', 'Berlepschs Tinam', 'Rusty Tinam', 'Chilean Tinam',
    #     'Philippine Megapode', 'Black-fronted Piping Gua', 'Quebracho Crested Tinam', 'White-browed Gua',
    #     'Baudo Gua', 'Greater Rhea', 'Grey Tinam', 'Great Spotted Kiwi', 'Choco Tinam', 'White-winged Gua',
    #     'Rusty-margined Gua', 'Malleefowl', 'New Guinea Scrubfowl', 'Lesser Rhea', 'Barred Tinam',
    #     'Colombian Chachalaca', 'Chaco Chachalaca', 'Cauca Gua', 'Dwarf Tinam', 'Thicket Tinam',
    #     'Melanesian Megapode', 'Brown Tinam', 'Darwins Nothura', 'Little Tinam', 'Brushland Tinam',
    #     'Moluccan Megapode', 'Red-throated Piping Gua', 'Red-billed Brushturkey', 'Orange-footed Scrubfowl',
    #     'Great Tinam', 'Elegant Crested Tinam', 'Little Chachalaca', 'Lesser Nothura', 'Grey-headed Chachalaca',
    #     'Puna Tinam', 'Tawny-breasted Tinam', 'Nicobar Megapode', 'Black Tinam', 'Bearded Gua', 'Band-tailed Gua',
    #     'White-bellied Nothura', 'Collared Brushturkey', 'Australian Brushturkey', 'White-bellied Chachalaca',
    #     'Wattled Brushturkey', 'Spotted Nothura', 'White-crested Gua', 'Male', 'Blue-throated Piping Gua',
    #     'Rufous-vented Chachalaca', 'Em', 'Chestnut-winged Chachalaca', 'Taczanowskis Tinam', 'Pale-browed Tinam',
    #     'Tepui Tinam', 'Spixs Gua', 'Dusky-legged Gua', 'Curve-billed Tinam', 'Micronesian Megapode',
    #     'Huayco Tinam', 'Buff-browed Chachalaca', 'Red-winged Tinam', 'Black-billed Brushturkey',
    #     'Little Spotted Kiwi', 'Marail Gua', 'Andean Gua', 'Rufous-bellied Chachalaca', 'Okarito Kiwi',
    #     'Andean Tinam', 'Rufous-headed Chachalaca', 'Scaled Chachalaca', 'Tongan Megapode', 'Yellow-legged Tinam',
    #     'Slaty-breasted Tinam', 'Ornate Tinam', 'Speckled Chachalaca', 'Bartletts Tinam', 'Sula Megapode',
    #     'Cinereous Tinam', 'Hooded Tinam', 'Southern Cassowary', 'Variegated Tinam', 'Chestnut-bellied Gua',
    #     'Dusky Megapode', 'Tanimbar Megapode', 'Vanuatu Megapode', 'Black-capped Tinam', 'Brazilian Tinam',
    #     'Red-legged Tinam', 'Undulated Tinam', 'Dwarf Cassowary', 'Solitary Tinam', 'Plain Chachalaca',
    #     'Northern Cassowary', 'Patagonian Tinam', 'White-throated Tinam', 'Common Ostrich', 'Red-faced Gua',
    #     'Biak Scrubfowl', 'Highland Tinam', 'Grey-legged Tinam', 'West Mexican Chachalaca', 'Somali Ostrich',
    #     'Small-billed Tinam', 'East Brazilian Chachalaca', 'Chestnut-headed Chachalaca',
    #     'North Island Brown Kiwi', 'Crested Gua', 'Trinidad Piping Gua'
    # ]

    # # Sidebar header
    # st.sidebar.header("Bird Species Identifiable by Model")

    # # Add a collapsible section for bird species
    # with st.sidebar.expander("View Bird Species"):
    #     for species in bird_species:
    #         st.write(species)
    add_vertical_space(2)
    # Display content in the sidebar
    # add_vertical_space(2)
    st.sidebar.header("")
    # st.sidebar.header("")
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
    # _,col2,_  = st.columns([0.1,0.9,0.1])
    # with col2:
    predicted_class = predicted_class.replace('_sound', '')
    st.markdown(f'<h2 style="text-align: center; color: #99ccff;">Identified Species: {predicted_class}</h2>', 
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

    # Display Bird Classifier (Hugging Face)
    # image = st.file_uploader("Upload a Bird Image", type=["jpg", "jpeg", "png"])

    # if image is not None:
    #     # Process the image with the Hugging Face model
    #     img = processor(images=image, return_tensors="pt").pixel_values
    #     outputs = model(img)
    #     predicted_label = model.config.id2label[torch.argmax(outputs.logits).item()]
    #     st.image(image, caption="Uploaded Bird Image", use_container_width=True)
    #     st.markdown(f"**Predicted Bird Species:** {predicted_label}")

    # st.markdown(f'<h2 style="text-align: center; color: white;">{predicted_class}</h2>',
    #             unsafe_allow_html=True)

    # bird_info = get_bird_info_and_image(predicted_class)

    # if bird_info:
    #     st.markdown(f"[Learn more about {predicted_class}]({bird_info['url']})")
    #     st.image(bird_info['image'], caption=f"{bird_info['title']} - {bird_info['description']}", use_container_width=True)
    # # return confidence
    # _,col2,_  = st.columns([0.1,0.9,0.1])
    # with col2:
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
            # st.markdown("### Ask questions about this bird")
            # st.markdown("#### Ask questions about this bird:")
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

# _,col2,_  = st.columns([0.1,0.9,0.1])
# with col2:
#     input_audio = st.file_uploader(label='Upload the Audio', type=['mp3', 'wav'])

# if input_audio is not None:

#     _,col2,_ = st.columns([0.2,0.8,0.2])
#     with col2:
#         prediction(input_audio)

# Add radio button for selection
classification_type = st.radio("Select Classification Type", ["Audio", "Image"], horizontal=True)

if classification_type == "Audio":
    input_audio = st.file_uploader(label='Upload the Audio', type=['mp3', 'wav'])
    if input_audio is not None:
        st.audio(input_audio, format='audio/wav')  # Streamlit supports wav, mp3, and ogg formats
        
        # # Optional: Save the file locally (if needed for processing)
        # with open("uploaded_audio.wav", "wb") as f:
        #     f.write(input_audio.getbuffer())
        # _, col2, _ = st.columns([0.1,0.9,0.1])
        # with col2:
        prediction(input_audio)

else:  # Image classification
    input_image = st.file_uploader(label='Upload the Image', type=['jpg', 'jpeg'])
    if input_image is not None:
        # Process image prediction
        image = Image.open(input_image)
        inputs = processor(image, return_tensors="pt")
        outputs = model(**inputs)
        predicted_class = model.config.id2label[outputs.logits.argmax().item()]
        predicted_class=predicted_class.capitalize()
        # Display results similar to audio
        
        _,col2,_  = st.columns([0.1,0.9,0.1])
        with col2:
            st.image(image)
            st.markdown(f'<h2 style="text-align: center; color: #99ccff;">Identified as: {predicted_class}</h2>', 
                        unsafe_allow_html=True)
        # st.markdown(f'<h2 style="text-align: center; color: #99ccff;">Identified Species of Bird: {predicted_class}</h2>', 
        #                 unsafe_allow_html=True)
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
                # st.markdown("### Ask questions about this bird")
                # st.markdown("#### Ask questions about this bird:")
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