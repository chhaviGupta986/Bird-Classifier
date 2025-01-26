# Bird Species Classification System

Birds play crucial roles in ecosystems, supporting processes like pollination, seed dispersal, and pest control. Traditional bird monitoring methods, such as visual observation and manual sound analysis, are time-intensive and prone to errors. Automated bird sound classification using deep learning offers faster and more accurate biodiversity monitoring. [Visit Website](https://ai-bird-classifier.streamlit.app/)

## Scope

- Classifying 114 bird species from recorded audio files.
- Classifying 525 bird species from image files.
- Using Mel Frequency Cepstral Coefficients (MFCC) as features for the model.
- Developing a CNN model for bird species identification using bird sound.
- Providing a user-friendly application for real-time classification of bird sounds and bird images.
- Providing options for users to learn more about the identified bird species or ask questions about it.

## Target Audience
- Birdwatchers, wildlife enthusiasts, and researchers interested in automatic bird sound classification.

## Model Architecture and Results

The Model for Bird Species classification using Sound is built using TensorFlow's Keras API. The architecture consists of a Convolutional Neural Network (CNN) with the following layers:

- **Convolutional Layers**: Extract features from audio and image inputs.
- **Pooling Layers**: Reduce spatial dimensions for efficient processing.
- **Dense Layers**: Perform classification of bird species based on extracted features.

It is trained on following dataset: https://www.kaggle.com/c/birdsong-recognition, with a **79.68% accuracy** in classifying bird sounds.

Model for Bird Species classification using Image has been taken from hugging face: https://huggingface.co/chriamue/bird-species-classifier, and has an accuracy of 96.8%.

## Future Work

- Explore more advanced techniques for feature extraction to improve model accuracy.
- Integrate audio augmentation strategies to increase the diversity of the training dataset and enhance model robustness.

## How to Run project:

1. Clone the repository:  
   `git clone https://github.com/chhaviGupta986/Bird-Classifier.git`
   
2. Install the necessary dependencies:  
   `pip install -r requirements.txt`
   
3. Launch the Streamlit app:  
   `streamlit run app.py`
   
4. Open the app in your browser at:  
   `http://localhost:8501`
