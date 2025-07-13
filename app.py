# app.py

import streamlit as st
import numpy as np
import cv2
import joblib
from PIL import Image
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Plant Pathology Classifier | AI-Powered Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Advanced Custom CSS for Professional Plant Pathology Theme ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0f1419 0%, #1a2332 100%);
        color: #ffffff;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a2332 100%);
    }
    
    /* Header Section */
    .hero-section {
        background: linear-gradient(135deg, #1e3a2f 0%, #2d5a43 50%, #1e3a2f 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(46, 160, 67, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(46, 160, 67, 0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.05); opacity: 0.8; }
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 1rem;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        position: relative;
        z-index: 1;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: #b8e6c4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    .hero-stats {
        display: flex;
        justify-content: center;
        gap: 3rem;
        margin-top: 2rem;
        position: relative;
        z-index: 1;
    }
    
    .stat-item {
        text-align: center;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(46, 160, 67, 0.3);
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #2ea043;
        display: block;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #b8e6c4;
        margin-top: 0.5rem;
    }
    
    /* Upload Section */
    .upload-section {
        background: linear-gradient(135deg, #1a2332 0%, #2a3441 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin: 2rem 0;
        border: 1px solid rgba(46, 160, 67, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .upload-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #2ea043 0%, #4ade80 50%, #2ea043 100%);
    }
    
    .section-title {
        font-size: 2rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 1rem;
        text-align: center;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
    }
    
    .upload-instructions {
        background: rgba(46, 160, 67, 0.1);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid #2ea043;
        color: #b8e6c4;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* File Uploader Styling */
    .stFileUploader > div {
        background: linear-gradient(135deg, #2a3441 0%, #3a4651 100%);
        border: 2px dashed #2ea043;
        border-radius: 15px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: #4ade80;
        background: linear-gradient(135deg, #2a3441 0%, #3a4651 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(46, 160, 67, 0.2);
    }
    
    /* Results Section */
    .results-container {
        background: linear-gradient(135deg, #1a2332 0%, #2a3441 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid rgba(46, 160, 67, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #2a3441 0%, #3a4651 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(46, 160, 67, 0.3);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #2ea043 0%, #4ade80 100%);
    }
    
    .probability-bar {
        background: rgba(46, 160, 67, 0.1);
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
        border-left: 4px solid #2ea043;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: all 0.3s ease;
    }
    
    .probability-bar:hover {
        background: rgba(46, 160, 67, 0.2);
        transform: translateX(5px);
    }
    
    .disease-name {
        font-weight: 600;
        color: #ffffff;
        font-size: 1.1rem;
        text-transform: capitalize;
    }
    
    .probability-value {
        font-weight: 700;
        color: #2ea043;
        font-size: 1.1rem;
    }
    
    .final-prediction {
        background: linear-gradient(135deg, #2ea043 0%, #4ade80 100%);
        color: #ffffff;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 1.5rem;
        box-shadow: 0 4px 20px rgba(46, 160, 67, 0.3);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #0f1419 0%, #1a2332 100%);
    }
    
    .sidebar-content {
        background: linear-gradient(135deg, #1a2332 0%, #2a3441 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid rgba(46, 160, 67, 0.2);
    }
    
    .sidebar-title {
        color: #2ea043;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .sidebar-text {
        color: #b8e6c4;
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    
    .feature-list {
        list-style: none;
        padding: 0;
    }
    
    .feature-item {
        background: rgba(46, 160, 67, 0.1);
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        border-left: 3px solid #2ea043;
        color: #ffffff;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #2ea043 0%, #4ade80 100%);
        color: #ffffff;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(46, 160, 67, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #1e7e34 0%, #2ea043 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(46, 160, 67, 0.4);
    }
    
    /* Info Boxes */
    .info-box {
        background: linear-gradient(135deg, #1a2332 0%, #2a3441 100%);
        border: 1px solid rgba(46, 160, 67, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #b8e6c4;
        line-height: 1.6;
    }
    
    .info-box h3 {
        color: #2ea043;
        font-size: 1.3rem;
        margin-bottom: 1rem;
    }
    
    /* Loading Animation */
    .loading-animation {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    .loader {
        border: 4px solid rgba(46, 160, 67, 0.3);
        border-top: 4px solid #2ea043;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        
        .hero-stats {
            flex-direction: column;
            gap: 1rem;
        }
        
        .section-title {
            font-size: 1.5rem;
        }
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Enhanced Sidebar ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2909/2909765.png", width=80)
st.sidebar.markdown(
    """
    <h1 style='color:#22c55e; text-align:center; margin-bottom:0.5rem;'>Plant Pathology AI</h1>
    <div style='background:#23243a; border-radius:12px; padding:1.2rem; color:#e5e7eb; margin-top:1.2rem;'>
        <div style='margin-bottom:1.2rem;'>
            <strong>🧪 Research-Grade Classification</strong><br>
            Advanced machine learning model trained on the Plant Pathology 2020 dataset.
        </div>
        <div style='margin-bottom:1.2rem;'>
            <h3 style='margin-bottom:0.5rem;'>🎯 Key Features</h3>
            <ul style='padding-left:1.2rem; margin:0;'>
                <li>🔬 Multi-class disease classification</li>
                <li>🚀 Instant predictions</li>
                <li>📊 Confidence scores</li>
                <li>🌿 Handcrafted features</li>
            </ul>
        </div>
        <div>
            <h3 style='margin-bottom:0.5rem;'>📝 Instructions</h3>
            <ol style='padding-left:1.2rem; margin:0;'>
                <li><b>Upload</b> a clear leaf image (JPG/PNG)</li>
                <li><b>Wait</b> for feature extraction & prediction</li>
                <li><b>View</b> the predicted probabilities</li>
            </ol>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Hero Section ---
st.markdown(
    """
    <div class="hero-section">
        <div class="hero-title">🌱 Plant Pathology AI Classifier</div>
        <div class="hero-subtitle">
            Advanced AI-powered disease detection for apple leaves using traditional machine learning
        </div>
        <div class="hero-stats">
            <div class="stat-item">
                <span class="stat-number">4</span>
                <span class="stat-label">Disease Classes</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Model and Feature Constants ---
MODEL_PATH = "./models/random_forest_model.joblib"
SCALER_PATH = "./models/feature_scaler.joblib"
IMG_WIDTH = 128
IMG_HEIGHT = 128
CHANNELS = 3
HIST_BINS = 32
target_cols = ['healthy', 'multiple_diseases', 'rust', 'scab']

# Disease descriptions for better user understanding
disease_descriptions = {
    'healthy': 'No disease detected - leaf appears healthy',
    'multiple_diseases': 'Multiple diseases present on the leaf',
    'rust': 'Fungal disease causing orange/brown spots',
    'scab': 'Fungal disease causing dark, scaly lesions'
}

# --- Load Model and Scaler ---
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler, None
    except Exception as e:
        return None, None, str(e)

# --- Feature Extraction Function ---
def extract_features_for_inference(image, bins=HIST_BINS):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    hist_features = []
    for i in range(CHANNELS):
        hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hist_features.extend(hist)
    return np.array(hist_features).reshape(1, -1)

# --- Upload Section ---
st.markdown(
    """
    <div class="upload-section">
        <div class="section-title">📸 Upload Leaf Image for Analysis</div>
        <div class="upload-instructions">
            <strong>🔍 For best results:</strong> Upload a clear, well-lit image of an apple leaf. 
            The model works best with images showing the full leaf surface with good lighting and focus.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a leaf image...", 
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image of an apple leaf for disease classification"
)

if uploaded_file is not None:
    # Create two columns for image and results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="📸 Uploaded Leaf Image", use_column_width=True)
    
    with col2:
        st.markdown(
            """
            <div class="results-container">
                <div class="section-title">🔬 Analysis Results</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Processing animation
        with st.spinner("🔄 Extracting features and analyzing..."):
            # Extract features
            features = extract_features_for_inference(image)
            
            # Load model
            model, scaler, model_error = load_model_and_scaler()
            
            if model_error:
                st.error(f"❌ Error loading model: {model_error}")
            else:
                # Make predictions
                scaled_features = scaler.transform(features)
                predictions_proba_list = model.predict_proba(scaled_features)
                probabilities = np.array([proba[:, 1] for proba in predictions_proba_list]).T[0]
                
                # Display results
                st.markdown(
                    """
                    <div class="prediction-card">
                        <div class="section-title">📊 Disease Probability Scores</div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Sort by probability for better visualization
                sorted_indices = np.argsort(probabilities)[::-1]
                
                for idx in sorted_indices:
                    prob = probabilities[idx]
                    disease = target_cols[idx]
                    description = disease_descriptions[disease]
                    
                    # Color based on probability
                    if prob > 0.7:
                        color = "#2ea043"
                    elif prob > 0.5:
                        color = "#ffa500"
                    else:
                        color = "#6c757d"
                    
                    st.markdown(
                        f"""
                        <div class="probability-bar">
                            <div>
                                <div class="disease-name">{disease.replace('_', ' ').title()}</div>
                                <div style="color: #b8e6c4; font-size: 0.9rem; margin-top: 0.2rem;">
                                    {description}
                                </div>
                            </div>
                            <div class="probability-value" style="color: {color};">
                                {prob:.2%}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                # Final prediction
                predicted_classes = [target_cols[i] for i, prob in enumerate(probabilities) if prob > 0.5]
                
                if not predicted_classes:
                    most_likely_class_idx = np.argmax(probabilities)
                    most_likely_class = target_cols[most_likely_class_idx]
                    confidence = probabilities[most_likely_class_idx]
                    
                    st.markdown(
                        f"""
                        <div class="final-prediction">
                            🎯 Most Likely Diagnosis: <strong>{most_likely_class.replace('_', ' ').title()}</strong>
                            <br>
                            <span style="font-size: 1.1rem;">Confidence: {confidence:.2%}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    predictions_text = ", ".join([cls.replace('_', ' ').title() for cls in predicted_classes])
                    st.markdown(
                        f"""
                        <div class="final-prediction">
                            ⚠️ High Confidence Predictions: <strong>{predictions_text}</strong>
                            <br>
                            <span style="font-size: 1rem;">Multiple conditions detected above 50% threshold</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown(
        """
        <div class="upload-section">
            <div style="text-align: center; padding: 3rem; color: #b8e6c4;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">📤</div>
                <div style="font-size: 1.2rem; margin-bottom: 0.5rem;">No image uploaded yet</div>
                <div style="font-size: 1rem; opacity: 0.8;">Please upload a JPG or PNG image of an apple leaf to begin analysis</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- About/Info Carousel Section ---
st.markdown("<br><hr>", unsafe_allow_html=True)

# --- Carousel Slide Data ---
about_slides = {
    "Technical Approach": """
    <h2 style='color:#6ee7b7;'><span style='font-size:1.5rem;'>🧑‍🔬 About This AI Model</span></h2>
    <p style='color:#a7f3d0; font-size:1.1rem;'>
    This application leverages the <b>Plant Pathology 2020 Challenge dataset</b> from Cornell University to provide accurate disease classification for apple leaves. The model uses traditional machine learning techniques with handcrafted color histogram features and Random Forest classification.
    </p>
    <div style='background:#181921; border-radius:12px; padding:1.5rem; color:#e5e7eb; margin-top:1rem;'>
        <h3>🎯 Technical Approach</h3>
        <p><b>Feature Extraction:</b> Color histogram features (RGB channels)<br>
        <b>Algorithm:</b> Random Forest Classifier<br>
        <b>Image Processing:</b> OpenCV for preprocessing<br>
        <b>Performance:</b> Optimized for speed and accuracy</p>
    </div>
    """,
    "Disease Categories": """
    <h2 style='color:#6ee7b7;'><span style='font-size:1.5rem;'>🧑‍🔬 About This AI Model</span></h2>
    <p style='color:#a7f3d0; font-size:1.1rem;'>
    This application leverages the <b>Plant Pathology 2020 Challenge dataset</b> from Cornell University to provide accurate disease classification for apple leaves. The model uses traditional machine learning techniques with handcrafted color histogram features and Random Forest classification.
    </p>
    <div style='background:#181921; border-radius:12px; padding:1.5rem; color:#e5e7eb; margin-top:1rem;'>
        <h3>🌿 Disease Categories</h3>
        <p><b>Healthy:</b> No disease symptoms detected<br>
        <b>Rust:</b> Fungal infection with orange/brown spots<br>
        <b>Scab:</b> Dark, scaly lesions on leaf surface<br>
        <b>Multiple Diseases:</b> Combination of different conditions</p>
    </div>
    """,
    "Best Practices": """
    <h2 style='color:#6ee7b7;'><span style='font-size:1.5rem;'>🧑‍🔬 About This AI Model</span></h2>
    <p style='color:#a7f3d0; font-size:1.1rem;'>
    This application leverages the <b>Plant Pathology 2020 Challenge dataset</b> from Cornell University to provide accurate disease classification for apple leaves. The model uses traditional machine learning techniques with handcrafted color histogram features and Random Forest classification.
    </p>
    <div style='background:#181921; border-radius:12px; padding:1.5rem; color:#e5e7eb; margin-top:1rem;'>
        <h3>💡 Best Practices</h3>
        <p>For optimal results, ensure your images are:</p>
        <ul>
            <li>Well-lit with natural lighting</li>
            <li>Focused and clear (avoid blur)</li>
            <li>Showing the full leaf surface</li>
            <li>Taken at a reasonable distance</li>
        </ul>
    </div>
    """
}
slide_keys = list(about_slides.keys())
slide_count = len(slide_keys)
slide_interval = 5  # seconds

# --- Session State for Carousel ---
if "carousel_index" not in st.session_state:
    st.session_state.carousel_index = 0
if "last_slide_time" not in st.session_state:
    st.session_state.last_slide_time = time.time()
if "manual_slide" not in st.session_state:
    st.session_state.manual_slide = False

# --- Manual Navigation Buttons ---
col1, col2, col3 = st.columns([1, 6, 1])
with col1:
    if st.button("⬅️", key="prev_slide"):
        st.session_state.carousel_index = (st.session_state.carousel_index - 1) % slide_count
        st.session_state.manual_slide = True
        st.session_state.last_slide_time = time.time()
with col3:
    if st.button("➡️", key="next_slide"):
        st.session_state.carousel_index = (st.session_state.carousel_index + 1) % slide_count
        st.session_state.manual_slide = True
        st.session_state.last_slide_time = time.time()

# --- Manual Slide Selection (Radio) ---
selected_slide = st.radio(
    "Select Info Slide:",
    slide_keys,
    index=st.session_state.carousel_index,
    horizontal=True,
    key="about_slide_selector"
)

# --- Detect Manual Change ---
if selected_slide != slide_keys[st.session_state.carousel_index]:
    st.session_state.carousel_index = slide_keys.index(selected_slide)
    st.session_state.manual_slide = True
    st.session_state.last_slide_time = time.time()
else:
    # Auto-advance if enough time has passed and not manually paused
    if not st.session_state.manual_slide and (time.time() - st.session_state.last_slide_time > slide_interval):
        st.session_state.carousel_index = (st.session_state.carousel_index + 1) % slide_count
        st.session_state.last_slide_time = time.time()
        st.experimental_rerun()
    # Resume auto-advance after 30 seconds of inactivity
    elif st.session_state.manual_slide and (time.time() - st.session_state.last_slide_time > 30):
        st.session_state.manual_slide = False
        st.session_state.last_slide_time = time.time()

# --- JavaScript Timer for Auto-Advance ---
st.markdown(
    f"""
    <script>
    setTimeout(function() {{
        window.parent.postMessage({{streamlitMessageType: "streamlit:rerun"}}, "*");
    }}, {slide_interval * 1000});
    </script>
    """,
    unsafe_allow_html=True
)

# --- Show Slide ---
st.markdown(about_slides[slide_keys[st.session_state.carousel_index]], unsafe_allow_html=True)

# --- Contact Info Section ---
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='background:#23243a; border-radius:12px; padding:1.2rem; color:#e5e7eb; margin-top:2rem; text-align:center;'>
        <h3 style='color:#6ee7b7; margin-bottom:1rem;'>Contact the Developer</h3>
        <a href='https://www.linkedin.com/in/ankit-kumar-gupta-6ba724266/' target='_blank' style='margin-right:18px; text-decoration:none;'>
            <img src='https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/linkedin.svg' width='28' style='vertical-align:middle; margin-right:6px; filter:invert(60%) sepia(80%) saturate(400%) hue-rotate(120deg);'>
            <span style='color:#a7f3d0; font-size:1.1rem; vertical-align:middle;'>LinkedIn</span>
        </a>
        <a href='mailto:ankitkumargupta030204@gmail.com' style='margin-right:18px; text-decoration:none;'>
            <img src='https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/gmail.svg' width='28' style='vertical-align:middle; margin-right:6px; filter:invert(60%) sepia(80%) saturate(400%) hue-rotate(0deg);'>
            <span style='color:#a7f3d0; font-size:1.1rem; vertical-align:middle;'>ankitkumargupta030204@gmail.com</span>
        </a>
        <a href='https://github.com/Ankit-03G' target='_blank' style='text-decoration:none;'>
            <img src='https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/github.svg' width='28' style='vertical-align:middle; margin-right:6px; filter:invert(60%) sepia(80%) saturate(400%) hue-rotate(220deg);'>
            <span style='color:#a7f3d0; font-size:1.1rem; vertical-align:middle;'>GitHub</span>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)
