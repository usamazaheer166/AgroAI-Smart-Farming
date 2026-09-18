from copyreg import pickle

import streamlit as st
import tensorflow as tf
import numpy as np
from pathlib import Path
import pickle

st.set_page_config(
    page_title="AgroAI – Smart Farming Assistant",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root Palette ── */
:root {
    --green-dark:   #1a3c2e;
    --green-mid:    #2d6a4f;
    --green-light:  #52b788;
    --green-pale:   #b7e4c7;
    --gold:         #e9c46a;
    --cream:        #fefae0;
    --bg:           #0f2318;
    --card:         #162d1f;
    --border:       #2d6a4f44;
    --text-main:    #e8f5e9;
    --text-muted:   #81c784;
}

/* ── Base Reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text-main) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1f14 0%, #1a3c2e 100%) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text-main) !important; }
[data-testid="stSidebar"] .stSelectbox label { color: var(--green-pale) !important; }

/* ── Sidebar Brand ── */
.sidebar-brand {
    text-align: center;
    padding: 1.5rem 0.5rem 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1rem;
}
.sidebar-brand h1 {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    color: var(--green-light) !important;
    margin: 0;
    letter-spacing: -0.5px;
}
.sidebar-brand p {
    font-size: 0.75rem;
    color: var(--text-muted) !important;
    margin: 0.2rem 0 0;
    text-transform: uppercase;
    letter-spacing: 2px;
}

/* ── Hero Banner ── */
.hero {
    background: linear-gradient(135deg, #1a3c2e 0%, #0f2318 50%, #1a3c2e 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, #52b78815 0%, transparent 70%);
    border-radius: 50%;
}
.hero-badge {
    display: inline-block;
    background: #52b78820;
    border: 1px solid var(--green-light);
    border-radius: 100px;
    padding: 0.3rem 1rem;
    font-size: 0.75rem;
    color: var(--green-light) !important;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2rem, 4vw, 3.5rem);
    font-weight: 900;
    line-height: 1.1;
    color: var(--cream) !important;
    margin: 0.5rem 0 1rem;
}
.hero h1 span { color: var(--green-light) !important; }
.hero p {
    font-size: 1.05rem;
    color: var(--text-muted) !important;
    max-width: 600px;
    line-height: 1.7;
}

/* ── Feature Cards ── */
.cards-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1.2rem;
    margin: 1.5rem 0;
}
.feature-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    transition: border-color 0.3s, transform 0.3s;
    cursor: default;
}
.feature-card:hover {
    border-color: var(--green-light);
    transform: translateY(-4px);
}
.feature-card .icon {
    font-size: 2rem;
    margin-bottom: 0.7rem;
}
.feature-card h3 {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    color: var(--cream) !important;
    margin: 0 0 0.4rem;
}
.feature-card p {
    font-size: 0.88rem;
    color: var(--text-muted) !important;
    margin: 0;
    line-height: 1.6;
}

/* ── Steps ── */
.steps-wrap { margin: 1rem 0; }
.step-item {
    display: flex;
    gap: 1rem;
    align-items: flex-start;
    margin-bottom: 1.2rem;
}
.step-num {
    flex-shrink: 0;
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: var(--green-mid);
    border: 2px solid var(--green-light);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.9rem;
    color: var(--cream) !important;
}
.step-content h4 {
    margin: 0 0 0.2rem;
    font-size: 0.95rem;
    color: var(--cream) !important;
}
.step-content p {
    margin: 0;
    font-size: 0.85rem;
    color: var(--text-muted) !important;
}

/* ── Section Headings ── */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    color: var(--cream) !important;
    margin: 0 0 0.3rem;
}
.section-sub {
    font-size: 0.9rem;
    color: var(--text-muted) !important;
    margin-bottom: 1.5rem;
}
.divider {
    height: 2px;
    background: linear-gradient(90deg, var(--green-light), transparent);
    border: none;
    margin: 1.5rem 0;
}

/* ── About Cards ── */
.about-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.8rem;
    margin-bottom: 1.2rem;
}
.about-card h3 {
    font-family: 'Playfair Display', serif;
    color: var(--green-light) !important;
    margin: 0 0 0.7rem;
    font-size: 1.15rem;
}
.about-card p, .about-card li {
    color: var(--text-muted) !important;
    font-size: 0.92rem;
    line-height: 1.7;
}
.stat-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-top: 1rem;
}
.stat-box {
    background: #52b78815;
    border: 1px solid var(--green-light);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    text-align: center;
    flex: 1;
    min-width: 120px;
}
.stat-box .num {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    color: var(--gold) !important;
    line-height: 1;
}
.stat-box .lbl {
    font-size: 0.75rem;
    color: var(--text-muted) !important;
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Upload / Result Box ── */
.result-box {
    background: linear-gradient(135deg, #1a3c2e, #0f2318);
    border: 2px solid var(--green-light);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-top: 1rem;
    text-align: center;
}
.result-box .label {
    font-size: 0.8rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-muted) !important;
    margin-bottom: 0.5rem;
}
.result-box .value {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    color: var(--gold) !important;
}

/* ── Streamlit Widgets Override ── */
.stButton > button {
    background: linear-gradient(135deg, var(--green-mid), var(--green-dark)) !important;
    color: var(--cream) !important;
    border: 1px solid var(--green-light) !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.8rem !important;
    letter-spacing: 0.5px;
    transition: all 0.3s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, var(--green-light), var(--green-mid)) !important;
    color: var(--bg) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px #52b78840 !important;
}
.stFileUploader {
    background: var(--card) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
.stNumberInput > div > div > input {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-main) !important;
    border-radius: 10px !important;
}
.stNumberInput label {
    color: var(--text-muted) !important;
    font-size: 0.9rem !important;
}
.stSuccess {
    background: #52b78820 !important;
    border: 1px solid var(--green-light) !important;
    border-radius: 12px !important;
    color: var(--cream) !important;
}
.stSpinner > div { color: var(--green-light) !important; }

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-main) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--green-mid); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)



def model_prediction(test_image):
    BASE_DIR = Path(__file__).resolve().parent.parent
    model_path = BASE_DIR / "Models" / "trained_model.h5"
    model = tf.keras.models.load_model(model_path)
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    return np.argmax(prediction)


CLASS_NAMES = [
    'Apple — Apple Scab', 'Apple — Black Rot', 'Apple — Cedar Apple Rust', 'Apple — Healthy',
    'Blueberry — Healthy',
    'Cherry — Powdery Mildew', 'Cherry — Healthy',
    'Corn — Cercospora / Gray Leaf Spot', 'Corn — Common Rust', 'Corn — Northern Leaf Blight', 'Corn — Healthy',
    'Grape — Black Rot', 'Grape — Esca (Black Measles)', 'Grape — Leaf Blight', 'Grape — Healthy',
    'Orange — Huanglongbing (Citrus Greening)',
    'Peach — Bacterial Spot', 'Peach — Healthy',
    'Pepper Bell — Bacterial Spot', 'Pepper Bell — Healthy',
    'Potato — Early Blight', 'Potato — Late Blight', 'Potato — Healthy',
    'Raspberry — Healthy', 'Soybean — Healthy',
    'Squash — Powdery Mildew',
    'Strawberry — Leaf Scorch', 'Strawberry — Healthy',
    'Tomato — Bacterial Spot', 'Tomato — Early Blight', 'Tomato — Late Blight',
    'Tomato — Leaf Mold', 'Tomato — Septoria Leaf Spot',
    'Tomato — Spider Mites', 'Tomato — Target Spot',
    'Tomato — Yellow Leaf Curl Virus', 'Tomato — Mosaic Virus', 'Tomato — Healthy',
]



st.sidebar.markdown("""
<div class="sidebar-brand">
    <h1>🌿 AgroAI</h1>
    <p>Smart Farming Assistant</p>
</div>
""", unsafe_allow_html=True)

app_mode = st.sidebar.selectbox(
    "Navigate",
    ["🏠  Home", "📖  About", "🔬  Disease Recognition", "🌾  Crop Recommendation"],
)

st.sidebar.markdown("<hr style='border-color:#2d6a4f44; margin:1.5rem 0'>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style='font-size:0.78rem; color:#81c784; padding:0 0.5rem; line-height:1.8;'>
    <b style='color:#b7e4c7'>Supported Crops</b><br>
    Apple · Blueberry · Cherry<br>
    Corn · Grape · Orange<br>
    Peach · Pepper · Potato<br>
    Raspberry · Soybean · Squash<br>
    Strawberry · Tomato
</div>
""", unsafe_allow_html=True)



if "Home" in app_mode:
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">🌱 AI-Powered Agriculture</div>
        <h1>Protect Your Crops<br>with <span>Smart AI</span></h1>
        <p>Upload a leaf image for instant disease detection, or enter soil & climate data to get the perfect crop recommendation — powered by deep learning.</p>
    </div>
    """, unsafe_allow_html=True)

    
    st.markdown("""
    <div class="cards-grid">
        <div class="feature-card">
            <div class="icon">🔬</div>
            <h3>Disease Recognition</h3>
            <p>Detect 38 plant diseases across 14 crop types instantly from a single leaf photo.</p>
        </div>
        <div class="feature-card">
            <div class="icon">🌾</div>
            <h3>Crop Recommendation</h3>
            <p>Input your soil NPK, temperature, humidity, pH and rainfall to find the ideal crop.</p>
        </div>
        <div class="feature-card">
            <div class="icon">⚡</div>
            <h3>Instant Results</h3>
            <p>Results in seconds — no waiting, no lab visits. Act fast to protect your harvest.</p>
        </div>
        <div class="feature-card">
            <div class="icon">🎯</div>
            <h3>High Accuracy</h3>
            <p>Trained on 87,000+ images with 80/20 train-validation split for robust performance.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<p class='section-title'>How It Works</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <p style='color:#52b788; font-size:0.8rem; letter-spacing:2px; text-transform:uppercase; margin-bottom:0.8rem;'>Disease Detection</p>
        <div class="steps-wrap">
            <div class="step-item">
                <div class="step-num">1</div>
                <div class="step-content">
                    <h4>Upload Leaf Image</h4>
                    <p>Take a clear photo of the affected leaf and upload it on the Disease Recognition page.</p>
                </div>
            </div>
            <div class="step-item">
                <div class="step-num">2</div>
                <div class="step-content">
                    <h4>AI Analysis</h4>
                    <p>Our CNN model processes the image and identifies disease patterns.</p>
                </div>
            </div>
            <div class="step-item">
                <div class="step-num">3</div>
                <div class="step-content">
                    <h4>Get Result</h4>
                    <p>Receive the disease name instantly and take action to protect your crop.</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <p style='color:#52b788; font-size:0.8rem; letter-spacing:2px; text-transform:uppercase; margin-bottom:0.8rem;'>Crop Recommendation</p>
        <div class="steps-wrap">
            <div class="step-item">
                <div class="step-num">1</div>
                <div class="step-content">
                    <h4>Enter Soil Data</h4>
                    <p>Provide Nitrogen, Phosphorus, Potassium levels from your soil test report.</p>
                </div>
            </div>
            <div class="step-item">
                <div class="step-num">2</div>
                <div class="step-content">
                    <h4>Add Climate Info</h4>
                    <p>Enter your area's temperature, humidity, pH, and annual rainfall.</p>
                </div>
            </div>
            <div class="step-item">
                <div class="step-num">3</div>
                <div class="step-content">
                    <h4>Get Best Crop</h4>
                    <p>The model recommends the most suitable crop for maximum yield.</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)



elif "About" in app_mode:
    st.markdown("""
    <div class="hero" style="padding:2rem 2.5rem;">
        <div class="hero-badge">📖 About This Project</div>
        <h1 style="font-size:2.2rem;">AgroAI — <span>Smart Farming</span></h1>
        <p>An end-to-end AI solution combining computer vision and machine learning to assist farmers in disease detection and crop planning.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        <div class="about-card">
            <h3>🔬 Disease Recognition Model</h3>
            <p>
                Our disease detection model is a deep Convolutional Neural Network (CNN) trained on the
                <b style='color:#b7e4c7'>PlantVillage dataset</b>, recreated with offline augmentation to improve
                generalization. It classifies plant leaf images into <b style='color:#b7e4c7'>38 categories</b>
                covering both healthy and diseased states across 14 crop types.
            </p>
            <ul>
                <li>Input size: 128 × 128 RGB images</li>
                <li>Architecture: Custom CNN with .h5 format</li>
                <li>Training: 70,295 images</li>
                <li>Validation: 17,572 images</li>
                <li>Test set: 33 images</li>
            </ul>
        </div>

        <div class="about-card">
            <h3>🌾 Crop Recommendation Model</h3>
            <p>
                The crop recommendation system uses a neural network trained on an agricultural dataset
                mapping soil nutrients and climate conditions to optimal crops. Input features include
                Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, and Rainfall.
            </p>
            <ul>
                <li>Model: Deep neural network (.keras format)</li>
                <li>Preprocessing: StandardScaler normalization</li>
                <li>Output: Label-encoded crop name via inverse transform</li>
                <li>Covers 22 major crops suited to diverse climates</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="about-card" style="text-align:center;">
            <h3>📊 Dataset Stats</h3>
            <div class="stat-row" style="flex-direction:column; gap:0.8rem;">
                <div class="stat-box"><div class="num">87K+</div><div class="lbl">Total Images</div></div>
                <div class="stat-box"><div class="num">38</div><div class="lbl">Disease Classes</div></div>
                <div class="stat-box"><div class="num">14</div><div class="lbl">Crop Types</div></div>
                <div class="stat-box"><div class="num">80/20</div><div class="lbl">Train / Val Split</div></div>
                <div class="stat-box"><div class="num">22</div><div class="lbl">Recommended Crops</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-card" style="margin-top:0;">
        <h3>🛠 Technology Stack</h3>
        <div style="display:flex; flex-wrap:wrap; gap:0.6rem; margin-top:0.5rem;">
            <span style="background:#52b78820;border:1px solid #52b788;border-radius:8px;padding:0.3rem 0.9rem;font-size:0.85rem;color:#b7e4c7;">TensorFlow / Keras</span>
            <span style="background:#52b78820;border:1px solid #52b788;border-radius:8px;padding:0.3rem 0.9rem;font-size:0.85rem;color:#b7e4c7;">Streamlit</span>
            <span style="background:#52b78820;border:1px solid #52b788;border-radius:8px;padding:0.3rem 0.9rem;font-size:0.85rem;color:#b7e4c7;">NumPy</span>
            <span style="background:#52b78820;border:1px solid #52b788;border-radius:8px;padding:0.3rem 0.9rem;font-size:0.85rem;color:#b7e4c7;">Scikit-learn</span>
            <span style="background:#52b78820;border:1px solid #52b788;border-radius:8px;padding:0.3rem 0.9rem;font-size:0.85rem;color:#b7e4c7;">Python 3.10+</span>
            <span style="background:#52b78820;border:1px solid #52b788;border-radius:8px;padding:0.3rem 0.9rem;font-size:0.85rem;color:#b7e4c7;">Pickle</span>
        </div>
    </div>
    """, unsafe_allow_html=True)



elif "Disease" in app_mode:
    st.markdown("""
    <div class="hero" style="padding:2rem 2.5rem;">
        <div class="hero-badge">🔬 Disease Recognition</div>
        <h1 style="font-size:2.2rem;">Scan Your <span>Plant Leaf</span></h1>
        <p>Upload a clear image of a diseased leaf. Our AI will analyze it and identify the disease within seconds.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("<p class='section-title' style='font-size:1.1rem;'>📁 Upload Leaf Image</p>", unsafe_allow_html=True)
        test_image = st.file_uploader("Choose an image (JPG / PNG / JPEG)", type=["jpg", "jpeg", "png"])

        if test_image:
            st.image(test_image, caption="Uploaded Leaf", use_container_width=True)

    with col2:
        st.markdown("<p class='section-title' style='font-size:1.1rem;'>🤖 AI Prediction</p>", unsafe_allow_html=True)

        if test_image:
            if st.button("🔍  Analyze Disease", use_container_width=True):
                with st.spinner("Analyzing leaf pattern..."):
                    result_index = model_prediction(test_image)
                    disease = CLASS_NAMES[result_index]

                is_healthy = "Healthy" in disease
                icon = "✅" if is_healthy else "⚠️"
                color = "#52b788" if is_healthy else "#e9c46a"

                st.markdown(f"""
                <div class="result-box" style="border-color:{color};">
                    <div class="label">Prediction Result</div>
                    <div class="value" style="color:{color};">{icon} {disease}</div>
                    <div style="font-size:0.8rem;color:#81c784;margin-top:0.5rem;">
                        {'Plant appears healthy 🎉' if is_healthy else 'Disease detected — take action promptly'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#162d1f;border:2px dashed #2d6a4f44;border-radius:16px;padding:3rem;text-align:center;">
                <div style="font-size:3rem;">🍃</div>
                <p style="color:#81c784;margin:0.5rem 0 0;font-size:0.9rem;">Upload an image to begin analysis</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.82rem; color:#81c784; line-height:1.9;">
            <b style="color:#b7e4c7">📌 Tips for best results:</b><br>
            • Use clear, well-lit photos<br>
            • Focus on a single leaf<br>
            • Avoid blurry or dark images<br>
            • Crop out background clutter<br>
            • Supported crops: Apple, Grape, Tomato, Potato, Corn & more
        </div>
        """, unsafe_allow_html=True)



elif "Crop" in app_mode:
    st.markdown("""
    <div class="hero" style="padding:2rem 2.5rem;">
        <div class="hero-badge">🌾 Crop Recommendation</div>
        <h1 style="font-size:2.2rem;">Find Your <span>Perfect Crop</span></h1>
        <p>Enter your soil and climate parameters below. Our model will recommend the most suitable crop to maximize your yield.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p class='section-title' style='font-size:1.15rem;'>🧪 Soil & Climate Parameters</p>", unsafe_allow_html=True)
    st.markdown("<p class='section-sub'>Fill in the details based on your soil test report and local climate data.</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        nitrogen    = st.number_input("🟢 Nitrogen (N)", min_value=0.0, max_value=200.0, step=0.1, help="kg/ha")
        temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0, max_value=50.0, step=0.1)
        ph          = st.number_input("🧪 pH Value", min_value=0.0, max_value=14.0, step=0.01)
    with col2:
        phosphorus  = st.number_input("🟡 Phosphorus (P)", min_value=0.0, max_value=200.0, step=0.1, help="kg/ha")
        humidity    = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
    with col3:
        potassium   = st.number_input("🔴 Potassium (K)", min_value=0.0, max_value=200.0, step=0.1, help="kg/ha")
        rainfall    = st.number_input("🌧️ Rainfall (mm)", min_value=0.0, max_value=500.0, step=0.1)

    st.markdown("<br>", unsafe_allow_html=True)

    col_btn, col_empty = st.columns([1, 2])
    with col_btn:
        predict_btn = st.button("🌱  Recommend Crop", use_container_width=True)

    if predict_btn:
        with st.spinner("Analysing soil & climate data..."):
            BASE_DIR = Path(__file__).resolve().parent.parent

            model = tf.keras.models.load_model(BASE_DIR / "Models" / "crop_recommendation_model.keras")
            import pickle as pkl

            with open(BASE_DIR / "Models" / "scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
            with open(BASE_DIR / "Models" / "label_encoder.pkl", "rb") as f:
                encoder = pkl.load(f)

            data = scaler.transform([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
            prediction = model.predict(data)
            predicted_crop = encoder.inverse_transform([np.argmax(prediction)])[0]

        st.markdown(f"""
        <div class="result-box" style="border-color:#e9c46a; margin-top:1rem;">
            <div class="label">Recommended Crop for Your Soil & Climate</div>
            <div class="value">🌾 {predicted_crop.upper()}</div>
            <div style="font-size:0.82rem;color:#81c784;margin-top:0.5rem;">
                Based on N={nitrogen} | P={phosphorus} | K={potassium} | Temp={temperature}°C | Humidity={humidity}% | pH={ph} | Rainfall={rainfall}mm
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div class="about-card" style="margin-top:0;">
        <h3>📌 How to get accurate values?</h3>
        <ul>
            <li><b style='color:#b7e4c7'>N, P, K</b> — Get a soil test done at your local agriculture office or lab (values in kg/ha).</li>
            <li><b style='color:#b7e4c7'>Temperature</b> — Average temperature of your region during the growing season (°C).</li>
            <li><b style='color:#b7e4c7'>Humidity</b> — Average relative humidity percentage of your area.</li>
            <li><b style='color:#b7e4c7'>pH</b> — Soil pH measured by a pH meter or test kit (0–14 scale; ideal 6–7 for most crops).</li>
            <li><b style='color:#b7e4c7'>Rainfall</b> — Annual average rainfall in millimeters from local weather data.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


    
