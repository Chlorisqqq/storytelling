import streamlit as st
from PIL import Image
from gtts import gTTS
import tempfile

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

# Page config
st.set_page_config(page_title="Kids Image Story Generator", page_icon="📖", layout="centered")
st.markdown(
    """
    <style>
    /* Main app background */
    .stApp {
        background: linear-gradient(to bottom, #fff8f0, #fef6ff);
        font-family: 'Trebuchet MS', 'Segoe UI', sans-serif;
    }

    /* Main title */
    h1 {
        color: #5b4b8a;
        text-align: center;
        font-weight: 800;
    }

    /* Subheaders */
    h2, h3 {
        color: #6c63a8;
        font-weight: 700;
    }

    /* Paragraph text */
    p, li {
        font-size: 17px;
        color: #333333;
    }

    /* Button styling */
    div.stButton > button {
        background-color: #ffcc70;
        color: #333333;
        border: none;
        border-radius: 12px;
        padding: 0.6em 1.2em;
        font-size: 16px;
        font-weight: bold;
        transition: 0.3s ease;
    }

    div.stButton > button:hover {
        background-color: #ffb84d;
        color: black;
    }

    /* File uploader */
    section[data-testid="stFileUploader"] {
        background-color: rgba(255, 255, 255, 0.7);
        padding: 1rem;
        border-radius: 14px;
        border: 2px dashed #d8c6f0;
    }

    /* Message boxes */
    div[data-testid="stAlert"] {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("Kids Image Story Generator")
st.markdown(
    """
    Welcome! Upload an image and this app will:

    1. **Describe the image**
    2. **Create a short story**
    3. **Read the story aloud**

    This app is designed for **children aged 3–10**.
    """)

st.info("👆 Start by uploading a JPG or PNG image below.")

# Load BLIP model
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# Load FLAN-T5 model
@st.cache_resource
def load_story_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model

# Image to caption
def img2text(uploaded_image):
    processor, model = load_blip_model()

    if uploaded_image.mode != "RGB":
        uploaded_image = uploaded_image.convert("RGB")

    inputs = processor(images=uploaded_image, return_tensors="pt")

    output = model.generate(
        **inputs,
        max_new_tokens=30
    )

    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption.strip()

# Caption to story
def text2story(caption):
    tokenizer, model = load_story_model()

    prompt = (
        "Write a short children's story based on this image description: "
        f"{caption}. "
        "Use 5 to 6 simple sentences. "
        "Do not repeat the same idea. "
        "Stay close to the image description, but add a few natural details. "
        "Use simple English and end with a happy ending."
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
    **inputs,
        max_new_tokens=130,
        min_new_tokens=55,
        do_sample=True,
        temperature=0.9,
        top_p=0.92,
        no_repeat_ngram_size=4,
        repetition_penalty=1.15
    )

    story = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    return story

# Story to audio
def text2speech(story_text):
    tts = gTTS(text=story_text, lang="en")

    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)
    return temp_audio.name

# Upload image
st.subheader("Step 1: Upload an Image 🖼️")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.success("✅ Image uploaded successfully!")

st.subheader("Step 2: Create Your Story ✨")

if st.button("✨ Create Story"):
    with st.spinner("Creating your story and audio..."):
        caption = img2text(image)
        story = text2story(caption)
        audio_file = text2speech(story)

    st.subheader("Step 3: Enjoy Your Story 🎉")

    st.markdown("### 🏷️ Picture Description")
    st.write(caption)

    st.markdown("### 📖 Your Story")
    st.write(story)

    word_count = len(story.split())
    st.markdown("### 📏 Story Word Count")
    st.write(f"{word_count} words")

    st.markdown("### 🔊 Listen to the Story")
    st.audio(audio_file)
