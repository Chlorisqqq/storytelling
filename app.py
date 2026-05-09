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

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Image to Story", page_icon="📖")
st.title("Image to Story Generator")
st.write("Upload an image to generate a caption, a short story, and audio narration.")

# ----------------------------
# Load BLIP model
# ----------------------------
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# ----------------------------
# Load FLAN-T5 model
# ----------------------------
@st.cache_resource
def load_story_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model

# ----------------------------
# Image to caption
# ----------------------------
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

# ----------------------------
# Caption to story
# ----------------------------
def text2story(caption):
    tokenizer, model = load_story_model()

    prompt = (
        "Write a short children's story based on this image description: "
        f"{caption}. "
        "Write a children's story of about 50 to 100 words. "
        "Use 5 to 6 simple sentences. "
        "Do not repeat the same idea. "
        "Stay close to the image description, but add a few natural details. "
        "Use simple English and end with a happy ending."
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        min_new_tokens=40,
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
        early_stopping=True
    )

    story = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    return story

# ----------------------------
# Story to audio
# ----------------------------
def text2speech(story_text):
    tts = gTTS(text=story_text, lang="en")

    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)
    return temp_audio.name

# ----------------------------
# Upload image
# ----------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Story"):
        with st.spinner("Generating..."):
            caption = img2text(image)
            story = text2story(caption)
            audio_file = text2speech(story)

        st.subheader("Image Caption")
        st.write(caption)

        st.subheader("Story")
        st.write(story)

        st.audio(audio_file)
