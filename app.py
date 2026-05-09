import os
import tempfile
import streamlit as st
from PIL import Image
from gtts import gTTS
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Kids Storytelling App",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Kids Storytelling App")
st.write("Upload an image and generate a children's story with audio.")


# --------------------------------------------------
# Load BLIP image captioning model
# Model: Salesforce/blip-image-captioning-base
# --------------------------------------------------
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    return processor, model


# --------------------------------------------------
# Load FLAN-T5 story generation model
# Model: google/flan-t5-base
# --------------------------------------------------
@st.cache_resource
def load_story_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    return tokenizer, model


# --------------------------------------------------
# Function 1: Image to text
# --------------------------------------------------
def img2text(uploaded_image):
    processor, model = load_blip_model()

    if uploaded_image.mode != "RGB":
        uploaded_image = uploaded_image.convert("RGB")

    inputs = processor(images=uploaded_image, return_tensors="pt")
    output = model.generate(
        **inputs,
        max_new_tokens=30,
        num_beams=5
    )

    caption = processor.decode(output[0], skip_special_tokens=True).strip()
    return caption


# --------------------------------------------------
# Helper: check repetitive output
# --------------------------------------------------
def is_too_repetitive(text):
    words = text.lower().split()
    if len(words) < 20:
        return True

    unique_ratio = len(set(words)) / len(words)
    return unique_ratio < 0.30


# --------------------------------------------------
# Function 2: Text to story
# --------------------------------------------------
def text2story(caption):
    tokenizer, model = load_story_model()

    prompt = (
        "Write a short children's story in simple English based on this image description: "
        f"{caption}. "
        "Write 5-6 sentences. "
        "Use clear and easy words for young children. "
        "Make the story cheerful, complete, and natural. "
        "Do not repeat the same idea. "
        "End with a happy ending."
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        min_new_tokens=45,
        do_sample=True,
        temperature=0.9,
        top_p=0.92,
        no_repeat_ngram_size=3
    )

    story = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()

    if len(story.split()) >= 30 and not is_too_repetitive(story):
        return story

    return (
        f"One day, {caption} made everyone smile. "
        f"The children laughed and played together happily. "
        f"Soon, they found a fun little adventure in the park. "
        f"They helped each other and shared a wonderful time. "
        f"At the end of the day, everyone went home with happy hearts."
    )


# --------------------------------------------------
# Function 3: Text to audio
# --------------------------------------------------
def text2audio(story_text):
    tts = gTTS(text=story_text, lang="en")
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)
    return temp_audio.name


# --------------------------------------------------
# Upload image
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# Main app
# --------------------------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Story"):
        try:
            with st.spinner("Generating image caption..."):
                caption = img2text(image)

            with st.spinner("Generating story..."):
                story = text2story(caption)

            with st.spinner("Generating audio..."):
                audio_path = text2audio(story)

            st.success("Story generated successfully!")

            st.subheader("🖼️ Image Caption")
            st.write(caption)

            st.subheader("✨ Story")
            st.write(story)
            st.write(f"**Word count:** {len(story.split())}")

            st.subheader("🔊 Story Audio")
            with open(audio_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/mp3")

                st.download_button(
                    label="Download Audio",
                    data=audio_bytes,
                    file_name="kids_story.mp3",
                    mime="audio/mpeg"
                )

            if os.path.exists(audio_path):
                os.remove(audio_path)

        except Exception as e:
            st.error(f"An error occurred: {e}")

else:
    st.info("Please upload an image to begin.")
