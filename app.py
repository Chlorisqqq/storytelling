import os
import re
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

st.set_page_config(
    page_title="Kids Storytelling App",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Kids Storytelling App")
st.write("Upload an image and generate a children's story with audio.")


@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model


@st.cache_resource
def load_story_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model


def img2text(uploaded_image):
    processor, model = load_blip_model()

    if uploaded_image.mode != "RGB":
        uploaded_image = uploaded_image.convert("RGB")

    inputs = processor(
        images=uploaded_image,
        text="Describe this image in detail.",
        return_tensors="pt"
    )

    output = model.generate(
        **inputs,
        max_new_tokens=50,
        num_beams=5
    )

    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption.strip()


def text2story(caption):
    tokenizer, model = load_story_model()

    prompt = (
        "Write a children's story based on this image description: "
        f"{caption}. "
        "Write 5 simple sentences. "
        "Stay close to the image description, but add a few natural details. "
        "Use simple English and end with a happy ending."
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        min_new_tokens=40,
        num_beams=4,
        no_repeat_ngram_size=3
    )

    story = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    return story


def text2audio(story_text):
    tts = gTTS(text=story_text, lang="en")
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)
    return temp_audio.name


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width="stretch")

    if st.button("Generate Story"):
        try:
            caption = img2text(image)
            story = text2story(caption)
            audio_path = text2audio(story)

            st.subheader("Image Caption")
            st.write(caption)

            st.subheader("Story")
            st.write(story)

            with open(audio_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/mp3")

            os.remove(audio_path)

        except Exception as e:
            st.error(f"An error occurred: {e}")
