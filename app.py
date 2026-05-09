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

    inputs = processor(images=uploaded_image, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=30)

    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption.strip()


def is_too_repetitive(text):
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    if len(words) < 20:
        return True

    unique_ratio = len(set(words)) / len(words)
    return unique_ratio < 0.30


def text2story(caption):
    tokenizer, model = load_story_model()

    prompt = (
        "Write a short children's story in simple English based on this image description: "
        f"{caption}. "
        "Write 5 to 6 sentences. "
        "Use clear and easy words for young children. "
        "Make the story cheerful and natural. "
        "End with a happy ending."
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        num_beams=4,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    story = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()

    if len(story.split()) >= 30 and not is_too_repetitive(story):
        return story

    return (
        f"One day, {caption} made everyone smile. "
        f"The children laughed and played together happily. "
        f"They had a fun little adventure and helped one another. "
        f"Everyone enjoyed the beautiful day very much. "
        f"At the end of the day, they went home with happy hearts."
    )


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
