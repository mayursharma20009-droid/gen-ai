import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Page config
st.set_page_config(page_title="AI Text Summarizer")

st.title("🕷️Advanced AI Text Summarizer")

# Load model (cached for speed)
@st.cache_resource
def load_model():
    model_name = "google-t5/t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# User input
text = st.text_area("Enter your text here:", height=200)

# Word count
word_count = len(text.split())
st.write(f"📝 Word Count: {word_count}")

# Button
if st.button("Summarize"):

    if not text.strip():
        st.warning("Please enter some text.")
    else:
        # Prepare input
        input_text = "summarize: " + text

        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

        # Function to generate summaries
        def generate_summary(max_len):
            outputs = model.generate(
                inputs["input_ids"],
                max_length=max_len,
                min_length=20,
                num_beams=4,
                early_stopping=True
            )
            return tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Generate multiple summaries
        short_summary = generate_summary(50)
        medium_summary = generate_summary(100)
        detailed_summary = generate_summary(150)

        # Display summaries
        st.subheader("🙊 Short Summary")
        st.write(short_summary)

        st.subheader("🙉 Medium Summary")
        st.write(medium_summary)

        st.subheader("🙈 Detailed Summary")
        st.write(detailed_summary)

        # Generate key points
        key_input = "summarize in bullet points: " + text

        key_inputs = tokenizer(
            key_input,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

        key_outputs = model.generate(
            key_inputs["input_ids"],
            max_length=120,
            num_beams=4,
            early_stopping=True
        )

        key_points = tokenizer.decode(
            key_outputs[0],
            skip_special_tokens=True
        )

        # Display key points
        st.subheader("🔑 Key Points")
        st.write(key_points)