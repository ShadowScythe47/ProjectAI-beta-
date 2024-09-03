import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer, MarianMTModel, MarianTokenizer


summarization_model = T5ForConditionalGeneration.from_pretrained('t5-small')
summarization_tokenizer = T5Tokenizer.from_pretrained('t5-small')

model_en_to_es = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-es')
tokenizer_en_to_es = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-es')

model_es_to_en = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-es-en')
tokenizer_es_to_en = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-es-en')

def summarize(text, max_length=150, min_length=40):
    input_ids = summarization_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = summarization_model.generate(input_ids, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

st.title("Text Processing Tool(Beta Version)")
option = st.sidebar.selectbox("Select an option", ["Translate Text", "Summarize Text"])
if option == "Translate Text":
    st.header("Language Translation")
    direction = st.selectbox("Select Translation Direction", ["English to Spanish", "Spanish to English"])
    direction_map = {
        "English to Spanish": "en_to_es",
        "Spanish to English": "es_to_en"
    }
    direction_param = direction_map[direction]

    text_input = st.text_area("Enter text to translate")
    if st.button("Translate"):
        if text_input:
            
            if direction_param == 'en_to_es':
                model = model_en_to_es
                tokenizer = tokenizer_en_to_es
            elif direction_param == 'es_to_en':
                model = model_es_to_en
                tokenizer = tokenizer_es_to_en
            else:
                st.write("Language Input Error")
            input_ids_1 = tokenizer.encode(text_input, return_tensors="pt")
            translated = model.generate(input_ids_1, max_length=512, num_beams=4, early_stopping=True)
            translation = tokenizer.decode(translated[0], skip_special_tokens=True)
            st.subheader("Translation")
            st.write(translation)
            
        else:
            st.write("Please enter some text to translate.")

elif option == "Summarize Text":
    st.header("Text Summarization")
    user_input = st.text_area("Enter the text you want to summarize", height=300) 
    max_len = st.slider("Max Length of Summary", 50, 300, 150)
    min_len = st.slider("Min Length of Summary", 20, 100, 40)

    if st.button("Summarize"):
        if user_input:
            summary = summarize(user_input, max_length=max_len, min_length=min_len)
            st.subheader("Summary")
            st.write(summary)
        else:
            st.write("Please enter some text to summarize.")
