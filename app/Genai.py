import streamlit as st
import google.generativeai as genai
import os

# Configure API key
GOOGLE_API_KEY = st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

st.title("🧠 Gemini Health Advisor")

model_list = {
    "Gemini 1.5 Pro (Latest)": "gemini-1.5-pro-latest",
    "Gemini 1.5 Flash (Latest)": "gemini-1.5-flash-latest"
}

selected_label = st.selectbox("Choose a Gemini model:", list(model_list.keys()))
selected_model = model_list[selected_label]

user_input = st.text_area("💬 Enter your health question:")

if st.button("🚀 Ask AI"):
    with st.spinner("Thinking..."):
        try:
            model = genai.GenerativeModel(model_name=selected_model)
            chat = model.start_chat(history=[])
            response = chat.send_message(user_input)
            st.success("✅ Response received:")
            st.markdown(response.text)
        except Exception as e:
            st.error(f"❌ Error fetching AI response: {str(e)}")
