import streamlit as st
import pytesseract
import os
import gc
import uuid
import re
import base64
import io
from PIL import Image
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from langchain_openai import ChatOpenAI

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
    
session_id = st.session_state.id

def reset_chat():
    st.session_state.messages = []  
    st.session_state.context = None  
    st.session_state.formatted_code = None 
    st.session_state.file_cache = {}  
    gc.collect() 
    
def display_image(image):
    """ Display the uploaded image in Streamlit with base64 encoding for inline display. """
    # Convert the image to a byte stream
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    
    # Base64 encode the image for inline display
    base64_img = base64.b64encode(img_bytes).decode('utf-8')
    
    # HTML code to embed the image inline
    image_html = f'<img src="data:image/png;base64,{base64_img}" width="800" style="max-width:100%; height:auto;">'
    
    # Display the image using Streamlit's markdown with HTML
    st.markdown(image_html, unsafe_allow_html=True)
    
@st.cache_resource
def load_llm(model_name="mistral"):
    if model_name == "mistral":
        llm = Ollama(model="mistral", request_timeout=120.0)
        Settings.llm = llm
    elif model_name == "OPENAI":
        # Ensure OpenAI API key is provided
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.5,
            openai_api_key=openai_api_key
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return llm


def clean_raw_text(raw_text):
    """ Clean up raw text to reduce OCR artifacts. """
    # Remove unwanted spaces, newline characters, and unnecessary symbols
    cleaned_text = re.sub(r'\s+', ' ', raw_text)
    cleaned_text = re.sub(r'[^a-zA-Z0-9_\-\+\*\/\=\.\(\)\[\]\{\}\:,\.\s]', '', cleaned_text)
    return cleaned_text.strip()

# Function to extract text
def extract_text(image):
    """ Convert PIL Image to bytes and extract text using pytesseract """
    import io
    img_bytes = io.BytesIO()
    image.save(img_bytes, format=image.format)
    img_bytes = img_bytes.getvalue()
    return pytesseract.image_to_string(Image.open(io.BytesIO(img_bytes)))

def refine_code(raw_text, model_choice):
    cleaned_text = clean_raw_text(raw_text)

    if not cleaned_text:
        return None

    prompt = f"""
        Here is some raw text extracted from an image:
        ```
            {cleaned_text}
        ```
        Please format it properly as Python code, fixing any errors,
        missing parentheses, indentation, and ensuring correct syntax.
    """
    
    try:
        llm = load_llm(model_name=model_choice)
        if llm is None:
            raise ValueError("LLM could not be loaded.")

        if isinstance(llm, Ollama): 
            response = llm.complete(prompt)
        elif isinstance(llm, ChatOpenAI):
            response = llm.predict(prompt)
        else:
            raise ValueError("Unsupported LLM type.")

        return response

    except Exception as e:
        st.error(f"An error occurred while refining the code: {str(e)}")
        return None


col1, col2 = st.columns([6, 1])

with col1:
    st.header(f"CodeExtraction Chatbot!")

with col2:
    st.button("Clear ↺", on_click=reset_chat)
    st.markdown('<p style="font-size: 12px; color: grey;">To clear the code, you must close the image.</p>', 
                unsafe_allow_html=True)
    
# Initialize session state
if "messages" not in st.session_state or \
    "formatted_code" not in st.session_state or\
        "uploaded_file" not in st.session_state:
            reset_chat()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Load image at left column
with st.sidebar:
    st.header("Add your image!")
    uploaded_file = st.file_uploader("Upload an image (PNG, JPG, JPEG)", 
                                     type=["png", "jpg", "jpeg"])

    # Model selection dropdown
    model_choice = st.selectbox("Select LLM Model",
                                ["mistral", "OPENAI"],
                                index=0)
        
    if uploaded_file:
        # Display Image in the left column
        image = Image.open(uploaded_file)
        display_image(image)
        
# Show formatted code at center
if uploaded_file:   
    # Extract Text from the uploaded image
    raw_text = extract_text(image)
            
    try:
        formatted_code = refine_code(raw_text, model_choice)
        st.subheader("Code")
        st.code(formatted_code, language="python")
        st.session_state.formatted_code = formatted_code
    except Exception as e:
        st.error(f"An error occurred while refining the code: {str(e)}")
        st.text("Traceback for debugging:")
        st.text(e)

    
# Chat Input
user_input = st.chat_input("Ask about the extracted code...")

if user_input:    
    st.session_state.messages.append({"role": "user", "content": user_input})

    prompt = f"""
        The following Python code was extracted from an image:
        ```
            {st.session_state.formatted_code}
        ```
        Carefully analyze this code.

        The user has asked the following question about it:
        "{user_input}"

        Please answer the question based on the provided Python code, 
        referencing variables, functions, and their purpose as needed.
    """
    
    try:
        # Load LLM model
        llm = load_llm(model_name=model_choice)

        if llm:
            if isinstance(llm, Ollama):
                response = llm.complete(prompt)
            elif isinstance(llm, ChatOpenAI):
                response = llm.predict(prompt)
            else:
                response = "Unsupported LLM type."

            # Display the assistant's response
            with st.chat_message("assistant"):
                st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.error("Failed to load the LLM model.")

    except Exception as e:
        st.error(f"An error occurred while processing the chat input: {str(e)}")