import streamlit as st
from PIL import Image
from helper import describe_image_with_gemini
# from rag_chain import get_rag_chain  # Uncomment if using RAG

st.set_page_config(page_title="Image Description with Gemini", layout="centered")
st.title("🖼️ Image Description App with Gemini")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing image with Gemini..."):
        description = describe_image_with_gemini(image)

    st.markdown("### 📜 Description")
    st.write(description)

    # Optional: Augment with RAG
    # st.markdown("### 🔍 Augmented Info")
    # rag_chain = get_rag_chain()
    # augmented_response = rag_chain.run(f"Give more information about: {description}")
    # st.write(augmented_response)
