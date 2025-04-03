import streamlit as st
from rag_pipeline import (
    upload_pdf, load_pdf, create_chunks, index_documents, retrieve_data, generate_questions_answers, check_image_relevance
)

st.title("📚 AI-Powered Question Answer Generator")

uploaded_files = st.file_uploader("📂 Upload PDFs", type="pdf", accept_multiple_files=True)
user_query = st.text_area("Enter a topic/question:", height=150, placeholder="Enter a topic to generate questions.")

question_marks = st.slider("Minimum question marks per question", min_value=1, max_value=10, value=2)
num_questions = st.slider("Number of questions to generate", min_value=1, max_value=10, value=5)

# Dropdown for Bloom’s Taxonomy selection
blooms_level = st.selectbox(
    "Select Bloom’s Taxonomy Level",
    ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create", "Mix"]
)

ask_question = st.button("Generate Q&A")

if ask_question:
    if uploaded_files:
        all_documents = []
        all_images = []

        for file in uploaded_files:
            file_path = upload_pdf(file)  
            documents, images = load_pdf(file_path)  # Updated to handle images
            all_documents.extend(documents)
            all_images.extend(images)

        text_chunks = create_chunks(all_documents)
        index_documents(text_chunks)

        response = generate_questions_answers(
            user_query=user_query,
            num_questions=num_questions,
            question_marks=question_marks,
            blooms_level=blooms_level
        )

        st.markdown("### Generated Questions & Answers")
        st.markdown(response)

        # Check for relevant images and append them
        relevant_images = check_image_relevance(user_query, all_images)
        if relevant_images:
            st.markdown("### Relevant Images and Extracted Text")
            for image, extracted_text in relevant_images:
                st.image(image, caption="Relevant Image", use_container_width=True)
                st.markdown(f"**Extracted Text:**\n{extracted_text}")

    else:
        st.error("⚠️ Please upload at least one PDF file.")
