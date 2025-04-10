import streamlit as st
import re
import os
from datetime import datetime
from rag_pipeline import (
    upload_pdf,
    load_pdf,
    create_chunks,
    index_documents,
    retrieve_data,
    generate_questions_answers,
    check_image_relevance,
)

st.set_page_config(page_title="AI-Powered Q&A Generator", layout="wide")
st.title("📚 AI-Powered Question Answer Generator")

# Initialize session state variables
if "selected_qna" not in st.session_state:
    st.session_state.selected_qna = []
if "generated_qna" not in st.session_state:
    st.session_state.generated_qna = []
if "questions_list" not in st.session_state:
    st.session_state.questions_list = []
if "generated_response" not in st.session_state:
    st.session_state.generated_response = None

# Function to add a question to selections
def add_question(question_index):
    full_qna = st.session_state.generated_qna[question_index]
    if full_qna not in st.session_state.selected_qna:
        st.session_state.selected_qna.append(full_qna)
        st.success(f"Question {question_index+1} added to selections!")

# Function to clear selections
def clear_selections():
    st.session_state.selected_qna = []
    st.success("All selections cleared!")

# Function to save selections to file
def save_selections():
    if not st.session_state.selected_qna:
        st.error("No questions selected to save!")
        return
        
    # Create a timestamp-based filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"selected_qna_{timestamp}.txt"
    
    file_content = "\n\n--------------------\n\n".join(st.session_state.selected_qna)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(file_content)
    
    return filename

# Improved parsing function for Q&A content
def parse_qa_content(response):
    qa_items = []
    generated_qna = []
    
    # Find all questions with their content
    qna_blocks = re.findall(r"(?:\*\*Q(\d+):\*\*)(.*?)(?=\*\*Q\d+:\*\*|\Z)", response, re.DOTALL)
    
    for i, (q_num, content) in enumerate(qna_blocks):
        # Extract question
        question_match = re.search(r"(.*?)(?:\*\*A\d+)", content, re.DOTALL)
        question = question_match.group(1).strip() if question_match else content.strip()
        
        # Extract LLM answer
        llm_match = re.search(r"\*\*A\d+ \(LLM\):\*\*(.*?)(?:\*\*A\d+ \(From Document\)|\*\*Document References|\Z)", content, re.DOTALL)
        llm_answer = llm_match.group(1).strip() if llm_match else "No LLM answer available"
        
        # Extract document answer
        doc_match = re.search(r"\*\*A\d+ \(From Document\):\*\*(.*?)(?:\*\*Document References|\Z)", content, re.DOTALL)
        doc_answer = doc_match.group(1).strip() if doc_match else ""
        
        # If no document answer found, try to extract document references
        if not doc_answer:
            doc_ref_match = re.search(r"\*\*Document References:\*\*(.*?)(?:\Z)", content, re.DOTALL)
            doc_answer = doc_ref_match.group(1).strip() if doc_ref_match else "No document answer available"
        
        # Format for display and storage
        q_display = f"**Q{i+1}:** {question}"
        a_llm = f"**A{i+1} (LLM):** {llm_answer}"
        a_doc = f"**A{i+1} (From Document):** {doc_answer}"
        
        qa_items.append({
            "question": q_display,
            "answer_llm": a_llm,
            "answer_doc": a_doc
        })
        
        full_qna = f"{q_display}\n\n{a_llm}\n\n{a_doc}"
        generated_qna.append(full_qna)
    
    return qa_items, generated_qna

# Function to generate QnA and store in session state
def generate_qna():
    if uploaded_files:
        all_documents, all_images = [], []

        for file in uploaded_files:
            file_path = upload_pdf(file)
            documents, images = load_pdf(file_path)
            all_documents.extend(documents)
            all_images.extend(images)

        chunks = create_chunks(all_documents)
        index_documents(chunks)

        response = generate_questions_answers(
            user_query=user_query,
            num_questions=num_questions,
            question_marks=question_marks,
            blooms_level=blooms_level,
        )
        
        # Store the response in session state
        st.session_state.generated_response = response
        
        # Use the improved parsing function
        st.session_state.questions_list, st.session_state.generated_qna = parse_qa_content(response)
        
        return all_images
    else:
        st.error("⚠️ Please upload at least one PDF file.")
        return None

# Sidebar Inputs
with st.sidebar:
    uploaded_files = st.file_uploader("📂 Upload PDFs", type="pdf", accept_multiple_files=True)
    user_query = st.text_area("🧠 Enter a topic/question:", height=150, placeholder="Enter a topic to generate questions.")
    question_marks = st.slider("✍️ Minimum question marks per question", min_value=1, max_value=10, value=2)
    num_questions = st.slider("🔢 Number of questions to generate", min_value=1, max_value=10, value=5)
    blooms_level = st.selectbox(
        "🎓 Select Bloom's Taxonomy Level",
        ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create", "Mix"]
    )
    ask_question = st.button("🚀 Generate Questions & Answers")

# Generate Q&A
if ask_question:
    with st.spinner("Generating questions and answers..."):
        all_images = generate_qna()

# Display generated questions (whether they were just generated or from a previous run)
if st.session_state.questions_list:
    st.markdown("### 🎯 Generated Questions & Answers")
    
    for i, qa_item in enumerate(st.session_state.questions_list):
        with st.expander(f"Question {i+1}"):
            st.markdown(qa_item["question"])
            st.markdown("---")
            st.markdown(qa_item["answer_llm"])
            st.markdown("---")
            st.markdown(qa_item["answer_doc"])
            
            # Add a save button for each question
            st.button(f"💾 Save Question {i+1}", key=f"save_btn_{i}", on_click=add_question, args=(i,))
    
    # Show relevant images if they were just generated
    if ask_question and all_images:
        relevant_images = check_image_relevance(user_query, all_images)
        if relevant_images:
            st.markdown("### 🖼️ Relevant Images and Extracted Text")
            for img, extracted_text in relevant_images:
                st.image(img, caption="Relevant Image", use_container_width=True)
                st.markdown(f"**Extracted Text:**\n{extracted_text}")

# Display and manage selected Q&As
if st.session_state.selected_qna:
    st.markdown("---")
    st.subheader("✅ Selected Questions & Answers")
    
    for idx, item in enumerate(st.session_state.selected_qna, 1):
        with st.expander(f"Selected Q&A {idx}"):
            st.markdown(item)
    
    # Create columns for the buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Save Selected Q&As to File"):
            filename = save_selections()
            if filename:
                st.success(f"✅ Saved to `{filename}`")
                
                with open(filename, "rb") as file:
                    st.download_button(
                        label="⬇️ Download Q&As File",
                        data=file,
                        file_name=filename,
                        mime="text/plain"
                    )
    
    with col2:
        st.button("🧹 Clear All Selections", on_click=clear_selections)
