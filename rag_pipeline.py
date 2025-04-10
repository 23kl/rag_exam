import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from groq import Groq
from langchain_ollama import OllamaEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import re

# Define directories
PDFS_DIRECTORY = "pdfs/"
FAISS_DB_PATH = "vectorstore/db_faiss"
TOP_N_LIMIT = 10

# Initialize Groq Client
groq_key = "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"

client = Groq(api_key=groq_key)

# Ensure necessary directories exist
os.makedirs(PDFS_DIRECTORY, exist_ok=True)
os.makedirs("vectorstore", exist_ok=True)

# Upload PDF
def upload_pdf(file):
    file_path = os.path.join(PDFS_DIRECTORY, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

# Load PDF and extract text
def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    images = extract_images_from_pdf(file_path)
    return documents, images

# Extract images from PDF
def extract_images_from_pdf(file_path):
    images = []
    doc = fitz.open(file_path)

    for page_number in range(len(doc)):
        for img_index, img in enumerate(doc[page_number].get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            img_format = base_image["ext"]
            
            image_path = f"extracted_images/page_{page_number+1}_img_{img_index+1}.{img_format}"
            os.makedirs("extracted_images", exist_ok=True)
            
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            
            images.append(image_path)
    
    # If no embedded images, use pdf2image as a fallback
    if not images:
        pdf_images = convert_from_path(file_path)
        for i, img in enumerate(pdf_images):
            image_path = f"extracted_images/page_{i+1}.jpg"
            img.save(image_path, "JPEG")
            images.append(image_path)
    
    return images

# Split documents into chunks
def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

# Get Ollama embeddings (DeepSeek R1 1.5B)
def get_embedding_model():
    return OllamaEmbeddings(model="deepseek-r1:1.5b")

# Index documents in FAISS
def index_documents(text_chunks):
    embedding_model = get_embedding_model()
    faiss_db = FAISS.from_documents(text_chunks, embedding_model)
    faiss_db.save_local(FAISS_DB_PATH)

# Load FAISS database
def load_faiss_db():
    return FAISS.load_local(
        FAISS_DB_PATH,
        get_embedding_model(),
        allow_dangerous_deserialization=True
    )

# Retrieve relevant data from FAISS
def retrieve_data(query):
    faiss_db = load_faiss_db()
    return faiss_db.similarity_search(query)

# Extract text from image using Tesseract OCR
def extract_text_from_image(image_path):
    try:
        img = Image.open(image_path)
        extracted_text = pytesseract.image_to_string(img)
        return extracted_text.strip()
    except Exception as e:
        return f"⚠️ Error extracting text from image: {str(e)}"

# Check if extracted text matches the question
def check_image_relevance(query, images):
    relevant_images = []
    for image in images:
        extracted_text = extract_text_from_image(image)
        similarity_score = calculate_similarity(query, extracted_text)
        if similarity_score > 0.5:  # Adjust threshold as needed
            relevant_images.append((image, extracted_text))
    return relevant_images

# Calculate similarity between two texts
def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return round(similarity[0][0], 4)

# Generate questions & answers using Groq
def generate_questions_answers(user_query, num_questions=3, question_marks=7, blooms_level="Apply", subject=None):
    # Import question bank
    from question_bank import get_question_bank
    
    retrieved_docs = retrieve_data(user_query)
    context = "\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else ""

    if num_questions > TOP_N_LIMIT:
        return f"⚠️ I can generate a maximum of {TOP_N_LIMIT} high-quality questions."
    
    if not context:
        context = f"The topic is '{user_query}'. Generate {num_questions} relevant and high-quality questions."

    # Set length instruction based on the depth required (using question_marks instead of question_depth)
    length_instruction = "Each answer should be concise and directly related to the topic."
    if question_marks > 5:
        length_instruction = ("Each answer should be **at least 20 lines long** and should include the following:\n"
                          "- **A clear introduction** explaining the concept.\n"
                          "- **Detailed subtopics** breaking down various aspects of the topic.\n"
                          "- **Relevant examples** to illustrate the concept.\n"
                          "- **Real-world applications** where applicable.\n"
                          "- **Additional insights** related to the question.\n"
                          "- **Properly formatted mathematical expressions** using LaTeX where required.\n"
                          "- Properly formatted mathematical expressions using LaTeX where required"
                          "- **A brief conclusion** summarizing the answer.")

    elif question_marks > 3:
        length_instruction = "Each answer should be moderately detailed (about 10-12 lines long)"

    # Get question bank templates based on subject
    question_bank = get_question_bank(subject)

    bloom_levels = {
        "Remember": "Questions that test recall of facts or basic concepts.",
        "Understand": "Questions that require explaining ideas or concepts.",
        "Apply": "Questions that involve applying knowledge to new situations.",
        "Analyze": "Questions that require analyzing or drawing connections.",
        "Evaluate": "Questions that involve evaluating or justifying a decision.",
        "Create": "Questions that require creating something new or original.",
        "Mix": "A mix of questions from all Bloom's Taxonomy levels."
    }

    bloom_instruction = (f"Generate a mix of questions from all Bloom's Taxonomy levels: {', '.join(bloom_levels.keys())}."
                         if blooms_level == "Mix" else
                         f"Generate questions aligned with the Bloom's Taxonomy level: {blooms_level}. {bloom_levels[blooms_level]}")

    prompt = f"""
    You are an expert in question generation. Based on the following text, generate exactly {num_questions} high-quality and precise questions.
    Each question should include at least {question_marks} question marks and be directly related to the topic: "{user_query}".
    **CRITICAL REQUIREMENT**: All answers MUST be at least {20} lines long with detailed explanations.
    **IMPORTANT INSTRUCTIONS**:
    1. Each question must be FOCUSED on a single aspect of the topic: "{user_query}"
    2. Generate questions that closely follow the style and format of the examples in the question bank below
    3. Questions should be clear, concise, and should NOT contain multiple subquestions
    4. {length_instruction}
    5. For mathematical equations, use LaTeX formatting: inline with `$` and block with `$$`
    6. {bloom_instruction}
    
    ### **Question Bank Format**:
    {question_bank}
    
    ### **Context:**
    {context}
    
    ### **Output Format (Follow this exactly)**:
    ********
    **Q1:** <single focused question>  
    ********
    **Bloom's Level:** <level>  
    ********
    **A1 (LLM):** <comprehensive answer>  
    ********
    
    ... (repeat for all {num_questions} questions)
    """
    
    try:
        completion = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",  # Use the Groq model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=4096,
            top_p=0.95,
            stream=False
        )

        response = completion.choices[0].message.content if completion.choices else "No response generated."
    
    except Exception as e:
        return f"⚠️ Error while generating questions: {str(e)}"
    
    # Clean and format response
    response = response.replace("\\(", "$").replace("\\)", "$")
    response = response.replace("\\[", "$$").replace("\\]", "$$")
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    response = response.replace("think tag", "")  
    
    # Fix LaTeX formatting
    response = response.replace("ˆ", "^")  # Replace ˆ with ^
    response = response.replace(""", '"').replace(""", '"')  # Replace curly quotes with straight quotes
    response = response.replace("'", "'").replace("'", "'")  # Replace curly apostrophes with straight apostrophes
    response = response.replace("\\", "")  # Remove unnecessary backslashes
    response = response.replace("????", "?").replace("???", "?").replace("??", "?")  # Remove excessive question marks
    
    # Generate document-based answers - IMPROVED VERSION
    def generate_answer_from_docs(question):
        relevant_chunks = retrieve_data(question)
        if not relevant_chunks:
            return "No relevant information found in the provided documents."
        
        # Use the full content of the retrieved chunks, not just the first sentence
        answer = " ".join([doc.page_content for doc in relevant_chunks])
        
        # If the answer is too long, limit it to a reasonable length
        if len(answer) > 1000:
            answer = answer[:1000] + "..."
            
        return answer
    
    # Reformat the response to ensure consistent structure with delimiter markers
    # First, let's ensure the response has consistent delimiters
    formatted_response = ""
    
    # Extract questions and answers using regex patterns
    question_blocks = re.findall(r"\*\*Q(\d+):\*\* (.*?)(?=\*\*Q\d+:|$)", response, re.DOTALL)
    
    for i, (q_num, content) in enumerate(question_blocks):
        q_num = int(q_num)
        
        # Extract the question text
        question_match = re.search(r"(.*?)(?:\*\*Bloom)", content, re.DOTALL)
        if question_match:
            question_text = question_match.group(1).strip()
        else:
            question_text = "Question text not found"
        
        # Extract bloom's level
        bloom_match = re.search(r"\*\*Bloom's Level:\*\* (.*?)(?:\n\*\*|\Z)", content, re.DOTALL)
        bloom_level = bloom_match.group(1).strip() if bloom_match else "Not specified"
        
        # Extract LLM answer
        llm_match = re.search(r"\*\*A\d+ \(LLM\):\*\* (.*?)(?:\n\*\*|\Z)", content, re.DOTALL)
        llm_answer = llm_match.group(1).strip() if llm_match else "No LLM answer available"
        
        # Generate document answers
        doc_answer = generate_answer_from_docs(question_text)
        
        # Calculate similarity
        similarity_score = calculate_similarity(llm_answer, doc_answer)
        
        # Format the output with consistent delimiters
        formatted_response += f"********\n**Q{q_num}:** {question_text}\n"
        formatted_response += f"********\n**Bloom's Level:** {bloom_level}\n"
        formatted_response += f"********\n**A{q_num} (LLM):** {llm_answer}\n"
        formatted_response += f"********\n**A{q_num} (From Document):** {doc_answer}\n"
        formatted_response += f"********\n**Similarity Score:** {similarity_score:.2f}\n"
    
    # Count generated questions for feedback
    generated_questions = formatted_response.count("**Q")
    
    if generated_questions < num_questions:
        formatted_response += f"\n\n⚠️ Note: Only {generated_questions} questions could be generated from the retrieved text."
    
    return formatted_response
