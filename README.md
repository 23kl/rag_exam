🧠 Smart QnA Generator
Smart QnA Generator is an intelligent system designed to extract content from PDF documents and generate highly relevant, context-aware questions and answers. It leverages advanced language models and vector-based similarity checks to ensure quality and relevance, making it a valuable tool for educational, research, and training purposes.

🎥 Video Demo



https://github.com/user-attachments/assets/d4d51406-b938-445c-a92c-b137db53cd35


🚀 Features
📄 PDF Content Extraction
Parses both text and images from uploaded PDF documents for comprehensive content understanding.

💬 Contextual Question Generation
Utilizes DeepSeek combined with LangChain to generate high-quality, meaningful questions based on extracted document content.

✅ Answer Validation
Employs Chroma vector database to compare generated answers with the source text and ensure contextual accuracy.

🖼 Image-Enhanced Answers
Enhances answers by matching keywords to relevant images extracted from the same PDF, offering visual context alongside text.

🛠 Technology Used
Frontend:
React.js
HTML5, CSS3, JavaScript

Backend:
Node.js
Express.js

AI/ML Tools:
DeepSeek LLM
LangChain for chaining context
Chroma for vector similarity
LoRA/QLoRA fine-tuning (used during model optimization)

Document Processing:
PyMuPDF (for PDF parsing)
OpenCV (for image alignment)


