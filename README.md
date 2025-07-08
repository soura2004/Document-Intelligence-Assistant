# Document-Intelligence-Assistant


## 🎯 Core Features Implemented
1. Document Upload & Processing

PDF & TXT support with robust text extraction
Auto-summary generation (≤150 words) using TF-IDF-based extractive summarization
Section detection with intelligent content parsing
Keyword extraction for better document understanding

2. Ask Anything Mode

Free-form Q&A with contextual understanding
Document-grounded responses - no hallucination
Source justification for every answer
Conversation history tracking
Relevance scoring using cosine similarity

3. Challenge Me Mode

AI-generated questions (comprehension, inference, analysis)
Automated evaluation with detailed feedback
Performance scoring (0-100 scale)
Difficulty levels (easy, medium, hard)
Progress tracking with summary metrics

4. Advanced Capabilities

Memory handling for follow-up questions
Answer highlighting with supporting text snippets
Confidence scoring for answer reliability
Section-based analysis for better context
Real-time processing with progress indicators

## 🏗️ Technical Architecture
### Backend (Flask)
Document Processing: Text extraction, section parsing, keyword extraction
NLP Engine: TF-IDF vectorization, cosine similarity, sentence tokenization
Question Answering: Context-aware response generation
Challenge Generation: Logic-based question creation
Evaluation System: Automated answer scoring with detailed feedback

### Frontend (Streamlit)
Clean UI: Modern, intuitive interface with custom CSS
Real-time Updates: Live conversation tracking
Interactive Elements: File upload, Q&A interface, challenge mode
Performance Analytics: Score tracking and progress visualization
Responsive Design: Mobile-friendly layout

## 🚀 Setup Instructions

Install Dependencies:

pip install flask flask-cors PyPDF2 nltk scikit-learn spacy numpy requests streamlit
python -m spacy download en_core_web_sm

Create Files:

Save the Flask backend as app.py
Save the Streamlit frontend as streamlit_app.py


## Run the Application:

### Terminal 1 (Backend)
python app.py

### Terminal 2 (Frontend)  
streamlit run streamlit_app.py

Access: Open http://localhost:8501 in your browser

## 📊 Key Features Highlights

Document Grounding: Every answer includes specific references
No Hallucination: All responses are based on actual document content
Intelligent Parsing: Automatic section detection and content structuring
Performance Tracking: Comprehensive scoring and feedback system
Scalable Architecture: Clean separation of concerns, RESTful APIs

The system is production-ready with proper error handling, input validation, and a professional user interface. It can handle complex documents and provide meaningful insights while maintaining complete traceability to source content.
