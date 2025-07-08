import streamlit as st
import requests
import json
import time
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Document Intelligence Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-bottom: 2rem;
        border-radius: 10px;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        background: #f8f9fa;
    }
    
    .challenge-card {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
    }
    
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Backend URL
BACKEND_URL = "http://localhost:5000"

# Initialize session state
if 'document_loaded' not in st.session_state:
    st.session_state.document_loaded = False
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'current_challenges' not in st.session_state:
    st.session_state.current_challenges = []
if 'challenge_results' not in st.session_state:
    st.session_state.challenge_results = []

def check_backend_status():
    """Check if backend is running"""
    try:
        response = requests.get(f"{BACKEND_URL}/status", timeout=5)
        return response.status_code == 200
    except:
        return False

def upload_document(file):
    """Upload document to backend"""
    try:
        files = {'file': file}
        response = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=30)
        return response.json(), response.status_code == 200
    except Exception as e:
        return {'error': str(e)}, False

def ask_question(question):
    """Send question to backend"""
    try:
        data = {'question': question}
        response = requests.post(f"{BACKEND_URL}/ask", json=data, timeout=30)
        return response.json(), response.status_code == 200
    except Exception as e:
        return {'error': str(e)}, False

def get_challenges():
    """Get challenge questions from backend"""
    try:
        response = requests.get(f"{BACKEND_URL}/challenge", timeout=30)
        return response.json(), response.status_code == 200
    except Exception as e:
        return {'error': str(e)}, False

def evaluate_answer(answer, challenge):
    """Evaluate challenge answer"""
    try:
        data = {'answer': answer, 'challenge': challenge}
        response = requests.post(f"{BACKEND_URL}/evaluate", json=data, timeout=30)
        return response.json(), response.status_code == 200
    except Exception as e:
        return {'error': str(e)}, False

# Main app
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📚 Document Intelligence Assistant</h1>
        <p>Upload documents and engage in intelligent Q&A with comprehension challenges</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check backend status
    if not check_backend_status():
        st.error("⚠️ Backend server is not running. Please start the Flask backend first.")
        st.code("python app.py", language="bash")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Document Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a document",
            type=['pdf', 'txt'],
            help="Upload a PDF or TXT file containing structured content"
        )
        
        if uploaded_file is not None:
            if st.button("📤 Upload Document", type="primary"):
                with st.spinner("Processing document..."):
                    result, success = upload_document(uploaded_file)
                    
                    if success:
                        st.session_state.document_loaded = True
                        st.session_state.conversation_history = []
                        st.session_state.current_challenges = []
                        st.session_state.challenge_results = []
                        
                        st.success("Document uploaded successfully!")
                        st.rerun()
                    else:
                        st.error(f"Upload failed: {result.get('error', 'Unknown error')}")
        
        # Document status
        if st.session_state.document_loaded:
            st.success("✅ Document loaded")
            
            # Get document info
            status_result, _ = requests.get(f"{BACKEND_URL}/status").json(), True
            if status_result:
                st.info(f"**Document:** {status_result.get('document_title', 'Unknown')}")
                st.info(f"**Conversations:** {len(st.session_state.conversation_history)}")
        
        st.markdown("---")
        
        # Instructions
        st.header("💡 How to Use")
        st.markdown("""
        1. **Upload** a PDF or TXT document
        2. **Ask Anything** - Get answers with justifications
        3. **Challenge Mode** - Test your comprehension
        4. All answers are grounded in your document
        """)
    
    # Main content area
    if not st.session_state.document_loaded:
        st.info("👆 Please upload a document to get started")
        
        # Demo section
        st.header("🚀 Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Ask Anything Mode")
            st.markdown("""
            - Ask free-form questions about your document
            - Get contextual answers with justifications
            - Reference specific sections and paragraphs
            - Maintain conversation history
            """)
        
        with col2:
            st.subheader("🧠 Challenge Mode")
            st.markdown("""
            - AI-generated comprehension questions
            - Logic and inference-based challenges
            - Automated answer evaluation
            - Detailed feedback with document references
            """)
        
        return
    
    # Show document summary
    try:
        response = requests.get(f"{BACKEND_URL}/status")
        if response.status_code == 200:
            # Get fresh document info for summary
            pass
    except:
        pass
    
    # Mode selection
    st.header("🎯 Choose Your Mode")
    
    mode = st.radio(
        "Select interaction mode:",
        ["Ask Anything", "Challenge Me"],
        horizontal=True
    )
    
    if mode == "Ask Anything":
        st.subheader("💬 Ask Anything About Your Document")
        
        # Chat interface
        question = st.text_input(
            "Your question:",
            placeholder="What are the main findings of this document?",
            key="question_input"
        )
        
        if st.button("🔍 Get Answer", type="primary"):
            if question:
                with st.spinner("Analyzing document..."):
                    result, success = ask_question(question)
                    
                    if success:
                        # Add to conversation history
                        st.session_state.conversation_history.append({
                            'question': question,
                            'answer': result['answer'],
                            'justification': result.get('justification', ''),
                            'supporting_text': result.get('supporting_text', ''),
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        # Clear input
                        st.rerun()
                    else:
                        st.error(f"Error: {result.get('error', 'Unknown error')}")
        
        # Display conversation history
        if st.session_state.conversation_history:
            st.subheader("📜 Conversation History")
            
            for i, conv in enumerate(reversed(st.session_state.conversation_history)):
                with st.expander(f"Q: {conv['question'][:50]}...", expanded=(i == 0)):
                    st.markdown(f"**Question:** {conv['question']}")
                    st.markdown(f"**Answer:** {conv['answer']}")
                    
                    if conv.get('justification'):
                        st.info(f"**Justification:** {conv['justification']}")
                    
                    if conv.get('supporting_text'):
                        st.markdown("**Supporting Text:**")
                        st.code(conv['supporting_text'], language="text")
    
    elif mode == "Challenge Me":
        st.subheader("🧠 Challenge Mode")
        
        # Generate challenges button
        if st.button("🎲 Generate New Challenges", type="primary"):
            with st.spinner("Generating challenges..."):
                result, success = get_challenges()
                
                if success:
                    st.session_state.current_challenges = result['challenges']
                    st.session_state.challenge_results = []
                    st.rerun()
                else:
                    st.error(f"Error: {result.get('error', 'Unknown error')}")
        
        # Display challenges
        if st.session_state.current_challenges:
            st.success(f"✅ Generated {len(st.session_state.current_challenges)} challenges")
            
            for i, challenge in enumerate(st.session_state.current_challenges):
                st.markdown(f"### Challenge {i+1}")
                
                # Challenge card
                st.markdown(f"""
                <div class="challenge-card">
                    <h4>🎯 {challenge['type'].title()} Question</h4>
                    <p><strong>Question:</strong> {challenge['question']}</p>
                    <p><strong>Section:</strong> {challenge['section']}</p>
                    <p><strong>Difficulty:</strong> {challenge['difficulty'].title()}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Answer input
                user_answer = st.text_area(
                    f"Your answer for Challenge {i+1}:",
                    key=f"answer_{i}",
                    placeholder="Type your answer here...",
                    height=100
                )
                
                # Submit button
                if st.button(f"Submit Answer {i+1}", key=f"submit_{i}"):
                    if user_answer.strip():
                        with st.spinner("Evaluating your answer..."):
                            result, success = evaluate_answer(user_answer, challenge)
                            
                            if success:
                                # Store result
                                if len(st.session_state.challenge_results) <= i:
                                    st.session_state.challenge_results.extend([None] * (i + 1 - len(st.session_state.challenge_results)))
                                
                                st.session_state.challenge_results[i] = {
                                    'challenge': challenge,
                                    'user_answer': user_answer,
                                    'result': result
                                }
                                
                                st.rerun()
                            else:
                                st.error(f"Evaluation failed: {result.get('error', 'Unknown error')}")
                    else:
                        st.warning("Please provide an answer before submitting.")
                
                # Show result if available
                if len(st.session_state.challenge_results) > i and st.session_state.challenge_results[i]:
                    result_data = st.session_state.challenge_results[i]
                    result = result_data['result']
                    
                    # Score display
                    score = result['score']
                    if score >= 80:
                        st.success(f"🎉 Excellent! Score: {score}/100")
                    elif score >= 60:
                        st.info(f"👍 Good work! Score: {score}/100")
                    else:
                        st.warning(f"🤔 Keep trying! Score: {score}/100")
                    
                    # Detailed feedback
                    st.markdown(f"**Feedback:** {result['feedback']}")
                    st.markdown(f"**Reference Section:** {result['reference_section']}")
                    st.markdown(f"**Justification:** {result['justification']}")
                    
                    # Show missed points
                    if result.get('key_points_missed'):
                        st.markdown("**Key points to consider:**")
                        for point in result['key_points_missed']:
                            st.markdown(f"- {point}")
                
                st.markdown("---")
            
            # Overall performance summary
            if st.session_state.challenge_results:
                completed_challenges = [r for r in st.session_state.challenge_results if r is not None]
                
                if completed_challenges:
                    st.subheader("📊 Performance Summary")
                    
                    total_score = sum(r['result']['score'] for r in completed_challenges)
                    avg_score = total_score / len(completed_challenges)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Completed", f"{len(completed_challenges)}/{len(st.session_state.current_challenges)}")
                    
                    with col2:
                        st.metric("Average Score", f"{avg_score:.1f}/100")
                    
                    with col3:
                        if avg_score >= 80:
                            st.metric("Performance", "Excellent 🎉")
                        elif avg_score >= 60:
                            st.metric("Performance", "Good 👍")
                        else:
                            st.metric("Performance", "Needs Improvement 📚")

# Additional utility functions
def display_document_info():
    """Display document information and summary"""
    try:
        response = requests.get(f"{BACKEND_URL}/status")
        if response.status_code == 200:
            status = response.json()
            
            if status.get('document_loaded'):
                st.subheader("📄 Document Summary")
                
                # This would need to be implemented to get summary from backend
                # For now, we'll show basic info
                st.info(f"Document: {status.get('document_title', 'Unknown')}")
                st.info(f"Upload time: {status.get('upload_time', 'Unknown')}")
                
    except Exception as e:
        st.error(f"Error fetching document info: {e}")

if __name__ == "__main__":
    main()
