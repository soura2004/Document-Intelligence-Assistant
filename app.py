
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import re
from datetime import datetime
import PyPDF2
import io
from typing import Dict, List, Tuple, Any
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

app = Flask(__name__)
CORS(app)

# Global variables to store document data
current_document = {
    'content': '',
    'title': '',
    'sections': [],
    'sentences': [],
    'keywords': [],
    'upload_time': None
}

conversation_history = []

class DocumentProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            self.nlp = None
            
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from uploaded PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    def process_document(self, content: str, title: str = "Document") -> Dict:
        """Process document and extract structured information"""
        # Clean and normalize text
        content = re.sub(r'\s+', ' ', content).strip()
        
        # Split into sentences
        sentences = sent_tokenize(content)
        
        # Extract sections (simple heuristic based on common patterns)
        sections = self._extract_sections(content)
        
        # Extract keywords
        keywords = self._extract_keywords(content)
        
        # Generate summary
        summary = self._generate_summary(content, sentences)
        
        return {
            'content': content,
            'title': title,
            'sections': sections,
            'sentences': sentences,
            'keywords': keywords,
            'summary': summary,
            'upload_time': datetime.now().isoformat()
        }
    
    def _extract_sections(self, content: str) -> List[Dict]:
        """Extract document sections based on common patterns"""
        sections = []
        
        # Pattern for numbered sections (1., 2., etc.)
        numbered_pattern = r'(\d+\.\s+[A-Z][^\n]*)'
        
        # Pattern for titled sections (Title Case followed by content)
        title_pattern = r'([A-Z][A-Za-z\s]{2,50})\n([A-Z][^.]*\.)'
        
        # Pattern for ALL CAPS sections
        caps_pattern = r'([A-Z\s]{3,50})\n([A-Z][^.]*\.)'
        
        matches = []
        
        # Find numbered sections
        for match in re.finditer(numbered_pattern, content):
            matches.append((match.start(), match.group(1), 'numbered'))
        
        # Find title sections
        for match in re.finditer(title_pattern, content):
            matches.append((match.start(), match.group(1), 'title'))
        
        # Find caps sections
        for match in re.finditer(caps_pattern, content):
            matches.append((match.start(), match.group(1), 'caps'))
        
        # Sort by position and create sections
        matches.sort(key=lambda x: x[0])
        
        for i, (pos, title, type_) in enumerate(matches):
            next_pos = matches[i + 1][0] if i + 1 < len(matches) else len(content)
            section_content = content[pos:next_pos].strip()
            
            sections.append({
                'title': title.strip(),
                'content': section_content,
                'type': type_,
                'position': pos
            })
        
        # If no clear sections found, create a single section
        if not sections:
            sections = [{
                'title': 'Main Content',
                'content': content,
                'type': 'main',
                'position': 0
            }]
        
        return sections
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract important keywords from the document"""
        # Simple keyword extraction using TF-IDF
        words = word_tokenize(content.lower())
        words = [word for word in words if word.isalpha() and word not in self.stop_words]
        
        # Get most frequent words
        word_freq = Counter(words)
        
        # Use TF-IDF for better keyword extraction
        try:
            vectorizer = TfidfVectorizer(max_features=20, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([content])
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Sort by TF-IDF score
            keyword_scores = list(zip(feature_names, tfidf_scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [kw[0] for kw in keyword_scores[:15]]
        except:
            # Fallback to frequency-based approach
            return [word for word, freq in word_freq.most_common(15)]
    
    def _generate_summary(self, content: str, sentences: List[str]) -> str:
        """Generate a concise summary of the document"""
        if len(sentences) <= 3:
            return content[:150] + "..." if len(content) > 150 else content
        
        try:
            # Simple extractive summarization using sentence ranking
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Calculate sentence importance based on average TF-IDF
            sentence_scores = []
            for i, sentence in enumerate(sentences):
                if len(sentence.split()) > 5:  # Only consider substantial sentences
                    avg_score = np.mean(tfidf_matrix[i].toarray())
                    sentence_scores.append((i, sentence, avg_score))
            
            # Sort by importance and select top sentences
            sentence_scores.sort(key=lambda x: x[2], reverse=True)
            
            # Select top 3-5 sentences for summary
            selected_sentences = sentence_scores[:min(5, len(sentence_scores))]
            selected_sentences.sort(key=lambda x: x[0])  # Sort by original order
            
            summary = " ".join([sent[1] for sent in selected_sentences])
            
            # Trim to 150 words
            words = summary.split()
            if len(words) > 150:
                summary = " ".join(words[:150]) + "..."
            
            return summary
        except:
            # Fallback to first few sentences
            summary = " ".join(sentences[:3])
            words = summary.split()
            if len(words) > 150:
                summary = " ".join(words[:150]) + "..."
            return summary

class QuestionAnswering:
    def __init__(self):
        self.processor = DocumentProcessor()
        
    def find_relevant_context(self, question: str, document: Dict) -> List[Tuple[str, float, str]]:
        """Find relevant sentences/sections for answering the question"""
        all_texts = []
        sources = []
        
        # Include sentences with section context
        for section in document['sections']:
            section_sentences = sent_tokenize(section['content'])
            for sentence in section_sentences:
                if len(sentence.split()) > 3:  # Only substantial sentences
                    all_texts.append(sentence)
                    sources.append(f"Section '{section['title']}'")
        
        if not all_texts:
            return []
        
        try:
            # Use TF-IDF to find most relevant sentences
            vectorizer = TfidfVectorizer(stop_words='english')
            doc_vectors = vectorizer.fit_transform(all_texts + [question])
            
            # Calculate similarity between question and each sentence
            question_vector = doc_vectors[-1]
            similarities = cosine_similarity(question_vector, doc_vectors[:-1]).flatten()
            
            # Get top relevant sentences
            relevant_indices = np.argsort(similarities)[::-1][:5]
            relevant_contexts = []
            
            for idx in relevant_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    relevant_contexts.append((
                        all_texts[idx], 
                        similarities[idx], 
                        sources[idx]
                    ))
            
            return relevant_contexts
        except:
            # Fallback to simple keyword matching
            question_words = set(word_tokenize(question.lower()))
            relevant_contexts = []
            
            for i, text in enumerate(all_texts):
                text_words = set(word_tokenize(text.lower()))
                overlap = len(question_words.intersection(text_words))
                if overlap > 0:
                    relevant_contexts.append((text, overlap / len(question_words), sources[i]))
            
            relevant_contexts.sort(key=lambda x: x[1], reverse=True)
            return relevant_contexts[:3]
    
    def answer_question(self, question: str, document: Dict) -> Dict:
        """Answer a question based on the document content"""
        relevant_contexts = self.find_relevant_context(question, document)
        
        if not relevant_contexts:
            return {
                'answer': "I couldn't find relevant information in the document to answer this question.",
                'confidence': 0.0,
                'sources': [],
                'justification': "No relevant content found in the document."
            }
        
        # Generate answer based on most relevant context
        best_context = relevant_contexts[0]
        
        # Simple answer generation (in a real application, you'd use an LLM here)
        answer = self._generate_answer(question, best_context[0])
        
        sources = [ctx[2] for ctx in relevant_contexts[:3]]
        justification = f"Based on content from {best_context[2]}: '{best_context[0][:100]}...'"
        
        return {
            'answer': answer,
            'confidence': best_context[1],
            'sources': sources,
            'justification': justification,
            'supporting_text': best_context[0]
        }
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate an answer based on context (simplified implementation)"""
        # This is a simplified implementation
        # In a real application, you would use an LLM or more sophisticated NLP
        
        question_lower = question.lower()
        
        # Simple pattern matching for common question types
        if any(word in question_lower for word in ['what', 'define', 'definition']):
            return f"Based on the document: {context}"
        elif any(word in question_lower for word in ['why', 'how', 'explain']):
            return f"According to the document: {context}"
        elif any(word in question_lower for word in ['when', 'where', 'who']):
            return f"The document indicates: {context}"
        else:
            return f"From the document: {context}"

class ChallengeGenerator:
    def __init__(self):
        self.processor = DocumentProcessor()
        
    def generate_challenges(self, document: Dict) -> List[Dict]:
        """Generate logic-based questions from the document"""
        challenges = []
        
        # Generate different types of questions
        challenges.extend(self._generate_comprehension_questions(document))
        challenges.extend(self._generate_inference_questions(document))
        challenges.extend(self._generate_analysis_questions(document))
        
        # Return top 3 challenges
        return challenges[:3]
    
    def _generate_comprehension_questions(self, document: Dict) -> List[Dict]:
        """Generate comprehension-based questions"""
        questions = []
        
        for section in document['sections']:
            sentences = sent_tokenize(section['content'])
            if len(sentences) > 2:
                # Find sentences with specific facts or statements
                for sentence in sentences:
                    if any(word in sentence.lower() for word in ['therefore', 'because', 'due to', 'result', 'caused by']):
                        question = f"According to the section '{section['title']}', what relationship is described?"
                        questions.append({
                            'question': question,
                            'type': 'comprehension',
                            'section': section['title'],
                            'reference_text': sentence,
                            'difficulty': 'medium'
                        })
        
        return questions
    
    def _generate_inference_questions(self, document: Dict) -> List[Dict]:
        """Generate inference-based questions"""
        questions = []
        
        # Look for patterns that require inference
        for section in document['sections']:
            if len(section['content']) > 200:
                question = f"Based on the information in '{section['title']}', what can you infer about the main concept discussed?"
                questions.append({
                    'question': question,
                    'type': 'inference',
                    'section': section['title'],
                    'reference_text': section['content'][:200] + "...",
                    'difficulty': 'hard'
                })
        
        return questions
    
    def _generate_analysis_questions(self, document: Dict) -> List[Dict]:
        """Generate analysis-based questions"""
        questions = []
        
        # Generate questions about document structure and key concepts
        if document['keywords']:
            top_keywords = document['keywords'][:5]
            question = f"How do these key concepts relate to each other in the document: {', '.join(top_keywords)}?"
            questions.append({
                'question': question,
                'type': 'analysis',
                'section': 'Overall Document',
                'reference_text': f"Key concepts: {', '.join(top_keywords)}",
                'difficulty': 'hard'
            })
        
        return questions
    
    def evaluate_answer(self, user_answer: str, challenge: Dict, document: Dict) -> Dict:
        """Evaluate user's answer to a challenge question"""
        # Simple evaluation based on keyword matching and content relevance
        user_words = set(word_tokenize(user_answer.lower()))
        reference_words = set(word_tokenize(challenge['reference_text'].lower()))
        
        overlap = len(user_words.intersection(reference_words))
        total_words = len(user_words.union(reference_words))
        
        if total_words == 0:
            similarity = 0
        else:
            similarity = overlap / total_words
        
        # Determine score based on similarity and answer length
        if similarity > 0.3 and len(user_answer.split()) > 5:
            score = min(100, int(similarity * 100) + 20)
            feedback = "Good answer! You've captured key concepts from the document."
        elif similarity > 0.2:
            score = min(80, int(similarity * 100) + 10)
            feedback = "Decent answer, but could be more comprehensive."
        else:
            score = max(20, int(similarity * 100))
            feedback = "Your answer needs more support from the document content."
        
        return {
            'score': score,
            'feedback': feedback,
            'reference_section': challenge['section'],
            'key_points_missed': self._identify_missed_points(user_answer, challenge),
            'justification': f"Evaluated based on content from {challenge['section']}"
        }
    
    def _identify_missed_points(self, user_answer: str, challenge: Dict) -> List[str]:
        """Identify key points the user might have missed"""
        # Simple implementation - in practice, you'd use more sophisticated NLP
        reference_sentences = sent_tokenize(challenge['reference_text'])
        user_words = set(word_tokenize(user_answer.lower()))
        
        missed_points = []
        for sentence in reference_sentences:
            sentence_words = set(word_tokenize(sentence.lower()))
            if len(sentence_words.intersection(user_words)) < 2:
                missed_points.append(sentence[:100] + "...")
        
        return missed_points[:3]

# Initialize processors
doc_processor = DocumentProcessor()
qa_system = QuestionAnswering()
challenge_generator = ChallengeGenerator()

# Flask routes
@app.route('/upload', methods=['POST'])
def upload_document():
    """Handle document upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Extract text based on file type
        if file.filename.lower().endswith('.pdf'):
            content = doc_processor.extract_text_from_pdf(file)
        elif file.filename.lower().endswith('.txt'):
            content = file.read().decode('utf-8')
        else:
            return jsonify({'error': 'Unsupported file type'}), 400
        
        # Process document
        processed_doc = doc_processor.process_document(content, file.filename)
        
        # Store globally
        global current_document
        current_document = processed_doc
        
        # Clear conversation history
        global conversation_history
        conversation_history = []
        
        return jsonify({
            'message': 'Document uploaded successfully',
            'title': processed_doc['title'],
            'summary': processed_doc['summary'],
            'sections': len(processed_doc['sections']),
            'keywords': processed_doc['keywords'][:10]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle user questions"""
    try:
        data = request.json
        question = data.get('question', '')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        if not current_document['content']:
            return jsonify({'error': 'No document uploaded'}), 400
        
        # Get answer
        result = qa_system.answer_question(question, current_document)
        
        # Store in conversation history
        conversation_history.append({
            'question': question,
            'answer': result['answer'],
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/challenge', methods=['GET'])
def get_challenges():
    """Generate challenge questions"""
    try:
        if not current_document['content']:
            return jsonify({'error': 'No document uploaded'}), 400
        
        challenges = challenge_generator.generate_challenges(current_document)
        
        return jsonify({
            'challenges': challenges,
            'count': len(challenges)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/evaluate', methods=['POST'])
def evaluate_answer():
    """Evaluate user's answer to a challenge"""
    try:
        data = request.json
        user_answer = data.get('answer', '')
        challenge_data = data.get('challenge', {})
        
        if not user_answer or not challenge_data:
            return jsonify({'error': 'Missing answer or challenge data'}), 400
        
        result = challenge_generator.evaluate_answer(user_answer, challenge_data, current_document)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Get current application status"""
    return jsonify({
        'document_loaded': bool(current_document['content']),
        'document_title': current_document['title'],
        'conversation_length': len(conversation_history),
        'upload_time': current_document['upload_time']
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
