"""
ingest.py - Run this ONCE locally to generate embeddings.json
Usage: python scripts/ingest.py
Output: public/embeddings.json
"""

import json
import math
import re
import os

# ─── Resume text (hardcoded from krishna-cv.pdf) ────────────────────────────
RESUME_TEXT = """
KRISHNA SINGH
Aspiring Machine Learning Engineer & Full Stack Developer
Email: ksingh75224@gmail.com | Phone: +91-9628251125 | Location: Kanpur Nagar, Uttar Pradesh, India
LinkedIn: linkedin.com/in/krishna-singh-108b01366 | GitHub: github.com/Krishna329-dot

PROFESSIONAL SUMMARY
Aspiring Machine Learning Engineer with a strong foundation in Python, data structures, and real-world project development.
Passionate about building intelligent systems, AI-powered applications, and solving practical problems using cutting-edge ML
and web technologies. Experienced in RAG-based systems, voice assistants, IoT sensing, and full-stack development.

TECHNICAL SKILLS
Languages: Python (Intermediate), JavaScript (Advanced), TypeScript (Intermediate), C++ (Beginner)
Frontend: HTML5 (Advanced), CSS3 (Advanced), React (Intermediate), Next.js (Intermediate), Bootstrap (Intermediate)
Backend: Node.js (Intermediate), FastAPI (Intermediate)
AI / ML: LangChain, FAISS, Ollama, Scikit-learn, NLP, RAG Systems, Machine Learning (Beginner)
Databases: MongoDB (Beginner)
Tools & DevOps: Git (Advanced), GitHub (Advanced), Linux (Beginner)
Soft Skills: Problem Solving & Logical Thinking, Team Collaboration, Quick Learner & Adaptable, Technical Support

WORK EXPERIENCE
Hackathon Project Developer
LifeDrop - Blood Donation Network | Jan 2025 - Jun 2025 | On-site, India
- Built a digital platform to connect blood donors, hospitals, and patients during emergencies.
- Developed AI-based donor matching system for efficient and quick emergency response.
- Implemented real-time emergency alert system for blood requests.
- Designed user-friendly admin dashboard for monitoring and request management.
- Integrated location-based services for finding nearby blood camps and banks.
- Received Certificate of Completion for successfully delivering project tasks.

PROJECTS

College Admission AI Chatbot
Stack: Python, LangChain, FAISS, Ollama (Phi-3), FastAPI, HTML5, CSS3, JavaScript
GitHub: github.com/Krishna329-dot/college-admission-chat-bot-
- Built a RAG-based AI chatbot acting as a College Reception Manager for student queries.
- Integrated LangChain + FAISS vector database for accurate PDF-based knowledge retrieval.
- Used Ollama (Phi-3:mini) as a local LLM with no cloud API dependency.
- Implemented built-in Text-to-Speech using Web Speech API for voice interaction.
- Designed a modern Glassmorphism UI with FastAPI backend.

Venom AI - AI Assistant
Stack: Python
GitHub: github.com/Krishna329-dot/venom-AI
- Built a powerful AI assistant using Python with context-aware response generation.
- Designed modular and extensible architecture for future AI integrations.
- Implemented natural language interaction for intelligent command processing.

Real-time Voice Assistant (JARVIS)
Stack: Python, SpeechRecognition, pyttsx3
GitHub: github.com/Krishna329-dot/real-time-voice-assistant-
- Developed a JARVIS-inspired voice-controlled assistant with real-time speech recognition.
- Automated system tasks including opening applications, web search, and command execution.
- Applied NLP concepts for natural language understanding and response generation.

LifeDrop - Blood Donation Network
Stack: Python, JavaScript, HTML5, CSS3
GitHub: github.com/Krishna329-dot/hackthon
- Hackathon project: digital platform connecting donors, hospitals, and patients.
- Implemented AI-based donor matching and real-time emergency alert system.
- Received Certificate of Completion for successful project delivery.

Fire Alarm Detection System
Stack: Python, C++
GitHub: github.com/Krishna329-dot/fire-alarm-dect
- Automated fire hazard detection system with instant alert triggering.
- Integrated sensor-based monitoring with hardware and software implementation.

ESP32 CSI Sensing
Stack: C++, Python
GitHub: github.com/Krishna329-dot/esp32-csi-sensing
- Used ESP32 microcontroller to capture WiFi Channel State Information (CSI) data.
- Built human presence detection and gesture recognition using WiFi signals.
- Applied signal processing and embedded systems programming concepts.

Email Spam Classifier
Stack: Python, Scikit-learn, NLP
- Developed ML model to classify emails as spam or not spam using NLP techniques.
- Used TF-IDF vectorization and trained Naive Bayes / Logistic Regression models.
- Built a simple interface for real-time spam prediction with high accuracy.

EDUCATION
Bachelor of Computer Applications (BCA)
Allenhouse Institute of Technology, Kanpur | 2025 - 2028

CONTACT & LINKS
Email: ksingh75224@gmail.com
Phone: +91-9628251125
LinkedIn: https://www.linkedin.com/in/krishna-singh-108b01366
GitHub: https://github.com/Krishna329-dot
Resume/CV: https://printfolio.vercel.app/krishna-cv.pdf
Portfolio: https://printfolio.vercel.app

LANGUAGES
Hindi (Native) | English (Intermediate)

AVAILABILITY
Open to internships, freelance projects, and full-time opportunities in ML/AI and Full Stack Development.
"""

# ─── Simple text chunker ─────────────────────────────────────────────────────
def chunk_text(text, chunk_size=300, overlap=50):
    """Split text into overlapping chunks by words."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words)
        if chunk.strip():
            chunks.append(chunk.strip())
        i += chunk_size - overlap
    return chunks

# ─── TF-IDF style simple embedding (no external ML library needed) ───────────
def simple_embedding(text, vocab):
    """Create a simple TF-IDF-like vector for a text given a vocabulary."""
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    word_count = {}
    for w in words:
        word_count[w] = word_count.get(w, 0) + 1
    
    total = len(words) if words else 1
    vector = []
    for term in vocab:
        tf = word_count.get(term, 0) / total
        vector.append(tf)
    return vector

def normalize(vec):
    """L2 normalize a vector."""
    magnitude = math.sqrt(sum(x * x for x in vec))
    if magnitude == 0:
        return vec
    return [x / magnitude for x in vec]

def build_vocab(chunks, max_terms=500):
    """Build vocabulary from chunks, filtering stopwords."""
    stopwords = set([
        'the','a','an','and','or','but','in','on','at','to','for','of','with',
        'is','are','was','were','be','been','being','have','has','had','do',
        'does','did','will','would','could','should','may','might','shall',
        'that','this','these','those','i','you','he','she','it','we','they',
        'me','him','her','us','them','my','your','his','its','our','their',
        'from','by','as','up','out','if','about','into','through','during',
        'before','after','above','below','between','each','both','few','more',
        'most','other','some','such','no','not','only','same','so','than',
        'too','very','just','also','can','all','any','over','per','using'
    ])
    
    word_freq = {}
    for chunk in chunks:
        words = re.findall(r'\b\w+\b', chunk.lower())
        for w in words:
            if w not in stopwords and len(w) > 2:
                word_freq[w] = word_freq.get(w, 0) + 1
    
    # Sort by frequency, take top max_terms
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    vocab = [w for w, _ in sorted_words[:max_terms]]
    return vocab

# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    print("Chunking resume text...")
    chunks = chunk_text(RESUME_TEXT, chunk_size=150, overlap=30)
    print(f"  Created {len(chunks)} chunks")

    print("Building vocabulary...")
    vocab = build_vocab(chunks, max_terms=400)
    print(f"  Vocabulary size: {len(vocab)}")

    print("Generating embeddings...")
    embeddings_data = []
    for i, chunk in enumerate(chunks):
        vec = simple_embedding(chunk, vocab)
        vec = normalize(vec)
        embeddings_data.append({
            "id": i,
            "text": chunk,
            "embedding": vec
        })
    print(f"  Generated {len(embeddings_data)} embeddings")

    output = {
        "vocab": vocab,
        "chunks": embeddings_data
    }

    # Output path
    out_path = os.path.join(os.path.dirname(__file__), "../public/embeddings.json")
    out_path = os.path.normpath(out_path)
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)
    
    size_kb = os.path.getsize(out_path) / 1024
    print(f"\nDone! Saved to: {out_path}")
    print(f"File size: {size_kb:.1f} KB")
    print("\nAb git push karo — embeddings.json Vercel pe static file ki tarah serve hoga.")

if __name__ == "__main__":
    main()
