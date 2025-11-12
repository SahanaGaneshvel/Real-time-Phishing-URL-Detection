import os
import logging
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# LangChain imports for RAG
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import numpy as np

# Gemini AI
import google.generativeai as genai

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load .env - explicitly specify the path to ensure it's found
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    logger.info(f"✅ Loading .env from: {env_path}")
else:
    # Try loading from current directory as fallback
    load_dotenv()
    logger.warning(f"⚠️ .env file not found at: {env_path}, trying default location")

# --- Gemini / RAG integration patch (replace existing Gemini init + helpers) ---

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
# Preferred model names in order (update if you want a different preference)
PREFERRED_MODELS = [
    "models/gemini-2.5-pro",
    "models/gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "models/gemini-pro-latest",
    "models/gemini-flash-latest"
]

gemini_model = None
gemini_model_name = None

def configure_genai():
    """Configure the genai client for the running environment."""
    if not GEMINI_API_KEY:
        logger.warning("⚠️ No GEMINI_API_KEY found in environment.")
        return False

    # Configure SDK to use the default endpoint; the sdk will pick the correct service surface.
    # But we explicitly set transport to rest for compatibility in this app.
    try:
        genai.configure(api_key=GEMINI_API_KEY, transport="rest")
        logger.info("✅ genai.configure() succeeded")
        return True
    except Exception as e:
        logger.exception("❌ genai.configure() failed: %s", e)
        return False

def pick_and_init_model():
    """Pick a working model from the account and initialize gemini_model."""
    global gemini_model, gemini_model_name

    if not configure_genai():
        gemini_model = None
        gemini_model_name = None
        return None

    try:
        models = [m.name for m in genai.list_models()]
    except Exception as e:
        logger.warning("⚠️ list_models() failed: %s — will attempt without pref list", e)
        models = []

    # prefer the preferred list if available; otherwise pick first gemini-like model
    chosen = None
    for cand in PREFERRED_MODELS:
        if cand in models:
            chosen = cand
            break

    if not chosen:
        for m in models:
            if "gemini" in m.lower() or "gemma" in m.lower():
                chosen = m
                break

    # fallback to a safe local mode
    if not chosen:
        logger.warning("⚠️ No suitable Gemini model found in account. Chatbot will use fallback text responses.")
        gemini_model = None
        gemini_model_name = None
        return None

    try:
        # Use the model string directly (the SDK accepts both short and fully-qualified names)
        gemini_model = genai.GenerativeModel(chosen)
        gemini_model_name = chosen
        logger.info("✅ Initialized Gemini model: %s", chosen)
        return chosen
    except Exception as e:
        logger.exception("❌ Failed initializing Gemini model '%s': %s", chosen, e)
        gemini_model = None
        gemini_model_name = None
        return None

# initialize now at import/startup
pick_and_init_model()

def generate_with_gemini(prompt_text, model=None, timeout_seconds=30):
    """Call Gemini and return text; raises or returns None on failure."""
    if model is None:
        model = gemini_model
    if not model:
        return None

    try:
        resp = model.generate_content(prompt_text)
        if hasattr(resp, "text") and resp.text:
            return resp.text
        if hasattr(resp, "candidates") and resp.candidates:
            parts = resp.candidates[0].content.parts
            if parts and hasattr(parts[0], "text"):
                return parts[0].text
            return str(resp.candidates[0])
        return str(resp)
    except Exception as e:
        logger.exception("❌ Gemini generate_content failed: %s", e)
        return None


# Flask app
app = Flask(__name__)
CORS(app)
logger.info("🚀 Starting RAG Backend...")

# Global variables
vectorstore = None
embeddings_model = None

class SimpleEmbeddings:
    """Simple embedding class that creates basic embeddings without external dependencies"""
    
    def __init__(self, model_name="simple"):
        self.model_name = model_name
        logger.info(f"✅ Simple embeddings initialized")
    
    def __call__(self, text):
        """Allow the embedding instance to be used as a callable (compat with LangChain)."""
        return self.embed_query(text)
    
    def embed_documents(self, texts):
        """Create simple embeddings for documents"""
        embeddings = []
        for text in texts:
            # Create a simple hash-based embedding
            embedding = self._text_to_embedding(text)
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text):
        """Create embedding for a query"""
        return self._text_to_embedding(text)
    
    def _text_to_embedding(self, text):
        """Convert text to a simple embedding vector"""
        # Create a simple embedding based on text characteristics
        text_lower = text.lower()
        
        # Create a 384-dimensional vector (standard embedding size)
        embedding = np.zeros(384)
        
        # Simple features based on text content
        embedding[0] = len(text) / 1000.0  # Normalized length
        embedding[1] = text.count(' ') / max(len(text), 1)  # Space ratio
        embedding[2] = text.count('.') / max(len(text), 1)  # Period ratio
        embedding[3] = text.count('!') / max(len(text), 1)  # Exclamation ratio
        embedding[4] = text.count('?') / max(len(text), 1)  # Question ratio
        
        # Add some randomness based on text hash
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        for i in range(5, min(20, len(embedding))):
            embedding[i] = int(text_hash[i-5:i-4], 16) / 16.0
        
        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.tolist()

def load_documents_from_data_folder():
    """Load documents from the data/ folder"""
    documents = []
    data_folder = Path("data")
    
    if not data_folder.exists():
        logger.warning("⚠️ Data folder not found, using sample documents")
        return [
            "Phishing is a type of cyber attack where attackers trick users into revealing sensitive information such as passwords, credit card numbers, or personal data.",
            "Common phishing techniques include email spoofing, website cloning, social engineering, spear phishing, and whaling attacks.",
            "How to identify phishing emails: Check sender email address carefully, look for spelling and grammar errors, be suspicious of urgent requests for personal information, hover over links to see actual destination, verify requests through alternative communication channels.",
            "How to identify phishing websites: Check the URL for misspellings or suspicious domains, look for HTTPS certificate validity, verify the website's security indicators, be cautious of websites asking for excessive personal information.",
            "Prevention measures: Use multi-factor authentication, keep software and browsers updated, use reputable antivirus software, educate employees about phishing tactics, implement email filtering solutions.",
            "What to do if you fall victim to phishing: Change passwords immediately, contact your bank or credit card company, monitor accounts for suspicious activity, report the incident to relevant authorities, update security software."
        ]
    
    # Walk through all files in data folder
    for file_path in data_folder.rglob("*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                if content:
                    documents.append(content)
                    logger.info(f"📄 Loaded document: {file_path}")
        except Exception as e:
            logger.error(f"❌ Error loading {file_path}: {str(e)}")
    
    logger.info(f"📚 Total documents loaded: {len(documents)}")
    return documents

def initialize_vectorstore():
    """Initialize or load the FAISS vectorstore"""
    global vectorstore, embeddings_model
    
    try:
        # Try to load existing vectorstore
        if Path("phishing_knowledge_db").exists():
            embeddings_model = SimpleEmbeddings()
            # Allow deserialization for trusted local files
            vectorstore = FAISS.load_local(
                "phishing_knowledge_db", 
                embeddings_model,
                allow_dangerous_deserialization=True
            )
            logger.info("✅ Loaded existing vectorstore from disk")
            return True
    except Exception as e:
        logger.warning(f"⚠️ Could not load existing vectorstore: {str(e)}")
    
    # Create new vectorstore
    logger.info("🔄 Creating new vectorstore...")
    documents = load_documents_from_data_folder()
    
    if not documents:
        logger.warning("⚠️ No documents to process, creating empty vectorstore")
        documents = [""]
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    # Convert strings to Document objects
    docs = []
    for i, doc_content in enumerate(documents):
        if doc_content.strip():
            docs.append(Document(page_content=doc_content, metadata={"source": f"doc_{i}"}))
    
    chunks = text_splitter.split_documents(docs)
    logger.info(f"📝 Split documents into {len(chunks)} chunks")
    
    # Create embeddings and vectorstore
    embeddings_model = SimpleEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings_model)
    
    # Save vectorstore
    vectorstore.save_local("phishing_knowledge_db")
    logger.info("✅ FAISS vectorstore created and saved successfully")
    
    return True

def generate_response_with_rag(question):
    """Generate response using RAG (Retrieval-Augmented Generation)"""
    global vectorstore

    if not vectorstore:
        return "❌ Knowledge base not properly initialized. Please check the system logs."

    logger.info("🔍 Using Gemini model: %s", gemini_model_name or "None")

    try:
        # Retrieve relevant documents for context
        docs = vectorstore.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        logger.debug("📚 Retrieved %d context documents", len(docs))

        prompt = f"""You are a friendly cybersecurity coach helping someone understand phishing risks.

Context:
{context if context.strip() else "No additional context retrieved."}

User Question:
{question}

Guidelines for your reply:
- Speak in clear, natural language—no encyclopedic tone.
- Start with a brief summary (1-2 sentences) that directly answers the question.
- Follow with concise, actionable advice or next steps in short paragraphs or a very short bullet list (max 3 bullets).
- Keep the response under 180 words and avoid generic filler.
- If the context lacks specific details, say so and offer best-practice guidance."""

        if gemini_model:
            gemini_text = generate_with_gemini(prompt)
            if gemini_text:
                logger.info("✅ Gemini response received (first 100 chars: %s...)", gemini_text[:100])
                return gemini_text
            logger.warning("⚠️ Gemini returned no content. Falling back to contextual response.")
        else:
            logger.warning("⚠️ Gemini model unavailable; falling back to contextual response.")

        if context.strip():
            fallback_response = (
                "⚠️ Gemini service is unavailable right now, so here's a quick summary from our knowledge base:\n\n"
                "Summary: Phishing attacks try to trick you into sharing sensitive details. Double-check sender info and "
                "avoid clicking unexpected links.\n\n"
                "Next steps:\n"
                "- Review the following reference notes pulled from our library.\n"
                "- Cross-check any suspicious message through a trusted channel before responding.\n\n"
                f"Reference notes:\n{context}"
            )
        else:
            fallback_response = (
                "⚠️ Gemini service is unavailable and no extra context was found. "
                "Use caution with unexpected links or requests and try again in a moment."
            )

        return fallback_response

    except Exception as e:
        logger.error(f"❌ Error generating RAG response: {str(e)}")
        return f"Error generating response: {str(e)}. Please try again later."

# Initialize vectorstore
initialize_vectorstore()

# ==== Routes ====
@app.route("/")
def home():
    return "✅ RAG Backend is running successfully"

@app.route("/test_gemini", methods=["GET"])
def test_gemini():
    """Test Gemini API connection"""
    try:
        model_name = gemini_model_name

        if not gemini_model:
            logger.info("🔄 Gemini model not initialized; attempting to pick one now.")
            model_name = pick_and_init_model()

        if not gemini_model:
            return jsonify({
                "message": "⚠️ Gemini API not configured",
                "error": "GEMINI_API_KEY not set or valid models unavailable.",
                "status": "not_configured",
                "model": model_name
            }), 503

        text = generate_with_gemini("Say 'Hello, RAG Backend!'")
        if not text:
            return jsonify({
                "message": "⚠️ Gemini responded without content.",
                "status": "degraded",
                "model": gemini_model_name
            }), 502

        return jsonify({
            "message": "✅ Gemini API working successfully!",
            "test_response": text,
            "status": "working",
            "model": gemini_model_name
        })
    except Exception as e:
        return jsonify({"error": f"❌ Gemini API test failed: {str(e)}"}), 500

@app.route("/api/chat", methods=["POST"])
def chat():
    """Handle chatbot queries - this is what the frontend calls"""
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "❌ Missing 'question' in request body"}), 400
        
        question = data['question']
        if not question.strip():
            return jsonify({"error": "❌ Question cannot be empty"}), 400
        
        logger.info(f"🔍 Processing chatbot query: {question}")
        
        # Generate response using RAG
        response = generate_response_with_rag(question)
        
        return jsonify({
            "success": True,
            "response": response,
            "question": question
        })
        
    except Exception as e:
        logger.error(f"❌ Error processing chatbot query: {str(e)}")
        return jsonify({"error": f"❌ Error processing query: {str(e)}"}), 500

@app.route("/query", methods=["POST"])
def query():
    """Handle RAG queries"""
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "❌ Missing 'question' in request body"}), 400
        
        question = data['question']
        if not question.strip():
            return jsonify({"error": "❌ Question cannot be empty"}), 400
        
        logger.info(f"🔍 Processing query: {question}")
        
        # Generate response using RAG
        response = generate_response_with_rag(question)
        
        return jsonify({
            "question": question,
            "response": response,
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"❌ Error processing query: {str(e)}")
        return jsonify({"error": f"❌ Error processing query: {str(e)}"}), 500

@app.route("/status", methods=["GET"])
def status():
    """Get system status"""
    api_key = GEMINI_API_KEY
    status_info = {
        "gemini_configured": gemini_model is not None,
        "vectorstore_loaded": vectorstore is not None,
        "embeddings_model_loaded": embeddings_model is not None,
        "api_key_present": bool(api_key),
        "model": gemini_model_name
    }
    
    return jsonify(status_info)

@app.route("/api/rag/health", methods=["GET"])
def rag_health():
    """Health check endpoint for RAG backend"""
    return jsonify({
        "status": "healthy",
        "gemini_configured": gemini_model is not None,
        "vectorstore_loaded": vectorstore is not None
    })

if __name__ == "__main__":
    logger.info("🌐 Server ready to accept requests on port 5002")
    app.run(debug=True, port=5002)
