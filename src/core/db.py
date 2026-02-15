import streamlit as st
import pymongo
import bcrypt
import time

# Use Streamlit's secrets for the URI
# Expected format in .streamlit/secrets.toml:
# [mongo]
# uri = "mongodb+srv://user:pass@cluster.mongodb.net/..."

@st.cache_resource
def init_connection():
    """
    Initializes and caches the MongoDB connection.
    Returns the database client.
    """
    if "mongo" not in st.secrets or "uri" not in st.secrets["mongo"]:
        # Silent return if not configured (prevents crashes before config)
        return None
        
    try:
        client = pymongo.MongoClient(st.secrets["mongo"]["uri"])
        # Quick check 
        client.admin.command('ping')
        return client
    except Exception as e:
        print(f"MongoDB Config Error: {e}")
        return None

def get_db():
    client = init_connection()
    if client:
        return client.get_database("qsar_db") # Name of your DB
    return None

def init_db():
    """
    For MongoDB, we just check connection. 
    Collections are created eagerly on first write.
    """
    client = init_connection()
    if not client:
        # Don't show error here, handled in UI if needed
        pass

def create_user(email, password):
    """
    Creates a new user in MongoDB.
    """
    db = get_db()
    if not db:
        st.error("Erro de conexão com Banco de Dados (MongoDB não configurado).")
        return False
        
    users = db.users
    
    # Check if exists
    if users.find_one({"email": email}):
        return False
        
    # Hash password
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    
    user_doc = {
        "email": email,
        "password_hash": hashed,
        "created_at": time.time()
    }
    
    try:
        users.insert_one(user_doc)
        return True
    except Exception as e:
        print(f"DB Insert Error: {e}")
        return False

def verify_user(email, password):
    """
    Verifies user credentials against MongoDB.
    """
    db = get_db()
    if not db:
        # Fallback to deny or check local? 
        # For now, if DB is missing, we can't login via DB.
        return False
        
    users = db.users
    user = users.find_one({"email": email})
    
    if user:
        stored_hash = user["password_hash"]
        if bcrypt.checkpw(password.encode('utf-8'), stored_hash):
            return True
            
    return False
