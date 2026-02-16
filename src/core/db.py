import streamlit as st
import bcrypt
import time
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

def get_connection():
    """
    Returns the SQL connection object using Streamlit's st.connection.
    Expects secrets to contain a 'postgres' section with a 'uri' key,
    OR standard Streamlit SQL secrets.
    """
    if "postgres" in st.secrets and "uri" in st.secrets["postgres"]:
        # Custom section
        return st.connection("qsar_db", type="sql", url=st.secrets["postgres"]["uri"])
    else:
        # Fallback to default 'connections.postgresql' if defined, or error
        try:
           return st.connection("postgresql", type="sql") 
        except:
           return None

def init_db():
    """
    Initializes the database table if it doesn't exist.
    """
    try:
        conn = get_connection()
        if conn is None:
            return

        # Create users table
        # We use 'text' for password_hash to store the decoded bcrypt hash
        create_table_sql = """
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at DOUBLE PRECISION
            );
        """
        with conn.session as s:
            s.execute(text(create_table_sql))
            s.commit()
            
    except Exception as e:
        print(f"DB Init Error: {e}")

def create_user(email, password):
    """
    Creates a new user in PostgreSQL.
    """
    conn = get_connection()
    if not conn:
        st.error("Erro de configuração do Banco de Dados.")
        return False
        
    # Check if exists
    # Note: st.connection.query() returns a DataFrame by default, but we can check if it's empty
    # Or use session for scalar query
    try:
        # Check existence
        existing = conn.query("SELECT email FROM users WHERE email = :email", params={"email": email}, ttl=0)
        if not existing.empty:
            return False
            
        # Hash password
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
        
        # Insert
        with conn.session as s:
            sql = "INSERT INTO users (email, password_hash, created_at) VALUES (:email, :pwd, :created)"
            s.execute(
                text(sql), 
                {"email": email, "pwd": hashed, "created": time.time()}
            )
            s.commit()
        return True
        
    except Exception as e:
        st.error(f"Erro ao criar usuário: {e}")
        return False

def verify_user(email, password):
    """
    Verifies user credentials against PostgreSQL.
    """
    conn = get_connection()
    if not conn:
        return False
        
    try:
        df = conn.query("SELECT password_hash FROM users WHERE email = :email", params={"email": email}, ttl=0)
        
        if df.empty:
            return False
            
        stored_hash = df.iloc[0]["password_hash"]
        
        # Encode back to bytes for bcrypt
        if bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8')):
            return True
            
        return False
    except Exception as e:
        print(f"Login error: {e}")
        return False
