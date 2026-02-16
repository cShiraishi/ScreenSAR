import streamlit as st
import bcrypt
import time
from sqlalchemy import text, MetaData, Table, Column, Integer, String, Float
from sqlalchemy.exc import SQLAlchemyError

def get_connection():
    """
    Returns the SQL connection object using Streamlit's st.connection.
    Using pool_pre_ping=True helps avoid 'SSL connection has been closed unexpectedly' 
    by checking if the connection is alive before using it.
    """
    if "postgres" in st.secrets and "uri" in st.secrets["postgres"]:
        return st.connection("qsar_db", type="sql", url=st.secrets["postgres"]["uri"], pool_pre_ping=True)
    else:
        try:
           return st.connection("qsar_db", type="sql", pool_pre_ping=True) 
        except:
           return None

def init_db():
    """
    Initializes the database table using Raw SQL for robustness.
    """
    try:
        conn = get_connection()
        if conn is None:
            st.error("Falha na conexão com o Banco de Dados.")
            return

        # Use raw SQL to ensure table creation works across versions/dialects
        # properly with the SQLAlchemy engine
        with conn.engine.connect() as c:
            # PostgreSQL / SQLite compatible CREATE TABLE
            # check dialect to adapt syntax if strictly needed, but basic SQL is fine
            
            # Using SQLAlchemy text() for safety
            sql = """
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at DOUBLE PRECISION
                );
            """
            
            # If SQLite, SERIAL doesn't work the same, but SQLAlchemy handles it? 
            # No, raw SQL implies dialect specific.
            # Let's check dialect.
            
            if conn.engine.dialect.name == 'sqlite':
                sql = """
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        email TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        created_at REAL
                    );
                """

            c.execute(text(sql))
            c.commit()
            
    except Exception as e:
        # Show this error explicitly so we know if init failed
        st.error(f"Erro Crítico ao Inicializar Banco de Dados (init_db): {e}")

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
