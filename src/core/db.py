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
            st.error("Falha na conexão com o Banco de Dados. Verifique st.secrets.")
            return

        # Use conn.session for more idiomatic Streamlit SQLConnection usage
        with conn.session as s:
            # Check dialect via the session's bind
            try:
                dialect = s.bind.dialect.name
            except AttributeError:
                # Fallback for different SQLAlchemy/Streamlit versions
                dialect = 'sqlite' # Default to sqlite or try to detect from engine if possible
                if hasattr(conn, 'engine'):
                     dialect = conn.engine.dialect.name

            # PostgreSQL / SQLite compatible CREATE TABLE
            if dialect == 'sqlite':
                sql = """
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        email TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        created_at REAL
                    );
                """
            else:
                # Assuming PostgreSQL or compatible
                sql = """
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        email TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        created_at DOUBLE PRECISION
                    );
                """

            s.execute(text(sql))
            s.commit()
            
    except Exception as e:
        # Show this error explicitly so we know if init failed
        st.error(f"Erro Crítico ao Inicializar Banco de Dados (init_db): {e}")
        import traceback
        st.code(traceback.format_exc())

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
