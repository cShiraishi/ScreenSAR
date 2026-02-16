import streamlit as st
import bcrypt
import time
from sqlalchemy import text, MetaData, Table, Column, Integer, String, Float
from sqlalchemy.exc import SQLAlchemyError

def get_connection():
    """
    Returns the SQL connection object using Streamlit's st.connection.
    Expects secrets to contain a 'postgres' section with a 'uri' key.
    """
    if "postgres" in st.secrets and "uri" in st.secrets["postgres"]:
        return st.connection("qsar_db", type="sql", url=st.secrets["postgres"]["uri"])
    else:
        # Fallback
        try:
           return st.connection("qsar_db", type="sql") 
        except:
           return None

def init_db():
    """
    Initializes the database table using SQLAlchemy metadata.
    This works for both SQLite and PostgreSQL.
    """
    try:
        conn = get_connection()
        if conn is None:
            return

        # Define table structure using SQLAlchemy Core (Dialect agnostic)
        metadata = MetaData()
        users = Table(
            'users', metadata,
            Column('id', Integer, primary_key=True),
            Column('email', String, unique=True, nullable=False),
            Column('password_hash', String, nullable=False),
            Column('created_at', Float)
        )
        
        # Create table if not exists
        metadata.create_all(conn.engine)
            
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
