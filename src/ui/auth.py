import streamlit as st
import time
import os
import textwrap
import google_auth_oauthlib.flow
from googleapiclient.discovery import build
from src.core.db import init_db, create_user, verify_user

# Initialize DB on load
init_db()

# Allow OAuth over HTTP for local testing
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

def check_authentication():
    """
    Checks if the user is authenticated.
    Returns True if authenticated, False otherwise.
    """
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        
    return st.session_state.authenticated

def login():
    """
    Handles the login flow (Login / Sign Up / Google).
    """
    if check_authentication():
        return

    # Handle Google Callback
    if 'code' in st.query_params:
        handle_google_callback()
        return

    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.container(border=True):
            st.markdown(
                """
                <div style="text-align: center; margin-bottom: 20px; margin-top: 10px;">
                    <h1 style="font-size: 2rem;">Curadoria QSAR</h1>
                    <p style="color: #666;">Acesso Seguro</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            tab1, tab2 = st.tabs(["Entrar", "Criar Conta"])
            
            with tab1:
                with st.form("login_form"):
                    email = st.text_input("Email")
                    password = st.text_input("Senha", type="password")
                    submit = st.form_submit_button("Entrar", use_container_width=True)
                    
                    if submit:
                        if verify_user(email, password):
                            st.session_state.authenticated = True
                            st.success("Login realizado com sucesso!")
                            time.sleep(1)
                            st.rerun()
                        elif authenticate_via_secrets(email, password):
                            st.session_state.authenticated = True
                            st.success("Login realizado com sucesso (Admin)!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Email ou senha incorretos.")

            with tab2:
                st.markdown("### Novo Usuário")
                with st.form("signup_form"):
                    new_email = st.text_input("Seu Email")
                    new_pass = st.text_input("Sua Senha", type="password")
                    confirm_pass = st.text_input("Confirmar Senha", type="password")
                    signup_btn = st.form_submit_button("Cadastrar", use_container_width=True)
                    
                    if signup_btn:
                        if not new_email or not new_pass:
                            st.warning("Preencha todos os campos.")
                        elif new_pass != confirm_pass:
                            st.error("As senhas não coincidem.")
                        else:
                            if create_user(new_email, new_pass):
                                st.success("Conta criada! Login automático...")
                                time.sleep(1)
                                st.session_state.authenticated = True
                                st.rerun()
                            else:
                                st.error("Este email já está cadastrado.")

            # Divider for Google Login
            st.markdown(
                """
                <div style="display: flex; align-items: center; justify-content: center; margin: 25px 0 15px 0;">
                    <hr style="flex-grow: 1; border: 0; border-top: 1px solid #eee;">
                    <span style="padding: 0 10px; color: #888; font-size: 0.8em; text-transform: uppercase; letter-spacing: 1px;">ou continue com</span>
                    <hr style="flex-grow: 1; border: 0; border-top: 1px solid #eee;">
                </div>
                """,
                unsafe_allow_html=True
            )
            
            render_google_login_button()

def render_google_login_button():
    try:
        # Check for secrets
        if 'google' not in st.secrets:
            # Silent fail for UI if not configured, or show warning if development
            return

        client_config = {
            "web": {
                "client_id": st.secrets["google"]["client_id"],
                "client_secret": st.secrets["google"]["client_secret"],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [st.secrets["google"].get("redirect_uri", "http://localhost:8501")],
            }
        }

        # API Scopes
        scopes = [
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/userinfo.profile",
            "openid"
        ]

        flow = google_auth_oauthlib.flow.Flow.from_client_config(
            client_config,
            scopes=scopes,
            redirect_uri=st.secrets["google"].get("redirect_uri", "http://localhost:8501")
        )
        
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true'
        )
        
        # Google Logo (Base64)
        google_logo_b64 = "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdib3g9IjAgMCA0OCA0OCI+PHBhdGggZmlsbD0iI0VBNDMzNSIgQz0iTTI0IDQ4YzEzLjI1NSAwIDI0LTEwLjc0NSAyNC0yNGMwLTIuMjQzLS4yOTQtNC40MDEtLjg0LTYuNDQxSDB2MTIuNTQxaDI3LjQ3NGMtMS40MjIgNi42NjctNy4yOTUgMTEuNDU5LTE0LjI3NCAxMS40NTktOC44MzcgMC0xNi03LjE2My0xNi0xNnMuMTYzLTE2IDE2LTE2YzMuOTc0IDAgNy41ODkgMS40NjggMTAuNDUgMy44NDRsOS4zODItOS42MDVDMjcuNTQyIDIuMjk5IDIxLjExNCAwIDEzLjk4OCAwIDIuMDggMCAwIDEwLjc0NSAwIDI0czEwLjc0NSAyNCAyNCAyNHoiLz48cGF0aCBmaWxsPSIjMzRBMODUiIGQ9Ik00Ny4xNiAyNC4wMDRINjMuNjFjMy4yMDYgMTIuMzA2IDUuMDMxIDI1LjIzMSA1LjAzMSAzOC43OCAwIDEzLjU0OS0xLjg0NSAyNi40ODItNS4xNzMgMzguNzg2SDQ2Ljg5NmwzMjMxLTguODU4Yy0yLjMwNy0xNi41MzMtMS45MTUtMzMuNzk3LTguMDkxLTQ5Ljg2NiIvPjxwYXRoIGZpbGw9IiM0Mjg1RjQiIGQ9Ik0yNCA0OGMxMy4yNTUgMCAyNC0xMC43NDUgMjQtMjRjMC0yLjI0My0uMjk0LTQuNDAxLS44NC02LjQ0MUgwdjEyLjU0MWgyNy40NzRjLTEuNDIyIDYuNjY3LTcuMjk1IDExLjQ1OS0xNC4yNzQgMTEuNDU5Ii8+PHBhdGggZmlsbD0iIzM0QTg1MyIgZD0iTTQ2Ljk4IDIzLjk5NkM0Ni45OCAzNi4xNDIgMzguNjExIDQ2LjcyNiAyNy4zNzggNDcuODU5bC04LjkwOS03LjQxOUMxMi4zMjMgMzkuNTAyIDE1Ljg2MSAzMy45MDYgMjQuMDA0IDM2LjY1OXYtMTIuNjYzIi8+PHBhdGggZmlsbD0iI0ZCQkMwNSIgZD0iTTEwLjUzIDMxLjQ0MWwtOS4zODIgOS42MDVjNS45OTIgNC45NDcgMTMuNzUyIDcuOTUyIDIyLjI1OCA3Ljk1MmM3LjEyNiAwIDEzLjU1NC0yLjI5OSAxOC41MTctNS45MDZsLTguOTA5LTcuNDE5Yy0zLjc4MSAyLjU3NS04LjY5NSAzLjcxNi0xNC4wMTEgMy43MTYtOC44MzcgMC0xNi03LjE2My0xNi0xNiAwLS41NjIuMDI5LTEuMTE3LjA4NC0xLjY2NiIvPjxwYXRoIGZpbGw9IiNFQTQzMzUiIGQ9Ik0yNCA5LjQxYzMuOTc0IDAgNy41ODkgMS40NjggMTAuNDUgMy44NDRsOS4zODItOS42MDVDMjcuNTQyIDIuMjk5IDIxLjExNCAwIDEzLjk4OCAwIDEyLjIyNSAwIDYuNjU3IDEuNTg2IDEuOTIgNC4zODVsOS44ODIgNy42ODJDMTQuOTk0IDEwLjM1MSAxOS4zIDEwLjkwNiAyNCA5LjQxIi8+PC9zdmc+"
        
        # Use simpler SVG for reliability
        google_svg = textwrap.dedent("""
        <svg xmlns="http://www.w3.org/2000/svg" width="18px" height="18px" viewBox="0 0 48 48" style="display: block;">
            <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"></path>
            <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"></path>
            <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"></path>
            <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"></path>
            <path fill="none" d="M0 0h48v48H0z"></path>
        </svg>
        """).strip()

        button_html = f'''
            <div style="
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 10px;
                width: 100%;
            ">
                <a href="{authorization_url}" target="_self" style="
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    background-color: white;
                    color: #1f1f1f;
                    border: 1px solid #747775;
                    border-radius: 4px;
                    padding: 0px 16px;
                    height: 40px;
                    text-decoration: none;
                    font-family: 'Roboto', arial, sans-serif;
                    font-weight: 500;
                    font-size: 14px;
                    width: 100%;
                    box-shadow: 0 1px 2px rgba(0,0,0,0.05); /* Subtle shadow */
                    transition: background-color 0.2s, box-shadow 0.2s, border-color 0.2s;
                "
                onmouseover="this.style.backgroundColor='#f8f1f1'; this.style.boxShadow='0 1px 3px rgba(60,64,67,0.3)';"
                onmouseout="this.style.backgroundColor='white'; this.style.boxShadow='0 1px 2px rgba(0,0,0,0.05)';"
                >
                    <div style="margin-right: 12px; display: flex; align-items: center;">
                        {google_svg}
                    </div>
                    <span>Entrar com Google</span>
                </a>
            </div>
        '''
        
        st.markdown(textwrap.dedent(button_html), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Erro no botão Google: {e}")

def handle_google_callback():
    try:
        client_config = {
            "web": {
                "client_id": st.secrets["google"]["client_id"],
                "client_secret": st.secrets["google"]["client_secret"],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [st.secrets["google"].get("redirect_uri", "http://localhost:8501")],
            }
        }
        
        flow = google_auth_oauthlib.flow.Flow.from_client_config(
            client_config,
            scopes=[
                "https://www.googleapis.com/auth/userinfo.email",
                "https://www.googleapis.com/auth/userinfo.profile",
                "openid"
            ],
            redirect_uri=st.secrets["google"].get("redirect_uri", "http://localhost:8501")
        )
        
        code = st.query_params['code']
        flow.fetch_token(code=code)
        credentials = flow.credentials
        
        # Get user info
        user_info_service = build('oauth2', 'v2', credentials=credentials)
        user_info = user_info_service.userinfo().get().execute()
        
        # Success
        st.session_state.authenticated = True
        st.session_state.user_info = user_info
        
        # Clear code
        st.query_params.clear()
        st.rerun()
            
    except Exception as e:
        import traceback
        st.error(f"Falha no Login Google: {e}")
        st.code(traceback.format_exc())
        # Do not clear params immediately so user can see the error, 
        # but provide a way to retry/clear
        if st.button("Tentar Novamente"):
            st.query_params.clear()
            st.rerun()

def authenticate_via_secrets(email, password):
    """
    Validates credentials against secrets.toml (Legacy support)
    """
    if 'passwords' in st.secrets and email in st.secrets['passwords']:
        if st.secrets['passwords'][email] == password:
            return True
    return False

def logout():
    """
    Logs the user out.
    """
    st.session_state.authenticated = False
    if 'user_info' in st.session_state:
        del st.session_state['user_info']
    st.rerun()
