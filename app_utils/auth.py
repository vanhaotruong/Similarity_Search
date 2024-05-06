import streamlit as st
from app_utils.generate_token import TokenManager
import hmac
from PIL import Image
import base64
from io import BytesIO

token_manager = TokenManager()


def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# change app_IMG to IMG
def check_password():
    def login_form():
        logo = Image.open("app_IMG/logo.png")
        logo_iand = Image.open("app_IMG/logo-dark.png")

        st.markdown(
            f'<div style="text-align: center"><img src="data:image/png;base64,{image_to_base64(logo)}" style="width:150px;"></div>',
            unsafe_allow_html=True,
        )
        st.divider()
        col1, col2, col3 = st.columns([2, 1, 2])

        with col2:
            with st.form("Credentials"):
                username = st.text_input("Username", key="username")
                password = st.text_input("Password", type="password", key="password")
                submitted = st.form_submit_button("Log in")
                if submitted:
                    password_entered(username, password)

        st.markdown(
            f'<h6 style="text-align: center">Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="12">&nbsp by &nbsp<a href="https://iand.dev"><img src="data:image/png;base64,{image_to_base64(logo_iand)}" alt="GRK" height="16"&nbsp</a></h6>',
            unsafe_allow_html=True,
        )

    def password_entered(username, password):
        if username in st.secrets["passwords"] and hmac.compare_digest(
            password,
            st.secrets["passwords"][username],
        ):
            st.session_state["password_correct"] = True
            st.session_state["user"] = username
            st.session_state["token"] = token_manager.get_token(username)
            st.rerun()
        else:
            st.session_state["password_correct"] = False
            st.error("Unknown user or Incorrect password.", icon="⚠️")

    if st.session_state.get("password_correct"):
        return True

    login_form()
    return False


