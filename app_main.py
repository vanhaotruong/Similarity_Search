
#############
# streamlit UI server
#############
# Can replace app into root folder for microservices

import streamlit as st
from app_utils.auth import check_password

# from app_utils.sidebar_info import logo




st.set_page_config(
    page_title="Tradingbot - Alphabot",
    page_icon="app_IMG/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "# **GRKdev** v1.2.0"},
)
# remove deployment button
st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)

if not check_password():

    st.stop()

#############
# Trading server
#############
# from chat_bot import chat_bot  # noqa: E402   
from app_utils.sidebar_info import logo
from app_utils.sidebar_info import display_sidebar_info,display_main_info
# if not  check_password() and not check_password():
#     st.stop()

if "user" in st.session_state and st.session_state["user"]:
    logo()
    # print(st.session_state.to_dict())
    # print("rotate in loop!")
    display_main_info()
    display_sidebar_info()
    # trading_bot()
    
else:
    st.write("User is not authenticated and the session status 'user' is not established.")