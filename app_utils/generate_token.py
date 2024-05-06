import jwt
import datetime
import streamlit as st
import os

SECRET_KEY = st.secrets.get("SECRET_KEY", os.getenv("SECRET_KEY"))


class TokenManager:
    def __init__(self):
        # You don't need to keep the token or expiry_time in the state if you are always going to generate a new one.
        pass

    def get_token(self, username):
        # Determine role based on username
        role = "user"
        if username in ["admin", "direction"]:
            role = "admin"
        elif username == "user":
            role = "user"
        else:
            st.error("Unknown user, cannot assign a token.")

        # Siempre generar un nuevo token
        return self.create_jwt(username, role)

    def create_jwt(self, username, role):
        # El expiry time siempre es 1 día a partir de ahora.
        expiry_time = datetime.datetime.utcnow() + datetime.timedelta(days=1)
        payload = {
            "exp": expiry_time,
            "iat": datetime.datetime.utcnow(),
            "sub": username,
            "role": role,
        }
        return jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    
# # Authentication decorator
# def token_required():
#     def decorator():
#         token_dict = token.to_dict()
#         # ensure the jwt-token is passed with the headers
#         if (token_dict.get('token') is not None ):
#             jwt = token_dict.get('token')
#             data = jwt.decode(token_dict, SECRET_KEY, algorithms=['HS256'])
#             print(data)
#             return True
#         if not token_dict: # throw error if no token provided
#             print("message: A valid token is missing!")
#             return False
#         try:
#            # decode the token to obtain user public_id
#             data = jwt.decode(token_dict, SECRET_KEY, algorithms=['HS256'])
#         except:
#             print("message: Invalid token!")
#             return False
#         return False
#     return decorator()