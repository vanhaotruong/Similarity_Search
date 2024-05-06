import streamlit as st
from app_utils.file_path import image_file_path
from app_utils.generate_token import TokenManager
from app_utils.similarity_search import indexing,extract_features,inference
from keras.applications.efficientnet import EfficientNetB0, preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


import glob, tqdm, faiss, os, annoy, random
import numpy as np
from PIL import Image
import base64
from io import BytesIO
token_manager = TokenManager()
token_input = {}
model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()
def finding_similar_image():
    # print('st.session_state[/"image_file_path/"]',st.session_state["image_file_path"])
    try :
        if st.session_state["method"] in ["null", "method"] :
                return print('Method null!')
    except:
        return print('Method null!')
    try :
        if st.session_state["image_file_path"] in ["null"]:
            return print('Image null!')
    except: 
        return print('Image null!')
    try:
        if st.session_state["extract_feature"] in ["false"]:
            return print('Extract null!')
    except:
        return print('Extract null!')
    # print("fiding similar image!",st.session_state["image_file_path"])
    target_size = (224, 224)
    img_path = st.session_state["image_file_path"]
    method =  st.session_state["method"]
    ## Load the image and extract the feature ##
    query_img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(query_img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = model.predict(x)

    ## Inference and return nneighbors nearest neighbors ##
    nneighbors = 7
    k_neighbors = inference(feature, method, nneighbors=nneighbors)

    ## Display the query image and its nneighbors nearest neighbors ##
    cols = st.columns(nneighbors)
    for i, (similarity, label, file_name) in enumerate(k_neighbors):
        cols[i].image(mpimg.imread(file_name), caption=f'Similarity: {similarity:.2f}, Label: {label}', use_column_width=True)
    return print("success run !")

def logo():
    logo = Image.open("app_IMG/logo.png")
    st.sidebar.markdown(
        f'<div style="text-align: center"><img src="data:image/png;base64,{image_to_base64(logo)}" style="width: 50%;"></div>',
        unsafe_allow_html=True,
    )


def footer():
    logo_grk = Image.open("app_IMG/grk_logo.png")
    st.sidebar.divider()
    st.sidebar.markdown(
        f'<h6 style="text-align: center">Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="12">&nbsp by &nbsp<a href="https://github.com/GRKdev"><img src="data:image/png;base64,{image_to_base64(logo_grk)}" alt="GRK" height="16"&nbsp</a></h6>',
        unsafe_allow_html=True,
    )


# def clear_chat_history():
#     print("method choosed: ", st.session_state["method"])

# function for UI
def create_indexing(method):
    model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
    dim = model.output_shape[1] # Dimension of the feature vector
    target_size = (224, 224)


    ################ Extract_Features ################
    # image_folder = ['./Intel_Classification_Dataset/seg_train', 
    #                 './Intel_Classification_Dataset/seg_test']
    image_folder = './Dataset'
    extract_features(image_folder, batch_size=640, target_size=target_size)


    ############### Create Indexing ################
    train_info_paths = glob.glob('./features/train_files/**/*.npz', recursive=True)
    train_info_paths = sorted(train_info_paths)
    
    dataset_features = []
    for train_info_path in tqdm.tqdm(train_info_paths):
        train_info = np.load(train_info_path)
        dataset_features.append(train_info['features'])

    dataset_features = np.vstack(dataset_features)
    indexing(dataset_features, method)


def display_sidebar_info():

    if "user" in st.session_state:
        user = st.session_state["user"]

    if "user" in st.session_state:
        user = st.session_state["user"]
        if user == "admin":
            st.sidebar.markdown(f"👑 **Administrator**: {user.title()}")

            uploaded_files = st.sidebar.file_uploader(
                "Upload Image Files (Image)",
                accept_multiple_files=True,
                type=["JPG", "JPEG", "GIF", "PNG"],
            )
            # print("uploaded_files: ",uploaded_files)
            if uploaded_files and st.sidebar.button("Extract to feature"):
                try:
                    file_paths = image_file_path(uploaded_files)
                    st.session_state["image_file_path"] = "null2"
                    st.session_state["image_file_path"] = file_paths[0]
                    print('st.session_state["image_file_path"]',st.session_state["image_file_path"])
                    st.session_state["extract_feature"] = "true"
                except Exception as e:
                    st.sidebar.error(f"Error extract image: {e}")

        else:
            st.sidebar.markdown(f"**User:** {user.title()}")
    option = st.sidebar.selectbox(
        "Select Method for Similarity Search and Indexing",
        (   
           'Method',
           'Annoy',
           'LSH',
           'IVFPQ'
        ),
    )
    if option == "Annoy":
        lines = [
            "Indexing Processing with Annoy method"
        ]
        st.session_state["method"] = "Annoy"
        for line in lines:
            st.sidebar.markdown(f"```markdown\n{line}\n```")
    elif option == "Method":
        lines = [
            "Choosing your indexing method"
        ]
        st.session_state["method"] = "method"
        for line in lines:
            st.sidebar.markdown(f"```markdown\n{line}\n```")

    elif option == "LSH":
        lines = [
            "Indexing Processing with LSH method"
        ]
        st.session_state["method"] = "LSH"
        for line in lines:
            st.sidebar.markdown(f"```markdown\n{line}\n```")

    elif option == "IVFPQ":
        lines = [
            "Indexing Processing with IVFPQ method"
        ]
        st.session_state["method"] = "IVFPQ"
        for line in lines:
            st.sidebar.markdown(f"```markdown\n{line}\n```")
    if st.sidebar.button("1. Finding similar image !"):
        finding_similar_image()
    
    # if st.sidebar.button("2. Clear Searching"):
    #     Clear_Searching()
    footer()

def display_main_info():
        
        text_wapper = """
        #### **Welcome to the Similarity Search Imgage**
        This intelligent tradingbot allows you get similarity image such as: JPG, JPEG, GIF, PNG.
        ##### What can you do?
        - 👤 **Indexing**: Indexing for feature and image for searching.
        - 🛒 **Select number of neibor**: Setup number of neibor for searching.
        - 📊 **Upload image**: Search image follow image upload
        """
        st.info( text_wapper)
        # st.info( st.markdown(text_wapper))
        # st.markdown(text_wapper)
