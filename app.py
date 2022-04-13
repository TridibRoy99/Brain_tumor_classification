
import streamlit as st
import base64
from pathlib import Path
import tensorflow as tf
from PIL import Image
import requests
import matplotlib as mpl
import matplotlib.pyplot as plt
import pathlib
import shutil
import numpy as np
import seaborn as sns
import sklearn
import pandas as pd
import cv2
import scipy
import os
from keras.preprocessing.image import load_img
from tempfile import NamedTemporaryFile
import io

st.set_option('deprecation.showfileUploaderEncoding', False)

st.set_page_config(
    page_title="Brain Tumor Detector",
    page_icon="🧠",
    layout="wide",
    menu_items={
         'Get Help': 'https://www.linkedin.com/in/tridib-roy-974374145/',
         'Report a bug': "https://www.linkedin.com/in/tridib-roy-974374145/",
         'About': "Portfolio WebApp"
     }
)

st.title("Brain tumor Classifier")
# st.image("https://media.giphy.com/media/3o6MbhQZGGeskpDJLi/giphy.gif")
with st.expander("Expand for details on the classification model!!"):
    st.info("__Description:__ This model classifies a CT scan image of brain as - Tumor or Non-tumor")
    st.info("__Framework / model used:__ This model uses tranfer learning with Tensorflow. Base model used - InceptionV3 \n")
    st.image("https://production-media.paperswithcode.com/methods/inceptionv3onc--oview_vjAbOfw.png")
    st.info("__Dataset used:__ It is trained on Brain MRI Images for Brain Tumor Detection from Kaggle")
    

name_cols=st.columns(2)

CT_url = st.file_uploader("Upload CT scan image of brain", type=["png","jpg","jpeg"]) 


try:
  st.image(CT_url,caption="Uploaded image")
  with st.spinner("Processing the image and loading necessary files....."):
    new_model = tf.keras.models.load_model(urllib.request.urlopen('https://drive.google.com/file/d/1Id2s1LXIVwvCJpU0wy7p39cf6aCrULqL/view?usp=sharing'))
    # im = Image.open(requests.get(CT_url, stream=True).raw)
    im = Image.open(CT_url)
    im = np.array(im).astype('float32')/255
    if len(im.shape)==2:
        im=tf.expand_dims(im,-1)
    else:
        pass

    im = tf.image.resize(im, (150,150))
    

    if im.shape[2]==1:
        im = tf.image.grayscale_to_rgb(tf.convert_to_tensor(im), name=None)
    else:
        pass
    

    clf=(new_model.predict(tf.expand_dims(im,axis=0)) > 0.65).astype("int32")[0][0]
    
    st.success("Processing Completed!")
    st.write("")
    st.write("")
    st.info("The model classification results are as follows:  ")
    
    st.write(f"Chance of being a tumor: {new_model.predict(tf.expand_dims(im,axis=0))[0][0]:.02%}")
    if clf==1:
        st.write('Identifying as **Tumor** detected')
    else:
        st.write('Identifying as **No Tumor** detected')

except:
  st.text("Waiting for image....")
