import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
from io import BytesIO

model_name = 'SRGAN_EP100.h5'

model = load_model(model_name)

def getType(fn):
    f = fn.split('.')[-1].lower()
    if f == 'jpg' or f == 'jpeg':
        return 'jpeg'
    else:
        return 'png'

def processImage(im):
    im = image.resize((96, 96))
    im = np.array(im)
    im = [im]
    im = np.array(im) / 127.5 - 1.
    return im
def predict(im):
    return model.predict([im])
def afterProcessing(t,default_size):
    k = t[0] * 255
    k = k.astype(np.uint8)
    k = Image.fromarray(k)
    k = k.resize(default_size)
    return k
def saveImage(k,format):
    buf = BytesIO()
    k.save(buf, format=format.upper())
    byte_im = buf.getvalue()
    return byte_im


st.markdown('# Super Resolution GAN')
st.markdown('Just upload your low-resolution image, then you\'re done!')

uploaded_file = st.file_uploader("Choose a Image", type=['jpg','png'])

if uploaded_file is not None:
    st.info('Processing...')
    filename = uploaded_file.name
    filetype = getType(filename)
    image = Image.open(uploaded_file)
    default_size = image.size

    im = processImage(image)

    t = predict(im)
    k = afterProcessing(t,default_size)
    loc = saveImage(k, filetype)

    st.success('Here is your high-resolution Image!!!')
    st.image(k)
    st.download_button('Download', loc, file_name=filename, mime='image'+filetype)
