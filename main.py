import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
from io import BytesIO
import tensorflow as tf

model_name = 'models/SRGAN_EP10000.h5'

model = load_model(model_name)

def getType(fn):
    f = fn.split('.')[-1].lower()
    if f == 'jpg' or f == 'jpeg':
        return 'jpeg'
    else:
        return 'png'

def crop(im, default_size, crop_size):
    cropped_img = []
    cropped_img_size = []
    pos = []
    imwidth, imheight = default_size
    for j in range(0,default_size[1],crop_size[1]):
        for i in range(0, default_size[0], crop_size[0]):
            if default_size[1] - j < 100 or default_size[0] - i < 100:
                if default_size[1] - j < 100 and default_size[0] - i < 100:
                    box = (i,j,default_size[0],default_size[1])
                    pos.append(box)
                    a = im.crop(box)
                    a = a.resize(crop_size)
                    a = np.array(a)
                    cropped_img.append(a)
                    cropped_img_size.append((default_size[0] - i,default_size[1] - j))
                if default_size[0] - i < 100:
                    box = (i,j,default_size[0],j+crop_size[1])
                    pos.append(box)
                    a=im.crop(box)
                    a = a.resize(crop_size)
                    a = np.array(a)
                    cropped_img.append(a)
                    cropped_img_size.append((default_size[0] - i,crop_size[1]))
                else:
                    box = (i,j,i+crop_size[0],default_size[1])
                    pos.append(box)
                    a=im.crop(box)

                    a = a.resize(crop_size)
                    a = np.array(a)
                    cropped_img.append(a)
                    cropped_img_size.append((crop_size[0],default_size[1]-j))
            else:
                box = (i,j, i+crop_size[0],j+crop_size[1])
                pos.append(box)
                a = im.crop(box)
                a = np.array(a)
                cropped_img.append(a)
                cropped_img_size.append(crop_size)
    return np.array(cropped_img) / 127.5 - 1. , cropped_img_size, pos




def processImage(im, default_size):
    return crop(im, default_size=default_size, crop_size=(100,100))
def predict(im):
    return model.predict(im)
def afterProcessing(t,default_size):
    t = 0.5 * t + 0.5
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

def saveImgsToOne(o, default_size, pos):
    new_image = Image.new('RGB',default_size, (250,250,250))
    for x in range(len(o)):
        new_image.paste(o[x],(pos[x][0],pos[x][1]))
    return new_image


st.markdown('# Super Resolution GAN')
st.markdown('Just upload your low-resolution image, then you\'re done!')

uploaded_file = st.file_uploader("Choose a Image", type=['jpg','png'])

if uploaded_file is not None:
    st.info('Processing...')
    filename = uploaded_file.name
    filetype = getType(filename)
    image = Image.open(uploaded_file).convert('RGB')
    default_size = image.size

    im, sizeList, pos = processImage(image, default_size)
    processedImgList = []

    for x in range(len(im)):
        t = tf.expand_dims(im[x], axis=0)
        o = model.predict(t)
        o = afterProcessing(o, sizeList[x])
        processedImgList.append(o)
    oneImg = saveImgsToOne(processedImgList,default_size, pos)
    loc = saveImage(oneImg, filetype)

    st.success('Here is your high-resolution Image!!!')
    st.image(oneImg)
    st.download_button('Download', loc, file_name=filename, mime='image'+filetype)
