import numpy as np
from tensorflow.keras.preprocessing import image

def preprocess(img):

    img = img.resize((224,224))
    img = image.img_to_array(img)
    img = img/255.0
    img = np.expand_dims(img,axis=0)

    return img