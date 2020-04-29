# make a prediction for a new image.
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import numpy as np

import os
#from tqdm import tqdm
 
# load and prepare the image
def load_image(filename):
    #print("start load image")
    # load the image
    img = load_img(filename, target_size=(240, 320))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 240, 320, 3)
    # center pixel data
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    #print("end load image")
    return img
 
# load an image and predict the class
def run_example(filepath):
    #print("start run example")
    # load the image
    img = load_image(filepath)

    # predict the class
    result = model.predict(img)
    print(result[0])

    if (result[0][0] < 0.5) :
        print(filename," : @@ blur @@")
	return result[0][0]
    else :
        print(filename, " :  ## non blur ##")
        return result[0][0]
 
if __name__ == "__main__":

  print("start!")
  path_dir = "/hci/fuse_sampledata/"
  file_list = os.listdir(path_dir)
  file_list.sort()
  i = 0
  
  model = load_model('/hci/model/tf2_fN3d_resnet_10_2.h5')
  print("model loaded")
  #for filename in file_list:
  #    i = i+1
  #  if(i<10):
  #    print(i)
  #    filepath = path_dir + filename
  #    run_example(filepath)
  #    K.clear_session()
  filename = "frame00402.png"
  filepath = path_dir + filename
  run_example(filepath)
  K.clear_session()



