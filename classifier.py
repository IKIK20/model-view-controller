import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

x=np.load('image.npz')['arr_0']
y=pd.read_csv('labels.csv')["labels"]
print(pd.Series(y).value_counts())
classes= ['A', 'B', 'C', 'D','E','F','G','H','I','J','K', 'L','M','N','O','P','Q', 'R','S','T','U', 'V','W','X','Y', 'Z']
nclasses= len(classes) 

xtrain,xtest,ytrain,ytest= train_test_split(x,y,train_size=3500, test_size= 500, random_state=9)
xtrainscale= xtrain/255.0
xtestscale= xtest/255.0

clf= LogisticRegression(solver="saga",multi_class="multinomial").fit(xtrainscale,ytrain)

def getPrediction(image):
    imgpil=Image.open(image)
    imgbw= imgpil.convert("L")
    imgresized= imgbw.resize((22,30),Image.ANTIALIAS)
    pixelfilter= 20
    minpixel= np.percentile(imgresized,pixelfilter)
    imgscaled= np.clip(imgresized-minpixel,0,255)
    maxpixel= np.max(imgresized)
    imgscaled= np.asarray(imgscaled)/maxpixel

    testsample= np.array(imgscaled).reshape(1,784)
    testpredict= clf.predict(testsample)
    return testpredict[0]
  