import pandas as pd
import numpy as np
import os
import cv2
fer_data=pd.read_csv('fer2013.csv',delimiter=',')
from PIL import Image



def save_fer_img():

    for index,row in fer_data.iterrows():
        #print(index,row)
        pixels=np.asarray(list(row['pixels'].split(' ')),dtype=np.uint8)
        label=row['try']
        if not os.path.exists('output'+"/"+str(label)):
            os.makedirs('output'+"/"+str(label))
        img=pixels.reshape((48,48))
        pathname=os.path.join('output'+"/"+str(label),str(index)+'.jpg')
        cv2.imwrite(pathname,img)
        
##        im = Image.open('output'+"/"+str(label),str(index)+'.jpg')
##        im_l = im.convert('RGB')
##        im_1.save('output'+"/"+str(label),str(index)+'.jpg')
        im = Image.open('output'+"/"+str(label)+"/"+str(index)+'.jpg')
        #im = Image.open('output'+"/"+str(0.0)+'/'+str(0)+'.jpg')
        im_l = im.convert('RGB')
        im_l=im_l.resize((224,224))
        im_l.save('output'+"/"+str(label)+"/"+str(index)+'.jpg')


        #print('image saved ias {}'.format(pathname))
save_fer_img()
