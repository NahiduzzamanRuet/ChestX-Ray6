import numpy as np
from keras.preprocessing.image import ImageDataGenerator

#load normal data
data = np.load('nnormal.npy')
target = np.load('nnormal_target.npy')

#load Pneumonia data
#data = np.load('rn12D.npy')
#target = np.load('rn12T.npy')

#load Covid19 data
#data = np.load('nCovid19.npy')
#target = np.load('nCovid19_target.npy')

#load Cardiomegaly data
#data = np.load('nCardiomegaly.npy')
#target = np.load('nCardiomegaly_target.npy')

#load Lung Opacity data
#data = np.load('nLungOpacity.npy')
#target = np.load('nLungOpacity_target.npy')

#load Pleural data
#data = np.load('nPleural.npy')
#target = np.load('nPleural_target.npy')



datagen = ImageDataGenerator(
        rotation_range=45,
        width_shift_range=0.20,
        height_shift_range=0.20,
        shear_range=0.20,
        zoom_range=0.20,
        horizontal_flip=True,
        fill_mode='nearest')

i = 0

for batch in datagen.flow(data, batch_size=1, save_to_dir=r'E:\Revision data\Final\XTRA\New folder\A_Normal', save_prefix='Normal', save_format='jpeg'):
#for batch in datagen.flow(data, batch_size=1, save_to_dir=r'E:\Revision data\Final\XTRA\New folder\B_Pneumonia', save_prefix='Pneumonia', save_format='jpeg'):
#for batch in datagen.flow(data, batch_size=1, save_to_dir=r'E:\Revision data\Final\XTRA\New folder\C_COVID19', save_prefix='Covid19', save_format='jpeg'):
#for batch in datagen.flow(data, batch_size=1, save_to_dir=r'E:\Revision data\Final\XTRA\New folder\D_Cardiomegaly', save_prefix='Cardiomegaly', save_format='jpeg'):
#for batch in datagen.flow(data, batch_size=1, save_to_dir=r'E:\Revision data\Final\XTRA\New folder\E_Lung Opacity', save_prefix='Lung_Opacity', save_format='jpeg'):
#for batch in datagen.flow(data, batch_size=1, save_to_dir=r'E:\Revision data\Final\XTRA\New folder\F_Pleural', save_prefix='Pleural', save_format='jpeg'):
    i += 1
    if i > 1900:
    #if i > 1000:
    #if i >3352:
    #if i > 2725:
    #if i > 2328:
    #if i > 2200:
        break  # otherwise the generator would loop indefinitely