"""Include All Library Files"""

from IPython.display import Image, display
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D
from keras.utils.np_utils import to_categorical
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.cm as cm
from sklearn import metrics
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from keras.applications.mobilenet_v2 import MobileNetV2
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc

"""Load GPU"""

print("Device: \n", tf.config.experimental.list_physical_devices())
print(tf.__version__)
print(tf.test.is_built_with_cuda())

"""Load Data"""

X_train = np.load('F:/Rivision1 code/xtrain.npy')
y_train = np.load('F:/Rivision1 code/ytrain.npy')
X_test = np.load('F:/Rivision1 code/xtest.npy')
y_test = np.load('F:/Rivision1 code/ytest.npy')
X_val = np.load('F:/Rivision1 code/xval.npy')
y_val = np.load('F:/Rivision1 code/yval.npy')

"""Total No. of Classes"""

noofClasses = 6


"""Create Target Categorical"""

y_train = to_categorical(y_train, noofClasses)
y_test = to_categorical(y_test, noofClasses)
y_val = to_categorical(y_val, noofClasses)

"""Design Lightweight ChestX-Ray6 Model"""

def myModel():
    model = tf.keras.Sequential()
    # 1st conv layer
    model.add(tf.keras.layers.Conv2D(32, 5, padding="same", input_shape=(150, 150, 3)))
    model.add(tf.keras.layers.Conv2D(32, 5, padding="same"))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPooling2D(2))
    # 2nd conv layer
    model.add(tf.keras.layers.Conv2D(64, 3, padding="valid"))
    model.add(tf.keras.layers.Conv2D(64, 3, padding="valid"))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPooling2D(2))
    # 3rd conv layer
    model.add(tf.keras.layers.Conv2D(128, 3, padding="valid"))
    model.add(tf.keras.layers.Conv2D(128, 3, padding="valid", name = "rrr"))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPooling2D(2))
    model.add(tf.keras.layers.Dropout(0.5))
    #Flatten Layer
    model.add(tf.keras.layers.Flatten())
    # FC1
    model.add(tf.keras.layers.Dense(1024))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    # FC2
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    # output FC
    model.add(tf.keras.layers.Dense(6, name = 'last_layer'))
    model.add(tf.keras.layers.Activation('softmax'))
    adam = tf.keras.optimizers.Adam(lr=0.001)

    model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')

    return model

"""Show Model"""

model = myModel()
print(model.summary())

batchSizeVal = 64
epochs = 100

start = time.time()
with tf.device('/GPU:0'):
    model = myModel()
    history = model.fit(X_train, y_train, batch_size= batchSizeVal, epochs=epochs, validation_data=(X_val, y_val), shuffle=1)

end = time.time()
elapsed = end - start
print("Total Time:", elapsed)

"""Save Model"""

model.save("F:/NAHID/Research/DR/model/chestx-ray631.h5")

"""Show Accuracy"""

plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

"""Show Loss"""

plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')

"""Show Result"""
start = time.time()
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score = ', score[0])
print('Test Accuracy = ', score[1])
end = time.time()
elapsed = end - start
print("Total Time:", elapsed)
pred = model.predict(X_test)
pred = np.argmax(pred, axis=-1)
y_pre = model.predict(X_test)
orig_test_labels1 = np.argmax(y_test, axis=-1)
cm1 = confusion_matrix(orig_test_labels1, pred)
cr1 = classification_report(orig_test_labels1, pred)
print(cm1)
print(cr1)
score = metrics.accuracy_score(orig_test_labels1, pred)
print("Accuracy score: {}".format(score))
pred_prob = model.predict(X_test)
c = roc_auc_score(orig_test_labels1, pred_prob, multi_class='ovo')
print("AUC:", c)

"""Compute ROC curve and ROC area for each class"""

fpr = {}
tpr = {}
roc_auc = {}
thresh = {}
lw = 2
precision = {}
recall = {}
for i in range(noofClasses):
    fpr[i], tpr[i], thresh[i] = roc_curve(orig_test_labels1, pred_prob[:, i], pos_label=i)
    precision[i], recall[i], _ = roc_curve(orig_test_labels1, pred_prob[:, i], pos_label=i)
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend(loc="best")
plt.title("precision vs. recall curve")
plt.show()

n_classes = 6

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()

colors = cycle(['green', 'darkorange', 'cornflowerblue', 'yellow', 'blue', 'red'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.4f})'
                                                       ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()


"""Visualization"""

img_size = (150, 150)
last_conv_layer_name = "rrr"
img_path ='../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/person1009_virus_1694.jpeg'
display(Image(img_path))

def get_img_array(img_path, size):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,size)
    array = np.expand_dims(image, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

"""Heat Map"""

# Prepare image
img_array = get_img_array(img_path, size=img_size)
preds = model.predict(img_array)
print("Predicted:", preds)
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
plt.matshow(heatmap)
plt.show()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    #image = cv2.imread(img_path)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)


    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))

"""Grad-CAM Visualization"""

save_and_display_gradcam(img_path, heatmap)

"""PCA"""

intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('last_layer').output)
feauture_engg_data = intermediate_layer_model.predict(X_test)
feauture_engg_dataframe = pd.DataFrame(feauture_engg_data)
feauture_engg_dataframe.to_csv('ChestXray6TestData.csv')
y_dataframe = pd.DataFrame(y_test)
y_dataframe.to_csv('ChestXray6TestData_y.csv')

pca_features = PCA(n_components=2)
principalComponents_features = pca_features.fit_transform(feauture_engg_dataframe)
principalDf = pd.DataFrame(data = principalComponents_features
             , columns = ['principal component 1', 'principal component 2'])
principalDf = pd.concat([principalDf, y_dataframe], axis = 1)

principalDf.to_csv('pca_normal.csv', index = False)
principalDf['Target'] = principalDf[0]*0+principalDf[1]*1+principalDf[2]*2+principalDf[3]*3+principalDf[4]*4+principalDf[5]*5
principalDf.drop(columns=[0, 1, 2, 3, 4, 5], inplace =True)
print('Explained variation per principal component: {}'.format(pca_features.explained_variance_ratio_))
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1: '+str(round(pca_features.explained_variance_ratio_[0]*100,4))+'%', fontsize = 12)
ax.set_ylabel('Principal Component 2: '+str(round(pca_features.explained_variance_ratio_[1]*100,4))+'%', fontsize = 12)
ax.set_title('ChestX-Ray6', fontsize = 12)
targets = [0, 1, 2, 3, 4, 5]
colors = ['r', 'g', 'b', 'c', 'y', 'm']
for target, color in zip(targets,colors):
    indicesToKeep = principalDf['Target'] == target
    ax.scatter(principalDf.loc[indicesToKeep, 'principal component 1']
               , principalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 8)
ax.legend(targets)
ax.grid()

"""Transfer Learning Model"""

"""VGG19"""

mo = VGG19(input_shape=(150, 150, 3), weights='imagenet', include_top=False)

for layer in mo.layers:
    layer.trainable = False
x = Flatten()(mo.output)

prediction = Dense(6, activation='softmax')(x)

# create a model object
model = Model(inputs=mo.input, outputs=prediction)
model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')

start = time.time()
history = model.fit(X_train, y_train, batch_size=644, epochs=100, validation_data=(X_test, y_test), shuffle=1)
end = time.time()
elapsed = end - start
print("Total Time:", elapsed)

"""Show Result"""
start = time.time()
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score = ', score[0])
print('Test Accuracy = ', score[1])
end = time.time()
elapsed = end - start
print("Total Time:", elapsed)

feature_extractor = model.predict(X_test)
feauture_engg_data = feature_extractor
feauture_engg_dataframe = pd.DataFrame(feauture_engg_data)
feauture_engg_dataframe.to_csv('mo.csv')
y_dataframe = pd.DataFrame(y_test)
y_dataframe.to_csv('mo_y.csv')

from sklearn.decomposition import PCA
pca_features = PCA(n_components=2)
principalComponents_features = pca_features.fit_transform(feauture_engg_dataframe)
principalDf = pd.DataFrame(data = principalComponents_features, columns = ['principal component 1', 'principal component 2'])

principalDf = pd.concat([principalDf, y_dataframe], axis = 1)

principalDf.to_csv('pca_normalmocsv', index = False)
principalDf.head()

principalDf['Target'] = principalDf[0]*0+principalDf[1]*1+principalDf[2]*2+principalDf[3]*3+principalDf[4]*4+principalDf[5]*5
principalDf.drop(columns=[0, 1, 2, 3, 4, 5], inplace =True)
print('Explained variation per principal component: {}'.format(pca_features.explained_variance_ratio_))
principalDf.head()

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1: '+str(round(pca_features.explained_variance_ratio_[0]*100,4))+'%', fontsize = 12)
ax.set_ylabel('Principal Component 2: '+str(round(pca_features.explained_variance_ratio_[1]*100,4))+'%', fontsize = 12)
ax.set_title('VGG19', fontsize = 12)
targets = [0, 1, 2, 3, 4, 5]
colors = ['r', 'g', 'b', 'c', 'y', 'm']
for target, color in zip(targets,colors):
    indicesToKeep = principalDf['Target'] == target
    ax.scatter(principalDf.loc[indicesToKeep, 'principal component 1']
               , principalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 8)
ax.legend(targets)
ax.grid()


"""ResNet50"""

IMAGE_SIZE = [150, 150]
Res = ResNet50(input_shape=(150, 150, 3), weights='imagenet', include_top=False)

for layer in Res.layers:
    layer.trainable = False
x = Flatten()(Res.output)

prediction = Dense(6, activation='softmax', name='lastres')(x)

# create a model object
model = Model(inputs=Res.input, outputs=prediction)
model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')

start = time.time()
history = model.fit(X_train, y_train, batch_size=64, epochs=100, validation_data=(X_test, y_test), shuffle=1)

end = time.time()
elapsed = end - start
print("Total Time:", elapsed)


"""Show Result"""
start = time.time()
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score = ', score[0])
print('Test Accuracy = ', score[1])
end = time.time()
elapsed = end - start
print("Total Time:", elapsed)


feature_extractor = model.predict(X_test)
feauture_engg_data = feature_extractor

feauture_engg_dataframe = pd.DataFrame(feauture_engg_data)
feauture_engg_dataframe.to_csv('res.csv')
y_dataframe = pd.DataFrame(y_test)
y_dataframe.to_csv('res_y.csv')

from sklearn.decomposition import PCA
pca_features = PCA(n_components=2)
principalComponents_features = pca_features.fit_transform(feauture_engg_dataframe)
principalDf = pd.DataFrame(data = principalComponents_features, columns = ['principal component 1', 'principal component 2'])

principalDf = pd.concat([principalDf, y_dataframe], axis = 1)

principalDf.to_csv('pca_normalres.csv', index = False)
principalDf.head()

principalDf['Target'] = principalDf[0]*0+principalDf[1]*1+principalDf[2]*2+principalDf[3]*3+principalDf[4]*4+principalDf[5]*5
principalDf.drop(columns=[0, 1, 2, 3, 4, 5], inplace =True)
print('Explained variation per principal component: {}'.format(pca_features.explained_variance_ratio_))
principalDf.head()

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1: '+str(round(pca_features.explained_variance_ratio_[0]*100,4))+'%', fontsize = 12)
ax.set_ylabel('Principal Component 2: '+str(round(pca_features.explained_variance_ratio_[1]*100,4))+'%', fontsize = 12)
ax.set_title('ResNet', fontsize = 12)
targets = [0, 1, 2, 3, 4, 5]
colors = ['r', 'g', 'b', 'c', 'y', 'm']
for target, color in zip(targets,colors):
    indicesToKeep = principalDf['Target'] == target
    ax.scatter(principalDf.loc[indicesToKeep, 'principal component 1']
               , principalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 8)
ax.legend(targets)
ax.grid()

"""DenseNet121"""

densenet = DenseNet121(input_shape=(150, 150, 3), weights='imagenet', include_top=False)
for layer in densenet.layers:
    layer.trainable = False
x = Flatten()(densenet.output)

prediction = Dense(6, activation='softmax', name='lastdense')(x)

# create a model object
model = Model(inputs=densenet.input, outputs=prediction)
model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')

start = time.time()
history = model.fit(X_train, y_train, batch_size=64, epochs=100, validation_data=(X_test, y_test), shuffle=1)
end = time.time()
elapsed = end - start
print("Total Time:", elapsed)


"""Show Result"""
start = time.time()
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score = ', score[0])
print('Test Accuracy = ', score[1])
end = time.time()
elapsed = end - start
print("Total Time:", elapsed)

feature_extractor = model.predict(X_test)
feauture_engg_data = feature_extractor
feauture_engg_dataframe = pd.DataFrame(feauture_engg_data)
feauture_engg_dataframe.to_csv('dense.csv')
y_dataframe = pd.DataFrame(y_test)
y_dataframe.to_csv('dense_y.csv')

from sklearn.decomposition import PCA
pca_features = PCA(n_components=2)
principalComponents_features = pca_features.fit_transform(feauture_engg_dataframe)
principalDf = pd.DataFrame(data = principalComponents_features, columns = ['principal component 1', 'principal component 2'])

principalDf = pd.concat([principalDf, y_dataframe], axis = 1)

principalDf.to_csv('pca_normaldense.csv', index = False)
principalDf.head()

principalDf['Target'] = principalDf[0]*0+principalDf[1]*1+principalDf[2]*2+principalDf[3]*3+principalDf[4]*4+principalDf[5]*5
principalDf.drop(columns=[0, 1, 2, 3, 4, 5], inplace =True)
print('Explained variation per principal component: {}'.format(pca_features.explained_variance_ratio_))
principalDf.head()

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1: '+str(round(pca_features.explained_variance_ratio_[0]*100,4))+'%', fontsize = 12)
ax.set_ylabel('Principal Component 2: '+str(round(pca_features.explained_variance_ratio_[1]*100,4))+'%', fontsize = 12)
ax.set_title('DenseNet121', fontsize = 12)
targets = [0, 1, 2, 3, 4, 5]
colors = ['r', 'g', 'b', 'c', 'y', 'm']
for target, color in zip(targets,colors):
    indicesToKeep = principalDf['Target'] == target
    ax.scatter(principalDf.loc[indicesToKeep, 'principal component 1']
               , principalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 8)
ax.legend(targets)
ax.grid()

"""MobileNetV2"""

mo = MobileNetV2(input_shape=(150, 150, 3), weights='imagenet', include_top=False)

for layer in mo.layers:
    layer.trainable = False
x = Flatten()(mo.output)

prediction = Dense(6, activation='softmax')(x)

# create a model object
model = Model(inputs=mo.input, outputs=prediction)

model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')

start = time.time()
history = model.fit(X_train, y_train, batch_size=64, epochs=100, validation_data=(X_test, y_test), shuffle=1)
end = time.time()
elapsed = end - start
print("Total Time:", elapsed)

"""Show Result"""
start = time.time()
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score = ', score[0])
print('Test Accuracy = ', score[1])
end = time.time()
elapsed = end - start
print("Total Time:", elapsed)

feature_extractor = model.predict(X_test)
feauture_engg_data = feature_extractor
feauture_engg_dataframe = pd.DataFrame(feauture_engg_data)
feauture_engg_dataframe.to_csv('mo.csv')
y_dataframe = pd.DataFrame(y_test)
y_dataframe.to_csv('mo_y.csv')

from sklearn.decomposition import PCA
pca_features = PCA(n_components=2)
principalComponents_features = pca_features.fit_transform(feauture_engg_dataframe)
principalDf = pd.DataFrame(data = principalComponents_features, columns = ['principal component 1', 'principal component 2'])

principalDf = pd.concat([principalDf, y_dataframe], axis = 1)

principalDf.to_csv('pca_normalmocsv', index = False)
principalDf.head()

principalDf['Target'] = principalDf[0]*0+principalDf[1]*1+principalDf[2]*2+principalDf[3]*3+principalDf[4]*4+principalDf[5]*5
principalDf.drop(columns=[0, 1, 2, 3, 4, 5], inplace =True)
print('Explained variation per principal component: {}'.format(pca_features.explained_variance_ratio_))
principalDf.head()

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1: '+str(round(pca_features.explained_variance_ratio_[0]*100,4))+'%', fontsize = 12)
ax.set_ylabel('Principal Component 2: '+str(round(pca_features.explained_variance_ratio_[1]*100,4))+'%', fontsize = 12)
ax.set_title('MobileNetV2', fontsize = 12)
targets = [0, 1, 2, 3, 4, 5]
colors = ['r', 'g', 'b', 'c', 'y', 'm']
for target, color in zip(targets,colors):
    indicesToKeep = principalDf['Target'] == target
    ax.scatter(principalDf.loc[indicesToKeep, 'principal component 1']
               , principalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 8)
ax.legend(targets)
ax.grid()