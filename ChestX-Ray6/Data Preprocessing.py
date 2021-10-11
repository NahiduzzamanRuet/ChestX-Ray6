import os
import cv2

data_path =r'F:/NAHID/Research/Final_Data/Train_data'
categories = os.listdir(data_path)
print("Number of classes", categories)
noofClasses = len(categories)
print("Total number of classes", noofClasses)
print("Importing images")

labels=[i for i in range(len(categories))]

label_dict=dict(zip(categories, labels)) #empty dictionary

print(label_dict)
print(categories)
print(labels)

data = []
target = []


for category in categories:
    folder_path = os.path.join(data_path, category)
    img_names = os.listdir(folder_path)

    for img_name in img_names:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        try:
            resized = cv2.resize(img, (150, 150))
            imga = resized/255.0
            imga = cv2.equalizeHist(imga)
            data.append(imga)
            target.append(label_dict[category])
            # appending the image and the label(categorized) into the list (dataset)

        except Exception as e:
            print('Exception:', e)
            # if any exception rasied, the exception will be printed here. And pass to the next image


print(len(data))
import numpy as np

data=np.array(data)
target=np.array(target)

print(data.shape)
print(target.shape)

np.save('xtrain1.npy', data)
np.save('ytrain1.npy', target)
