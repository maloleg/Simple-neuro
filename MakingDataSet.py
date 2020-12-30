import numpy as np
from PIL import Image
import time
import tensorflow as tf
import cv2

triangleNames = ['triangles/done/done_1.jpg', 'triangles/done/done_2.jpg', 'triangles/done/done_3.jpg',
                 'triangles/done/done_4.jpg', 'triangles/done/done_5.jpg', 'triangles/done/done_6.jpg',
                 'triangles/done/done_7.jpg', 'triangles/done/done_8.jpg', 'triangles/done/done_9.jpg',
                 'triangles/done/done_10.jpg', 'triangles/done/done_11.jpg', 'triangles/done/done_12.jpg',
                 'triangles/done/done_13.jpg', 'triangles/done/done_14.jpg', 'triangles/done/done_15.jpg',
                 'triangles/done/done_16.jpg', 'triangles/done/done_17.jpg', 'triangles/done/done_18.jpg',
                 'triangles/done/done_19.jpg', 'triangles/done/done_20.jpg', 'triangles/done/done_21.jpg',
                 'triangles/done/done_22.jpg', 'triangles/done/done_23.jpg', 'triangles/done/done_24.jpg',
                 'triangles/done/done_25.jpg', 'triangles/done/done_26.jpg', 'triangles/done/done_27.jpg',
                 'triangles/done/done_28.jpg', 'triangles/done/done_29.jpg', 'triangles/done/done_30.jpg',
                 'triangles/done/done_31.jpg', 'triangles/done/done_32.jpg', 'triangles/done/done_33.jpg',
                 'triangles/done/done_34.jpg', 'triangles/done/done_35.jpg', 'triangles/done/done_36.jpg',
                 'triangles/done/done_37.jpg', 'triangles/done/done_38.jpg', 'triangles/done/done_39.jpg',
                 'triangles/done/done_40.jpg', 'triangles/done/done_41.jpg', 'triangles/done/done_42.jpg',
                 'triangles/done/done_43.jpg', 'triangles/done/done_44.jpg', 'triangles/done/done_45.jpg',
                 'triangles/done/done_46.jpg', 'triangles/done/done_47.jpg', 'triangles/done/done_48.jpg',
                 'triangles/done/done_49.jpg', 'triangles/done/done_50.jpg']

squareNames = ['squares/done/done_1.jpg', 'squares/done/done_2.jpg', 'squares/done/done_3.jpg',
               'squares/done/done_4.jpg', 'squares/done/done_5.jpg', 'squares/done/done_6.jpg',
               'squares/done/done_7.jpg', 'squares/done/done_8.jpg', 'squares/done/done_9.jpg',
               'squares/done/done_10.jpg', 'squares/done/done_11.jpg', 'squares/done/done_12.jpg',
               'squares/done/done_13.jpg', 'squares/done/done_14.jpg', 'squares/done/done_15.jpg',
               'squares/done/done_16.jpg', 'squares/done/done_17.jpg', 'squares/done/done_18.jpg',
               'squares/done/done_19.jpg', 'squares/done/done_20.jpg', 'squares/done/done_21.jpg',
               'squares/done/done_22.jpg', 'squares/done/done_23.jpg', 'squares/done/done_24.jpg',
               'squares/done/done_25.jpg', 'squares/done/done_26.jpg', 'squares/done/done_27.jpg',
               'squares/done/done_28.jpg', 'squares/done/done_29.jpg', 'squares/done/done_30.jpg',
               'squares/done/done_31.jpg', 'squares/done/done_32.jpg', 'squares/done/done_33.jpg',
               'squares/done/done_34.jpg', 'squares/done/done_35.jpg', 'squares/done/done_36.jpg',
               'squares/done/done_37.jpg', 'squares/done/done_38.jpg', 'squares/done/done_39.jpg',
               'squares/done/done_40.jpg', 'squares/done/done_41.jpg', 'squares/done/done_42.jpg',
               'squares/done/done_43.jpg', 'squares/done/done_44.jpg', 'squares/done/done_45.jpg',
               'squares/done/done_46.jpg', 'squares/done/done_47.jpg', 'squares/done/done_48.jpg',
               'squares/done/done_49.jpg', 'squares/done/done_50.jpg']

circleNames = ['circles/done/done_1.jpg', 'circles/done/done_2.jpg', 'circles/done/done_3.jpg',
               'circles/done/done_4.jpg', 'circles/done/done_5.jpg', 'circles/done/done_6.jpg',
               'circles/done/done_7.jpg', 'circles/done/done_8.jpg', 'circles/done/done_9.jpg',
               'circles/done/done_10.jpg', 'circles/done/done_11.jpg', 'circles/done/done_12.jpg',
               'circles/done/done_13.jpg', 'circles/done/done_14.jpg', 'circles/done/done_15.jpg',
               'circles/done/done_16.jpg', 'circles/done/done_17.jpg', 'circles/done/done_18.jpg',
               'circles/done/done_19.jpg', 'circles/done/done_20.jpg', 'circles/done/done_21.jpg',
               'circles/done/done_22.jpg', 'circles/done/done_23.jpg', 'circles/done/done_24.jpg',
               'circles/done/done_25.jpg', 'circles/done/done_26.jpg', 'circles/done/done_27.jpg',
               'circles/done/done_28.jpg', 'circles/done/done_29.jpg', 'circles/done/done_30.jpg',
               'circles/done/done_31.jpg', 'circles/done/done_32.jpg', 'circles/done/done_33.jpg',
               'circles/done/done_34.jpg', 'circles/done/done_35.jpg', 'circles/done/done_36.jpg',
               'circles/done/done_37.jpg', 'circles/done/done_38.jpg', 'circles/done/done_39.jpg',
               'circles/done/done_40.jpg', 'circles/done/done_41.jpg', 'circles/done/done_42.jpg',
               'circles/done/done_43.jpg', 'circles/done/done_44.jpg', 'circles/done/done_45.jpg',
               'circles/done/done_46.jpg', 'circles/done/done_47.jpg', 'circles/done/done_48.jpg',
               'circles/done/done_49.jpg', 'circles/done/done_50.jpg']

# data preparation
# for i in range(1, 51):
#     file = Image.open('triangles/' + str(i) + '.jpg').convert('L').resize((50, 50))
#     file.save('triangles/done/done_' + str(i) + '.jpg')
#     triangleNames.append('triangles/done/done_' + str(i) + '.jpg')
# for i in range(1, 51):
#     file = Image.open('squares/' + str(i) + '.jpg').convert('L').resize((50, 50))
#     file.save('squares/done/done_' + str(i) + '.jpg')
#     squareNames.append('squares/done/done_' + str(i) + '.jpg')
# for i in range(1, 51):
#     file = Image.open('circles/' + str(i) + '.jpg').convert('L').resize((50, 50))
#     file.save('circles/done/done_' + str(i) + '.jpg')
#     circleNames.append('circles/done/done_' + str(i) + '.jpg')

# print(triangleNames)
# print(squareNames)
# print(circleNames)

names = triangleNames + squareNames + circleNames
labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
          2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
          2, 2]

images = []
for i in names:
    im = Image.open(i)
    images.append((np.array(im)))
# dataset = (images, labels)
# print(type(dataset))
np.savez('./imagesdataset.npz', DataX=images, DataY=labels)