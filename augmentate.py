import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2


db_path = 'face_db'
augmented_path = 'augmented_faces'


if not os.path.exists(augmented_path):   #if directory doesnt exist then create one
    os.makedirs(augmented_path)


def load_images_from_directory(directory):
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg"):                          #taking in only jpg files
                image_paths.append(os.path.join(root, file))
    return pd.DataFrame(image_paths, columns=['image_path'])

face_df = load_images_from_directory(db_path)

def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))               
    return img

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

def augment_and_save(image_path, save_path, num_augmentations=5):
    img = load_image(image_path)
    img = img.reshape((1,) + img.shape)  
    i = 0
    for batch in datagen.flow(img, batch_size=1, save_to_dir=save_path, save_prefix='aug', save_format='jpg'):
        i += 1
        if i >= num_augmentations:
            break  

for idx, row in face_df.iterrows():
    image_path = row['image_path']
    image_name = os.path.basename(image_path).split('.')[0]
    save_dir = os.path.join(augmented_path, image_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    augment_and_save(image_path, save_dir)

print("Data augmentation complete and saved in:", augmented_path)
