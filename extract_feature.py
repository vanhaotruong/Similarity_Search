import tensorflow as tf
from keras.applications.efficientnet import EfficientNetB0, preprocess_input
from keras.preprocessing import image
import glob, os, tqdm
import numpy as np

model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
target_size = (150, 150)

files = glob.glob('./features/total_train/**/*.*', recursive=True)
for f in files:
    os.remove(f)

# train_files
train_files = glob.glob('./Intel_Classification_Dataset/seg_train/**/*.jpg', recursive=True)
test_files = glob.glob('./Intel_Classification_Dataset/seg_test/**/*.jpg', recursive=True)

total_train_files = train_files + test_files
total_train_files = sorted(total_train_files)

batch_size = 640

for i in range(0, len(total_train_files), batch_size):
    batch_files = total_train_files[i:i+batch_size]
    batch_images = []
    
    labels = []  # List to store the labels
    file_names = []

    for file in batch_files:
        # Extract the label from the file path and add it to the labels list
        label = os.path.basename(os.path.dirname(file))

        labels.append(label)
        file_names.append(file)

        img = image.load_img(file, target_size=target_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        batch_images.append(x)
        
    batch_images = np.vstack(batch_images)

    features = model.predict(batch_images)
    np.savez(f'./features/total_train/batch_number_{i//batch_size}.npz', 
             features= features, labels= labels, file_names= file_names)
    print(f'Batch number {i//batch_size} saved')