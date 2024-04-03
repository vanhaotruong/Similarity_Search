import tensorflow as tf
from keras.applications.efficientnet import EfficientNetB0, preprocess_input
from keras.preprocessing import image
import glob, tqdm, faiss, os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
target_size = (150, 150)
dim = 1280  # EffientNetB0 output has 1280 features when input image size is (150, 150)
nbit= 64


####### Train faiss.IndexLSH
index = faiss.IndexLSH(dim, nbit)
train_paths = glob.glob('./features/total_train/**/*.npz', recursive=True)
train_paths = sorted(train_paths)

labels = []
file_names = []
for train_path in tqdm.tqdm(train_paths):
    traindata = np.load(train_path)

    index.add(traindata['features'])
    labels.extend(traindata['labels'])
    file_names.extend(traindata['file_names'])

faiss.write_index(index, 'IndexLSH.index')

######## Inference
index = faiss.read_index('IndexLSH.index')
print(f"Number of images added to the index: {index.ntotal}")

predict_files = glob.glob('./Intel_Classification_Dataset/seg_pred/**/*.jpg', recursive=True)

nneighbors = 7
idx = 5563
img = image.load_img(predict_files[idx], target_size=target_size)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
features = model.predict(x)

D, I = index.search(features, k= nneighbors)
print(D)
print(I)

# Display the query image
plt.figure(figsize=(60, 20))
plt.subplot(1, nneighbors + 1, 1)
plt.imshow(img)
plt.title('Query Image')

for i, ind in enumerate(I[0]):
    plt.subplot(1, nneighbors + 1, i+2)
    plt.imshow(mpimg.imread(file_names[ind]))
    plt.title(f'Rank {i+1}, Label: {labels[ind]}')

plt.show()



