import tensorflow as tf
from keras.applications.efficientnet import EfficientNetB0, preprocess_input
from keras.preprocessing import image
import glob, tqdm, faiss, os, annoy
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
target_size = (150, 150)
dim = 1280  # EffientNetB0 output has 1280 features when input image size is (150, 150)
nbit= 64


####### Train annoy.AnnoyIndex
index = annoy.AnnoyIndex(dim)

train_paths = glob.glob('./features/total_train/**/*.npz', recursive=True)
train_paths = sorted(train_paths)

n_trees = 10

index = annoy.AnnoyIndex(dim, 'angular')

labels = []
file_names = []
all_features = []
for train_path in tqdm.tqdm(train_paths):
    traindata = np.load(train_path)

    all_features.append(traindata['features'])
    labels.extend(traindata['labels'])
    file_names.extend(traindata['file_names'])

all_features = np.vstack(all_features)

for i, feature in tqdm.tqdm(enumerate(all_features)):
    index.add_item(i, feature)

index.build(n_trees)
index.save('IndexAnnoy.index')

######## Inference
index = annoy.AnnoyIndex(dim, 'angular')
index.load('IndexAnnoy.index')
print(f"Number of images added to the index: {index.get_n_items()}")

predict_files = glob.glob('./Intel_Classification_Dataset/seg_pred/**/*.jpg', recursive=True)

nneighbors = 7
idx = 553
img = image.load_img(predict_files[idx], target_size=target_size)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
features = model.predict(x)
features = np.squeeze(features)

I, D = index.get_nns_by_vector(features, n= nneighbors, include_distances=True)
print(D)
print(I)

# Display the query image
plt.figure(figsize=(60, 20))
plt.subplot(1, nneighbors + 1, 1)
plt.imshow(img)
plt.title('Query Image')

for i, ind in enumerate(I):
    plt.subplot(1, nneighbors + 1, i+2)
    plt.imshow(mpimg.imread(file_names[ind]))
    plt.title(f'Rank {i+1}, Label: {labels[ind]}')

plt.show()



