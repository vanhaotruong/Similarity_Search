import tensorflow as tf
from keras.applications.efficientnet import EfficientNetB0, preprocess_input
from keras.preprocessing import image
import glob, tqdm, faiss
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
target_size = (150, 150)
dim = 1280  # EffientNetB0 output has 1280 features when input image size is (150, 150)
nbit= 64


####### Train faiss.IndexLSH
index = faiss.IndexLSH(dim, nbit)
train_features = glob.glob('./features/total_train/**/*.npy', recursive=True)
for train_feature in tqdm.tqdm(train_features):
    features = np.load(train_feature)
    index.add(features)

faiss.write_index(index, 'IndexLSH.lsh')
print(f"Number of images added to the index: {index.ntotal}")

labels = np.load('./features/labels.npy')
file_name = np.load('./features/file_name.npy')

# Print the lengths of the arrays
print(f'Length of labels: {len(labels)}')
print(f'Length of file_name: {len(file_name)}')


######## Inference
# Load the arrays from the .npy files
index = faiss.read_index('IndexLSH.lsh')
print(f"Number of images added to the index: {index.ntotal}")

labels = np.load('./features/labels.npy')
file_name = np.load('./features/file_name.npy')

# Print the lengths of the arrays
print(f'Length of labels: {len(labels)}')
print(f'Length of file_name: {len(file_name)}')

predict_files = glob.glob('./Intel_Classification_Dataset/seg_pred/**/*.jpg', recursive=True)

nneighbors = 7
img = image.load_img(predict_files[7063], target_size=target_size)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
features = model.predict(x)
features = features[:,:512]

D, I = index.search(features, k= nneighbors)

# Display the query image
plt.figure(figsize=(60, 20))
plt.subplot(1, nneighbors + 1, 1)
plt.imshow(img)
plt.title('Query Image')

for i, ind in enumerate(I[0]):
    plt.subplot(1, nneighbors + 1, i+2)
    plt.imshow(mpimg.imread(file_name[ind]))
    plt.title(f'Rank {i+1}, Label: {labels[ind]}')

plt.show()



