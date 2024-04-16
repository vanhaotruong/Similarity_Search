import tensorflow as tf
from keras.applications.efficientnet import EfficientNetB0, preprocess_input
from keras.preprocessing import image
import glob, tqdm, faiss, os, annoy, random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_fscore_support
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def extract_features(image_folder, batch_size, target_size):
    '''
    - image_folder: str or list of strings. If str, it is the path to the image folder. 
    If list of strings, it is the list of paths to several image folders.
    - batch_size: int. Number of images to process at a time.
    - target_size: tuple. Size of the image to resize to.

    Functions: 
    - Split the dataset into training set and testing set.
    - Extract features from the images using the EfficientNetB0 model.
    - Save the features, labels, and file names in a .npz file. Each .npz file contains the features, 
    labels, and file names of a batch of images.
    - For training set, save the files in the './features/train_files' folder.
    - For testing set, save the files in the './features/test_files' folder.
    
    '''

    model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')

    if not os.path.exists('./features'):
        os.makedirs('./features')
    if not os.path.exists('./features/train_files'):
        os.makedirs('./features/train_files')
    if not os.path.exists('./features/test_files'):
        os.makedirs('./features/test_files')

    files = glob.glob('./features/**/*.*', recursive=True)
    for f in files:
        os.remove(f)

    if isinstance(image_folder, str):
        dataset = glob.glob(f'{image_folder}/**/*.jpg', recursive=True)
    elif isinstance(image_folder, list):    # list of strings: Multiple image folders            
        dataset = None
        for img_folder in image_folder:
            ds = glob.glob(f'{img_folder}/**/*.jpg', recursive=True)
            if dataset is None:
                dataset = ds
            else:
                dataset += ds

    train_files, test_files = train_test_split(dataset, test_size=0.1, random_state=42)

    train_files = sorted(train_files)
    test_files = sorted(test_files)
    print(len(train_files), len(test_files))

    for idx, files in enumerate([train_files, test_files]):
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i+batch_size]
            batch_images = []
            
            labels = []  # List to store the labels
            file_names = []

            for file in batch_files:
                # Extract the label from the file path and add it to the labels list
                label = os.path.basename(os.path.dirname(file))
                label = label.split('-')[-1]

                labels.append(label)
                file_names.append(file)

                img = image.load_img(file, target_size=target_size)
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                batch_images.append(x)
                
            batch_images = np.vstack(batch_images)

            features = model.predict(batch_images)

            if idx == 0:
                np.savez(f'./features/train_files/batch_number_{i//batch_size:02}.npz', 
                        features= features, labels= labels, file_names= file_names)
                print(f'Train_files: Batch number {i//batch_size} saved')
            else:
                np.savez(f'./features/test_files/batch_number_{i//batch_size:02}.npz', 
                        features= features, labels= labels, file_names= file_names)
                print(f'Test_files: Batch number {i//batch_size} saved')

def indexing(dataset_features, method):
    '''
    - dataset_features: numpy array. Features extracted from WHOLE TRAINING SET.
    - method: str. Indexing method to use. Acceptable values are 'Annoy', 'LSH', 'IVFPQ'.

    Functions:
    - Build the index using the dataset_features.
    - Save the result in the './Index' folder.
    - For 'Annoy' method, save the index in the './Index/IndexAnnoy.ann' file.
    - For 'LSH' method, save the index in the './Index/IndexLSH.lsh' file.
    - For 'IVFPQ' method, save the index in the './Index/IndexIVF.ivf' file.
    
    '''

    if not os.path.exists('./Index'):
        os.makedirs('./Index')

    if method == 'Annoy':
        n_trees = 256    # Number of trees to build in the index
        index = annoy.AnnoyIndex(dim, 'angular')
        for i, feature in tqdm.tqdm(enumerate(dataset_features)):
            index.add_item(i, feature)
        index.build(n_trees)
        index.save('./Index/IndexAnnoy.ann')
    elif method == 'LSH':
        nbit= 256 # number of bits in the code
        index = faiss.IndexLSH(dim, nbit)
        index.add(dataset_features)
        faiss.write_index(index, './Index/IndexLSH.lsh')
    elif method == 'IVFPQ':
        nlist = 256 # number of clusters
        m = 8  # number of subquantizers
        nbits = 8  # bits allocated per subquantizer
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
        index.train(dataset_features)
        index.add(dataset_features)
        faiss.write_index(index, './Index/IndexIVF.ivf')

def inference(feature, method, index_file=None, train_info_folder = None, nneighbors=7):
    '''
    - feature: numpy 2d array, has shape = (1, N). Feature extracted from SINGLE query image.
    - method: str. Indexing method to use. Acceptable values are 'Annoy', 'LSH', 'IVFPQ'.
    - index_file: str. Path to the index file. If None, use the default path. which is
        - 'Annoy':  './Index/IndexAnnoy.ann'
        - 'LSH':    './Index/IndexLSH.lsh'
        - 'IVFPQ':  './Index/IndexIVF.ivf'
    - train_info_folder: str. Path to the folder containing the training set information.
    If None, use the default path which is './features/train_files/**/*.npz'.
    - nneighbors: int. Number of nearest neighbors to return.

    Return:
    - k_neighbors: list of tuples. Each tuple contains the simiarity, label, and file name of the nearest neighbor.
    '''
    if train_info_folder:
        train_info_paths = glob.glob(train_info_folder)
    else:
        train_info_paths = glob.glob('./features/train_files/**/*.npz', recursive=True)

    train_info_paths = sorted(train_info_paths)
    batch_size = np.load(train_info_paths[0])['features'].shape[0]

    file_names = []
    labels = []

    for idx, train_info_path in enumerate(train_info_paths):
        train_info = np.load(train_info_path)
        file_names.extend(train_info['file_names'])
        labels.extend(train_info['labels'])

    k_neighbors = []
    if method == 'Annoy':
        index = annoy.AnnoyIndex(dim, 'angular')
        if index_file:
            index.load(index_file)
        else:
            index.load('./Index/IndexAnnoy.ann')
        feature = np.squeeze(feature)
        I, D = index.get_nns_by_vector(feature, n= nneighbors, include_distances=True)
    elif method == 'LSH':
        if index_file:
            index = faiss.read_index(index_file)
        else:
            index = faiss.read_index('./Index/IndexLSH.lsh')
        D, I = index.search(feature, k= nneighbors)
        D = D[0]
        I = I[0]        
    elif method == 'IVFPQ':
        if index_file:
            index = faiss.read_index(index_file)
        else:
            index = faiss.read_index('./Index/IndexIVF.ivf')
        D, I = index.search(feature, k= nneighbors)
        D = D[0]
        I = I[0]  
    else:
        print('Invalid method')
        return None
    
    
    for i in I:
        batch_idx = i//batch_size
        sub_idx = i%batch_size
        feature = np.squeeze(feature)
        nb_feature = np.load(train_info_paths[batch_idx])['features'][sub_idx]
        similarity = np.dot(feature, nb_feature.T)/(np.linalg.norm(feature)*np.linalg.norm(nb_feature))
        k_neighbors.append((similarity, labels[i], file_names[i]))
    return k_neighbors

if __name__ == '__main__':
    method = 'Annoy'

    model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
    dim = model.output_shape[1] # Dimension of the feature vector
    target_size = (224, 224)


    ################ Extract_Features ################
    # image_folder = ['./Intel_Classification_Dataset/seg_train', 
    #                 './Intel_Classification_Dataset/seg_test']
    image_folder = './Dataset'
    extract_features(image_folder, batch_size=640, target_size=target_size)


    ############### Create Indexing ################
    train_info_paths = glob.glob('./features/train_files/**/*.npz', recursive=True)
    train_info_paths = sorted(train_info_paths)
    
    dataset_features = []
    for train_info_path in tqdm.tqdm(train_info_paths):
        train_info = np.load(train_info_path)
        dataset_features.append(train_info['features'])

    dataset_features = np.vstack(dataset_features)
    indexing(dataset_features, method)

    # ################# Calculate Accuracy, Precision, Recall, F1-Score ################
    testset_info = glob.glob('./features/test_files/**/*.npz', recursive=True)
    
    testset_info = sorted(testset_info)

    y_true_labels = []
    y_pred_labels = []
    for info in tqdm.tqdm(testset_info):
        info = np.load(info)
        features = info['features']
        y_true_labels.extend(info['labels'])

        for i, feature in enumerate(features):
            if method == 'IVFPQ' or method == 'LSH':
                feature = np.expand_dims(feature, axis=0)
            nearest_neighbor = inference(feature, method, nneighbors=1)
            y_pred_labels.append(nearest_neighbor[0][1])

    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    precision = precision_score(y_true_labels, y_pred_labels, average='macro')
    recall = recall_score(y_true_labels, y_pred_labels, average='macro')
    f1 = f1_score(y_true_labels, y_pred_labels, average='macro')

    print(f'Method: {method}')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-Score: {f1}')

    cm = confusion_matrix(y_true_labels, y_pred_labels)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Show the plot
    plt.show()

    ############### Inference ################
    ## Randomly select an image from the dataset ##
    # image_paths = glob.glob('./Intel_Classification_Dataset/**/*.jpg', recursive=True) 
    image_paths = glob.glob('./Dataset/**/*.jpg', recursive=True) 
    img_path = random.choice(image_paths)

    ## Load the image and extract the feature ##
    query_label = os.path.basename(os.path.dirname(img_path))
    query_label = query_label.split('-')[-1]
    query_img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(query_img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = model.predict(x)

    ## Inference and return nneighbors nearest neighbors ##
    nneighbors = 7
    k_neighbors = inference(feature, method, nneighbors=nneighbors)

    ## Display the query image and its nneighbors nearest neighbors ##
    plt.figure(figsize=(60, 20))
    plt.subplot(1, nneighbors + 1, 1)
    plt.imshow(query_img)
    plt.title(f'Query: {query_label}')

    for i, (similarity, label, file_name) in enumerate(k_neighbors):
        plt.subplot(1, nneighbors + 1, i+2)
        plt.imshow(mpimg.imread(file_name))
        plt.title(f'{similarity:.2f}, {label}')

    plt.show()
 

