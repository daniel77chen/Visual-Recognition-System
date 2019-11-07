#! /usr/bin/env python
import os
import cv2
import numpy as np
#import timeit, time
from time import perf_counter
from sklearn import neighbors, svm, cluster, preprocessing
from scipy import spatial


def load_data():
    test_path = '../data/test/'
    train_path = '../data/train/'
    
    train_classes = sorted([dirname for dirname in os.listdir(train_path) if not dirname.startswith('.')], key=lambda s: s.upper())
    test_classes = sorted([dirname for dirname in os.listdir(test_path) if not dirname.startswith('.')], key=lambda s: s.upper())
    train_labels = []
    test_labels = []
    train_images = []
    test_images = []
    for i, label in enumerate(train_classes):
        for filename in os.listdir(train_path + label + '/'):
            image = cv2.imread(train_path + label + '/' + filename, cv2.IMREAD_GRAYSCALE)
            train_images.append(image)
            train_labels.append(i)
    for i, label in enumerate(test_classes):
        for filename in os.listdir(test_path + label + '/'):
            image = cv2.imread(test_path + label + '/' + filename, cv2.IMREAD_GRAYSCALE)
            test_images.append(image)
            test_labels.append(i)
            
    return train_images, test_images, train_labels, test_labels


def KNN_classifier(train_features, train_labels, test_features, num_neighbors):
    # outputs labels for all testing images

    # train_features is an N x d matrix, where d is the dimensionality of the
    # feature representation and N is the number of training features.
    # train_labels is an N x 1 array, where each entry is an integer
    # indicating the ground truth category for each training image.
    # test_features is an M x d array, where d is the dimensionality of the
    # feature representation and M is the number of testing features.
    # num_neighbors is the number of neighbors for the KNN classifier

    classifier = neighbors.KNeighborsClassifier(n_neighbors=num_neighbors)
    classifier.fit(train_features,train_labels)

    predicted_categories = classifier.predict(test_features)

    # predicted_categories is an M x 1 array, where each entry is an integer
    # indicating the predicted category for each test image.
    return predicted_categories


def SVM_classifier(train_features, train_labels, test_features, is_linear, svm_lambda):
    # this function will train a linear svm for every category (i.e. one vs all)
    # and then use the learned linear classifiers to predict the category of
    # every test image. every test feature will be evaluated with all 15 svms
    # and the most confident svm will "win". confidence, or distance from the
    # margin, is w*x + b where '*' is the inner product or dot product and w and
    # b are the learned hyperplane parameters.

    # train_features is an N x d matrix, where d is the dimensionality of
    # the feature representation and N the number of training features.
    # train_labels is an N x 1 array, where each entry is an integer 
    # indicating the ground truth category for each training image.
    # test_features is an M x d matrix, where d is the dimensionality of the
    # feature representation and M is the number of testing features.
    # is_linear is a boolean. If true, you will train linear SVMs. Otherwise, you 
    # will use SVMs with a Radial Basis Function (RBF) Kernel.
    # svm_lambda is a scalar, the value of the regularizer for the SVMs

    if is_linear == True:
        classifier = svm.LinearSVC(C = svm_lambda)
        classifier.fit(train_features,train_labels)
        predicted_categories = classifier.predict(test_features)
    else:
    # predicted_categories is an M x 1 array, where each entry is an integer
    # indicating the predicted category for each test feature.
    
        svm_15 = []
        set_15 = set(train_labels)
        labels_15 = list(set_15)
        training_15 = []
        predicted_categories = []

        if is_linear == True:
            classifier = svm.LinearSVC(C = svm_lambda)
            classifier.fit(train_features,train_labels)
            predicted_categories = classifier.predict(test_features)
        else:
            for label in labels_15:
                temp = []
                for i in train_labels:
                    if label == i:
                        temp.append(label)
                    else:
                        temp.append(-1)
                training_15.append(temp)
            for i in range(15):
                grp = svm.SVC(C=svm_lambda, decision_function_shape="ovr", kernel='rbf', gamma='scale') 
                grp.fit(train_features, training_15[i])
                svm_15.append(grp)
            for test in test_features: #decide on each image 
                max_conf = -1
                most_win = None  
                min_conf_none = sys.maxsize
                least_win = None  

                for i, grp in enumerate(svm_15): #maybe not necessary
                    maybe_label = grp.predict(test.reshape(1, -1))
                    conf = abs(grp.decision_function(test.reshape(1, -1)))
                    if maybe_label == -1 and most_win is None:
                        if conf < min_conf_none:
                            min_conf_none = conf
                            least_win = i
                    elif maybe_label == -1 and most_win is not None:
                        continue
                    else:
                        if conf > max_conf:
                            max_conf = conf
                            most_win = maybe_label
                if most_win is not None:
                    predicted_categories.append(most_win) 
                else:
                    predicted_categories.append(least_win) #okay
        return predicted_categories

def imresize(input_image, target_size):
    # resizes the input image, represented as a 2D array, to a new image of size [target_size, target_size]. 
    # Normalizes the output image to be zero-mean, and in the [-1, 1] range.
    
    resized_image = cv2.resize(input_image, dsize=(target_size,target_size))
    output_image = cv2.normalize(src=resized_image,dst=None,alpha=-1,beta=1, norm_type=cv2.NORM_MINMAX)

    return output_image


def reportAccuracy(true_labels, predicted_labels):
    # generates and returns the accuracy of a model

    # true_labels is a N x 1 list, where each entry is an integer
    # and N is the size of the testing set.
    # predicted_labels is a N x 1 list, where each entry is an 
    # integer, and N is the size of the testing set. These labels 
    # were produced by your system.

    total = len(true_labels)
    num_correct = 0
    for i in range(total):
    	if true_labels[i] == predicted_labels[i]:
		    num_correct = num_correct + 1

    accuracy = num_correct/total*100

    # accuracy is a scalar, defined in the spec (in %)
    return accuracy

def getCenters(descriptors, labels):
    vocabulary = []

    groups = {}
    counts = {}
    labels = labels.tolist()

    rows = len(labels)
    cols = len(labels[0])

    for i in labels:
        if i not in groups:
            groups[i] = [0] * cols
            counts[i] = 0
    
    for i, des in enumerate(descriptors):
        counts[labels[i]] = counts[labels[i]] + 1
        for j, drow in enumerate(des):
            groups[labels[i]][j] += drow.item()

    for i in range(len(groups)):
        label = groups[i]
        for j in range(len(label)):
            label[j] /= counts[i]
        vocabulary.append(label)

    return vocabulary

def buildDict(train_images, dict_size, feature_type, clustering_type):
    # this function will sample descriptors from the training images,
    # cluster them, and then return the cluster centers.

    # train_images is a list of N images, represented as 2D arrays
    # dict_size is the size of the vocabulary,
    # feature_type is a string specifying the type of feature that we are interested in.
    # Valid values are "sift", "surf" and "orb"
    # clustering_type is one of "kmeans" or "hierarchical"

    # the output 'vocabulary' should be a list of length dict_size, with elements of size d, where d is the 
    # dimention of the feature. each row is a cluster centroid / visual word.
    vocabulary = []
    n = len(train_images)
    features = []
    if feature_type == "sift":
        for i in range(n):
            sift = cv2.xfeatures2d.SIFT_create()
            kp_sift, descriptors = sift.detectAndCompute(train_images[i], None)
            for des in descriptors:
                features.append(des)  
    if feature_type == "surf":
        for i in range(n):
            surf = cv2.xfeatures2d.SURF_create()
            kp_surf, descriptors = surf.detectAndCompute(train_images[i], None)
            for des in descriptors:
                features.append(des) 
    if feature_type  == "orb":
        for i in range(n):
            orb = cv2.ORB_create()
            kp_orb, descriptors = orb.detectAndCompute(train_images[i], None)
            if descriptors is None:
                continue
            for des in descriptors:
                features.append(des) 
    if clustering_type == "kmeans":
        kmeans = cluster.KMeans(n_clusters=dict_size).fit(features)
        vocabulary = kmeans.cluster_centers_     

    # NOTE: Should you run out of memory or have performance issues, feel free to limit the 
    # number of descriptors you store per image.
    return vocabulary


def computeBow(image, vocabulary, feature_type):
    # extracts features from the image, and returns a BOW representation using a vocabulary

    # image is 2D array
    # vocabulary is an array of size dict_size x d
    # feature type is a string (from "sift", "surf", "orb") specifying the feature
    # used to create the vocabulary

    features = []

    if feature_type == "sift": 
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image,None) #dont care about keypoints
        for des in descriptors:
            features.append(des)    #128 numbers
    elif feature_type == "surf":
        surf = cv2.xfeatures2d.SURF_create()
        keypoints, descriptors = surf.detectAndCompute(image,None)
        for des in descriptors:
            features.append(des)    #64 numbers
    elif feature_type == "orb":
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(image,None)
        if des is None:
            features.append([0] * 32)   #if des is none, you still have to append list of 0's
        else:
            for des in descriptors:
                features.append(des)  #32 numbers

    num_bins = len(vocabulary)  
    Bow = [0] * num_bins
    size_magnitude = len(features)
    
    for feature in features:
        feature = np.reshape(feature, (1,-1)) 
        temp = spatial.distance.cdist(vocabulary, feature, "euclidean") #math stuff, find which bin to put it in
        which_bin = np.where(temp == np.amin(temp))[0][0]
        Bow[which_bin] = Bow[which_bin] + 1     #incrase it by one
    for i, bin in enumerate(Bow):
        Bow[i] = Bow[i] / float(size_magnitude)
    


def tinyImages(train_features, test_features, train_labels, test_labels):
    # Resizes training images and flattens them to train a KNN classifier using the training labels
    # Classifies the resized and flattened testing images using the trained classifier
    # Returns the accuracy of the system, and the overall runtime (including resizing and classification)
    # Does so for 8x8, 16x16, and 32x32 images, with 1, 3 and 6 neighbors
    
    # train_features is a list of N images, represented as 2D arrays
    # test_features is a list of M images, represented as 2D arrays
    # train_labels is a list of N integers, containing the label values for the train set
    # test_labels is a list of M integers, containing the label values for the test set

    # classResult is a 18x1 array, containing accuracies and runtimes, in the following order:
    # accuracies and runtimes for 8x8 scales, 16x16 scales, 32x32 scales
    # [8x8 scale 1 neighbor accuracy, 8x8 scale 1 neighbor runtime, 8x8 scale 3 neighbor accuracy, 
    # 8x8 scale 3 neighbor runtime, ...]
    # Accuracies are a percentage, runtimes are in seconds
    
    N = len(train_labels)
    M = len(test_labels)
	
    # create array of resized images
    train_8 = []
    train_16 = []
    train_32 = []
    test_8 = []
    test_16 = []
    test_32 = []

    start8resize = perf_counter()
    for i in range(0,N):
        train_8.append(imresize(train_features[i],8).flatten().tolist())
    for i in range(0,M):
        test_8.append(imresize(test_features[i],8).flatten().tolist())
    end8resize = perf_counter()

    start16resize = perf_counter()
    for i in range(0,N):
	    train_16.append(imresize(train_features[i],16).flatten().tolist())
    for i in range(0,M):
	    test_16.append(imresize(test_features[i],16).flatten().tolist())
    end16resize = perf_counter()

    start32resize = perf_counter()
    for i in range(0,N):
        train_32.append(imresize(train_features[i],32).flatten().tolist())
    for i in range(0,M):
        test_32.append(imresize(test_features[i],32).flatten().tolist())
    end32resize = perf_counter()

    start_8_1 = perf_counter()
    predicted_8_1 = KNN_classifier(train_8, train_labels, test_8, 1)   
    end_8_1 = perf_counter()
    start_8_3 = perf_counter()
    predicted_8_3 = KNN_classifier(train_8, train_labels, test_8, 3)   
    end_8_3 = perf_counter()
    start_8_6 = perf_counter()
    predicted_8_6 = KNN_classifier(train_8, train_labels, test_8, 6)   
    end_8_6 = perf_counter()
    start_16_1 = perf_counter()
    predicted_16_1 = KNN_classifier(train_16, train_labels, test_16, 1)   
    end_16_1 = perf_counter()
    start_16_3 = perf_counter()
    predicted_16_3 = KNN_classifier(train_16, train_labels, test_16, 3)   
    end_16_3 = perf_counter()
    start_16_6 = perf_counter()
    predicted_16_6 = KNN_classifier(train_16, train_labels, test_16, 6)   
    end_16_6 = perf_counter()
    start_32_1 = perf_counter()
    predicted_32_1 = KNN_classifier(train_32, train_labels, test_32, 1)   
    end_32_1 = perf_counter()
    start_32_3 = perf_counter()
    predicted_32_3 = KNN_classifier(train_32, train_labels, test_32, 3)   
    end_32_3 = perf_counter()
    start_32_6 = perf_counter()
    predicted_32_6 = KNN_classifier(train_32, train_labels, test_32, 6)   
    end_32_6 = perf_counter()

    accuracy_8_1 = reportAccuracy(test_labels, predicted_8_1)
    accuracy_8_3 = reportAccuracy(test_labels, predicted_8_3)
    accuracy_8_6 = reportAccuracy(test_labels, predicted_8_6)
    accuracy_16_1 = reportAccuracy(test_labels, predicted_16_1)
    accuracy_16_3 = reportAccuracy(test_labels, predicted_16_3)
    accuracy_16_6 = reportAccuracy(test_labels, predicted_16_6)
    accuracy_32_1 = reportAccuracy(test_labels, predicted_32_1)
    accuracy_32_3 = reportAccuracy(test_labels, predicted_32_3)
    accuracy_32_6 = reportAccuracy(test_labels, predicted_32_6)

    runtime_8_1 = end_8_1 - start_8_1 + end8resize - start8resize
    runtime_8_3 = end_8_3 - start_8_3 + end8resize - start8resize
    runtime_8_6 = end_8_6 - start_8_6 + end8resize - start8resize
    runtime_16_1 = end_16_1 - start_16_1 + end16resize - start16resize
    runtime_16_3 = end_16_3 - start_16_3 + end16resize - start16resize
    runtime_16_6 = end_16_6 - start_16_6 + end16resize - start16resize
    runtime_32_1 = end_32_1 - start_32_1 + end32resize - start32resize
    runtime_32_3 = end_32_3 - start_32_3 + end32resize - start32resize
    runtime_32_6 = end_32_6 - start_32_6 + end32resize - start32resize

    classResult = []
    classResult.extend([accuracy_8_1, runtime_8_1, accuracy_8_3, runtime_8_3, accuracy_8_6, runtime_8_6, 
        accuracy_16_1, runtime_16_1 , accuracy_16_3, runtime_16_3, accuracy_16_6, runtime_16_6, 
        accuracy_32_1, runtime_32_1 , accuracy_32_3, runtime_32_3, accuracy_32_6, runtime_32_6])
    print(classResult)
    return classResult
    
