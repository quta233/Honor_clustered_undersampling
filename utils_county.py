import glob
import numpy as np
import torch
import os
from torch.utils.data import random_split
from sklearn.cluster import KMeans
import random
import cv2 as cv
from kneed import KneeLocator

NETWORK_FOLDER = "/mnt/windmills/images/2017/IA/tif/256/"

NETWORK_FOLDER2 = "/work/windmills/256/"


def get_trainset_random(label, major_file, min_size):
    zipped_maj = list(zip(major_file, label))
    zipped_maj.sort(key=lambda tup: tup[1])
    sorted_maj, sorted_maj_labels = zip(*zipped_maj)
    sorted_maj, sorted_maj_labels = np.asarray(sorted_maj), np.asarray(sorted_maj_labels)

    # print(sorted_maj, sorted_maj_labels)
    unique, counts = np.unique(label, return_counts=True)
    k_inds = dict(zip(unique, counts))
    #print(k_inds)

    majority_size = len(major_file)
    rebalanced_train = []
    ptr = 0
    # check = 0
    for i in k_inds.keys():
        ct = k_inds[i]
        p = round(min_size * (ct / majority_size))
        cur_name = sorted_maj[ptr:ptr + ct]
        random.shuffle(cur_name)
        rebalanced_train = np.append(rebalanced_train, cur_name[:p])
        ptr += ct
        # check += p

    # print(len(rebalanced_train))
    # print(check)
    # print(min_size)
    return rebalanced_train

def get_multi_random(label, major_file, min_size):
    zipped_maj = list(zip(major_file, label))
    zipped_maj.sort(key=lambda tup: tup[1])
    sorted_maj, sorted_maj_labels = zip(*zipped_maj)
    sorted_maj, sorted_maj_labels = np.asarray(sorted_maj), np.asarray(sorted_maj_labels)

    # print(sorted_maj, sorted_maj_labels)
    unique, counts = np.unique(label, return_counts=True)
    k_inds = dict(zip(unique, counts))
    # print(unique)

    majority_size = len(major_file)
    rebalanced_train = []
    rebalanced_label = []
    ptr = 0
    check = 0
    for i in k_inds.keys():
        ct = k_inds[i]
        p = round(min_size * (ct / majority_size))
        cur_name = sorted_maj[ptr:ptr + ct]
        cur_label = [i+1]*p
        rebalanced_train = np.append(rebalanced_train, cur_name[:p])
        rebalanced_label = np.append(rebalanced_label, cur_label)
        ptr += ct
        check += p

    # print(len(rebalanced_train))
    # print(check)
    # print(min_size)
    return rebalanced_train, rebalanced_label, len(unique)

def bow_voc(train_name, method):
    imgsKeypoints = {}

    # select out 10 features for each image
    kmeansTrainer = cv.BOWKMeansTrainer(10)
    maj_name = []
    bow_ext = ""
    if method == "orb":
        orb = cv.ORB_create()
        for filename in train_name:
            img = cv.imread(filename, 0)
            kp, des = orb.detectAndCompute(img, None)
            if des is not None:
                maj_name.append(filename)
                kmeansTrainer.add(des.astype(np.float32))
            imgsKeypoints[filename] = kp
        vocabulary = kmeansTrainer.cluster()

        orb2 = cv.ORB_create()
        bow_ext = cv.BOWImgDescriptorExtractor(orb2, cv.BFMatcher(cv.NORM_HAMMING))
        bow_ext.setVocabulary(vocabulary.astype(np.uint8))
    elif method == "sift":
        sift = cv.SIFT_create()
        for filename in train_name:
            img = cv.imread(filename, 0)
            kp, des = sift.detectAndCompute(img, None)
            if des is not None:
                maj_name.append(filename)
                kmeansTrainer.add(des)
            imgsKeypoints[filename] = kp
        vocabulary = kmeansTrainer.cluster()

        sift2 = cv.SIFT_create()
        bow_ext = cv.BOWImgDescriptorExtractor(sift2, cv.BFMatcher(cv.NORM_L1))
        bow_ext.setVocabulary(vocabulary)

    hist = []
    for f in maj_name:
        img = cv.imread(f, 0)
        histogram = bow_ext.compute(img, imgsKeypoints[f])
        if histogram is not None:
            histogram = histogram[0]
            hist.append(histogram)

    hist = np.array(hist)
    return hist, maj_name

def multi(maj_name, min_name, hist):
    ssd_total = []
    K = range(2, 60)
    for pre_k in K:
        k_cluster = KMeans(n_clusters=pre_k, n_init=30).fit(hist)
        compactness = k_cluster.inertia_
        ssd_total.append(compactness)

    # opt_k = optimal_num(ssd_total1)
    opt_k = KneeLocator(x=range(2, 60), y=ssd_total, curve='convex', direction='decreasing', S=1)
    print(opt_k.knee)
    k_cluster_opt = KMeans(n_clusters=opt_k.knee).fit(hist)
    label = np.array(k_cluster_opt.labels_)
    min_size = len(min_name)
    train_set, train_label, num_class = get_multi_random(label, maj_name, min_size)
    train_set = np.append(train_set, min_name)
    min_label = [0]*len(min_name)
    train_label = np.append(train_label, min_label)
    #print(train_label[:20], len(train_label))
    return train_set, train_label, num_class+1


def bin(maj_name, min_name, hist):
    ssd_total = []
    K = range(2, 60)
    for pre_k in K:
        k_cluster = KMeans(n_clusters=pre_k, n_init=30).fit(hist)
        compactness = k_cluster.inertia_
        ssd_total.append(compactness)

    # opt_k = optimal_num(ssd_total1)
    opt_k = KneeLocator(x=range(2, 60), y=ssd_total, curve='convex', direction='decreasing', S=1)
    print(opt_k.knee)
    k_cluster_opt = KMeans(n_clusters=opt_k.knee).fit(hist)
    label = np.array(k_cluster_opt.labels_)

    ## find points by distance
    # X_dist = k_cluster_opt.transform(hist) ** 2
    # center_dist = np.array([X_dist[i][x] for i, x in enumerate(label)])

    min_size = len(min_name)
    train_set = get_trainset_random(label, maj_name, min_size)
    train_set = np.append(train_set, min_name)
    # print(train_set[:20], len(train_set))
    return train_set


def lin(maj_name, min_name, hist):
    pre_k = len(min_name)
    k_cluster = KMeans(n_clusters=pre_k).fit(hist)

    # no feature selection part here, add for image
    label = np.array(k_cluster.labels_)
    X_dist = k_cluster.transform(hist) ** 2
    center_dist = np.array([X_dist[i][x] for i, x in enumerate(label)])

    zipped_maj = list(zip(maj_name, label, center_dist))
    zipped_maj.sort(key=lambda x: (x[1], x[2]))
    sorted_maj, sorted_maj_labels, sorted_maj_dis = zip(*zipped_maj)
    sorted_maj = np.asarray(sorted_maj)
    sorted_maj_labels = np.asarray(sorted_maj_labels)
    sorted_maj_dis = np.asarray(sorted_maj_dis)
    #print(sorted_maj[:10], sorted_maj_labels[:10], sorted_maj_dis[:10])
    unique, counts = np.unique(label, return_counts=True)
    k_inds = dict(zip(unique, counts))

    newmaj_name = []
    ptr = 0

    for i in k_inds.values():
        cur_name = sorted_maj[ptr]
        newmaj_name.append(cur_name)
        ptr += i

    # newtrain_name = newmaj_name + min_name
    newtrain_name = np.append(newmaj_name, min_name)
    # print(newtrain_name[:20], len(newtrain_name))
    return newtrain_name


def randomtrain(maj_name, min_name, r_seed):
    min_len = len(min_name)
    rest_len = len(maj_name) - min_len
    newmaj_name, rest_name = random_split(maj_name, [min_len, rest_len], generator=torch.Generator().manual_seed(r_seed))
    #print(min_len, len(newmaj_name))
    newtrain_name = newmaj_name + min_name
    return newtrain_name


def splitData(tvmaj_set, tvmin_set, r_seed):
    # CWD = os.getcwd()
    # os.chdir(imdir)
    validatemaj_size = round(len(tvmaj_set) * 0.2)
    validatemin_size = round(len(tvmin_set) * 0.15)
    # print(len(tvmaj_set), len(tvmin_set))
    trainmaj_size = len(tvmaj_set) - validatemaj_size
    trainmin_size = len(tvmin_set) - validatemin_size
    trainmaj_name, validatemaj_name = random_split(tvmaj_set, [trainmaj_size, validatemaj_size], generator=torch.Generator().manual_seed(r_seed))
    trainmin_name, validatemin_name = random_split(tvmin_set, [trainmin_size, validatemin_size], generator=torch.Generator().manual_seed(r_seed))
    validate_name = np.append(validatemaj_name, validatemin_name)
    # os.chdir(CWD)
    return trainmaj_name, trainmin_name, validate_name
