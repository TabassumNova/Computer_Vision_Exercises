import cv2
import numpy as np
import pickle
import os
import random
from scipy import spatial
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import operator
import itertools

# FaceNet to extract face embeddings.
class FaceNet:

    def __init__(self):
        self.dim_embeddings = 128
        self.facenet = cv2.dnn.readNetFromONNX("resnet50_128.onnx")

    # Predict embedding from a given face image.
    def predict(self, face):
        # Normalize face image using mean subtraction.
        face = face - (131.0912, 103.8827, 91.4953)

        # Forward pass through deep neural network. The input size should be 224 x 224.
        reshaped = np.reshape(face, (1, 3, 224, 224))
        self.facenet.setInput(reshaped)
        embedding = np.squeeze(self.facenet.forward())
        return embedding / np.linalg.norm(embedding)

    # Get dimensionality of the extracted embeddings.
    def get_embedding_dimensionality(self):
        return self.dim_embeddings


# The FaceRecognizer model enables supervised face identification.
class FaceRecognizer:

    # Prepare FaceRecognizer; specify all parameters for face identification.
    def __init__(self, num_neighbours=11, max_distance=0.8, min_prob=0.5):
        # ToDo: Prepare FaceNet and set all parameters for kNN.
        self.facenet = FaceNet()
        self.num_neighbours = num_neighbours
        self.max_distance = max_distance
        self.min_prob = min_prob
        # The underlying gallery: class labels and embeddings.
        self.labels = []
        self.embeddings = np.empty((0, self.facenet.get_embedding_dimensionality()))

        # Load face recognizer from pickle file if available.
        if os.path.exists("recognition_gallery.pkl"):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        with open("recognition_gallery.pkl", 'wb') as f:
            pickle.dump((self.labels, self.embeddings), f)

    # Load trained model from a pickle file.
    def load(self):
        with open("recognition_gallery.pkl", 'rb') as f:
            (self.labels, self.embeddings) = pickle.load(f)

    # ToDo
    def update(self, face, label):
        self.embeddings = np.append(self.embeddings, [self.facenet.predict(face)], axis=0)
        self.labels.append(label)
        # self.save()
        return None

    # ToDo
    def predict(self, face):
        ## new
        self.load()
        # print('test:',self.embeddings.shape, len(self.labels))
        predicted_embedding = self.facenet.predict(face)
        # print('test:', predicted_embedding.shape, self.embeddings.shape)
        all_dist = spatial.distance.cdist(predicted_embedding.reshape(1,-1), self.embeddings.reshape(-1,self.facenet.get_embedding_dimensionality()))
        # print('all_dist:', all_dist.shape)
        rows, cols = np.where(all_dist <= self.max_distance)
        dist = []
        for i in range(len(rows)):
            # indices = [rows[i], cols[i]]
            dist.append(all_dist[rows[i], cols[i]])
        dist = np.array(dist)
        # print(all_dist)
        # dist = np.take(all_dist, indices)
        # print('dist:', dist)
        dist_dict = {e: dist[i] for i, e in enumerate(cols)}
        sorted_d = dict(sorted(dist_dict.items(), key=operator.itemgetter(1), reverse=False))
        final_dict = sorted_d if len(sorted_d) <= self.num_neighbours else dict(itertools.islice(sorted_d.items(), self.num_neighbours))
        key_list = np.array(list(final_dict.keys()))
        # print('key_list:',key_list)
        label = np.take(self.labels, key_list)
        unique, counts = np.unique(label, return_counts=True)
        max_c = np.argmax(counts)
        predicted_label = unique[max_c]
        # print()
        probability = counts[max_c]/self.num_neighbours
        min_distance = 0
        for i,e in enumerate(label):
            if e == predicted_label:
                index = key_list[i]
                min_distance = final_dict[index]
                break
        ## openset
        if min_distance > self.max_distance or probability < self.min_prob:
            predicted_label = 'unknown'
        ## end

        # model = KNeighborsClassifier(n_neighbors=self.num_neighbours)
        # self.load()
        # model.fit(self.embeddings, self.labels)
        # new_embedding = self.facenet.predict(face)
        # new_label = model.predict(new_embedding)
        # probability = model.predict_proba(new_embedding)
        # ## not sure
        # # distance = min(model.kneighbors(new_embedding,self.num_neighbours))
        # selected_embeddings = self.embeddings[np.where(self.labels == new_label)]
        # neigh = NearestNeighbors(n_neighbors=self.num_neighbours).fit(selected_embeddings)
        # dist, index = neigh.kneighbors(new_embedding)
        #
        # if dist>self.max_distance or probability<self.min_prob:
        #     new_label = 'unknown'

        return predicted_label, probability, min_distance


# The FaceClustering class enables unsupervised clustering of face images according to their identity and
# re-identification.
class FaceClustering:

    # Prepare FaceClustering; specify all parameters of clustering algorithm.
    def __init__(self,num_clusters=4, max_iter=25):
        # ToDo: Prepare FaceNet.
        self.facenet = FaceNet()
        # The underlying gallery: embeddings without class labels.
        self.embeddings = np.empty((0, self.facenet.get_embedding_dimensionality()))

        # Number of cluster centers for k-means clustering.
        self.num_clusters = num_clusters
        # Cluster centers.
        self.cluster_center = np.empty((num_clusters, self.facenet.get_embedding_dimensionality()))
        # Cluster index associated with the different samples.
        self.cluster_membership = []

        # Maximum number of iterations for k-means clustering.
        self.max_iter = max_iter

        # Load face clustering from pickle file if available.
        if os.path.exists("clustering_gallery.pkl"):
            self.load()
        self.count = 0

    # Save the trained model as a pickle file.
    def save(self):
        with open("clustering_gallery.pkl", 'wb') as f:
            pickle.dump((self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership), f)

    # Load trained model from a pickle file.
    def load(self):
        with open("clustering_gallery.pkl", 'rb') as f:
            (self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership) = pickle.load(f)

    # ToDo
    def update(self, face):
        if os.path.exists("clustering_gallery.pkl") and self.count==0:
            self.load()
            # print(len(self.embeddings))
            self.count = self.count + 1
        self.embeddings = np.append(self.embeddings, [self.facenet.predict(face)], axis=0)
        # print(len(self.embeddings))
        # self.embeddings = self.facenet.predict(face)
        # self.save()
        return None
    
    # ToDo
    @property
    def fit(self):
        ## new
        print(len(self.embeddings))
        if self.count==0:
            idx = np.random.randint(self.embeddings.shape[0], size=self.num_clusters)
            center = self.embeddings[idx,:]
            print('ramdom center')
        else:
            center = self.cluster_center
            print('pre_center')
        # center = np.random.randint(self.embeddings.reshape(-1,self.facenet.get_embedding_dimensionality()), size=self.num_clusters)
        member_dict = {}
        dist_dict = {}
        center_dict = {}
        for iteration in range(self.max_iter):
            center_dict[iteration] = center
            sum_dist = 0
            ## assignment
            for e in self.embeddings:
                dist = spatial.distance.cdist(e.reshape(1,-1), center.reshape(-1,self.facenet.get_embedding_dimensionality()))
                min_d = np.argmin(dist)
                sum_dist = sum_dist + dist[0, min_d]
                # print(min_d)
                if min_d not in list(member_dict.keys()):
                    member_dict[min_d] = [e]
                else:
                    member_dict[min_d].append(e)

            n_cluster = int(self.num_clusters)
            # print(n_cluster)
            # print(len(member_dict[0]))
            for c in range(self.num_clusters):
                s = 0
                if c in member_dict:
                    for i in member_dict[c]:
                        s = s + i
                    # s = [sum_+i for i in member_dict[c]]
                    # s = np.array(list(map(sum, zip(*member_dict[c]))))
                    center[c] = s/len(member_dict[c])
                else:
                    pass
            dist_dict[iteration] = sum_dist

        sorted_dist = dict(sorted(dist_dict.items(), key=operator.itemgetter(1), reverse=False))
        x = list(sorted_dist.keys())[0]
        centers = center_dict[x]
        labels = []
        for e in self.embeddings:
            dist = spatial.distance.cdist(e.reshape(1,-1), centers.reshape(-1,self.facenet.get_embedding_dimensionality()))
            min_d = np.argmin(dist)
            labels.append(min_d)

        self.cluster_center = centers
        self.cluster_membership = labels

        ## end
        # kmeans = KMeans(n_clusters=2, random_state=0).fit(self.embeddings)
        # self.cluster_center = kmeans.cluster_centers_
        # self.cluster_membership = kmeans.labels_
        return None

    # ToDo
    def predict(self, face):
        # print('test: ', self.cluster_center.shape)
        embeddings = self.facenet.predict(face)
        dist = spatial.distance.cdist(embeddings.reshape(1,-1), self.cluster_center.reshape(-1,self.facenet.get_embedding_dimensionality()))
        label = np.argmin(dist)
        # min_dist = dist[label]
        return label, dist.T
