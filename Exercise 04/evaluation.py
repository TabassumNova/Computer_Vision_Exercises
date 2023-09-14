import numpy as np
import pickle
from classifier import NearestNeighborClassifier
import csv
import pandas as pd

# Class label for unknown subjects in test and training data.
UNKNOWN_LABEL = -1


# Evaluation of open-set face identification.
class OpenSetEvaluation:

    def __init__(self,
                 classifier=NearestNeighborClassifier(),
                 false_alarm_rate_range=np.logspace(-3, 0, 1000, endpoint=True)):

        # The false alarm rates.
        self.false_alarm_rate_range = false_alarm_rate_range

        # Datasets (embeddings + labels) used for training and testing.
        self.train_embeddings = []
        self.train_labels = []
        self.test_embeddings = []
        self.test_labels = []

        # The evaluated classifier (see classifier.py)
        self.classifier = classifier

    # Prepare the evaluation by reading training and test data from file.
    def prepare_input_data(self, train_data_file, test_data_file):
        self.train_embeddings = np.array(pd.read_csv('train_embeddings.csv',header= None, skiprows = [0]))
        self.train_labels = np.array(pd.read_csv('train_labels.csv',header= None, skiprows = [0]))
        self.test_embeddings = np.array(pd.read_csv('test_embeddings.csv',header= None, skiprows = [0]))
        self.test_labels = np.array(pd.read_csv('test_labels.csv', header=None, skiprows=[0]))

        # with open(train_data_file, "rb") as f:
        #     (self.train_embeddings, self.train_labels) = pickle.load(f)
        # with open(test_data_file,"rb", encoding='iso-8859-1') as f:
        #     (self.test_embeddings, self.test_labels) = pickle.load(f)

    # Run the evaluation and find performance measure (identification rates) at different similarity thresholds.
    def run(self):
        similarity_thresholds = None
        identification_rates = None

        self.classifier.fit(self.train_embeddings, self.train_labels)
        prediction_labels, similarities = self.classifier.predict_labels_and_similarities(self.test_embeddings)
        ir_list = []
        for f in self.false_alarm_rate_range:
            similarity_thresholds = self.select_similarity_threshold(similarities, f)
            labels = []
            for p in range(len(self.test_labels)):
                if self.test_labels[p] != -1 and similarities[p] >= similarity_thresholds:
                    labels.append(self.test_labels[p])

            ir_list.append(self.calc_identification_rate(labels))
        identification_rates = np.array(ir_list)

        # Report all performance measures.
        evaluation_results = {'similarity_thresholds': similarity_thresholds,
                              'identification_rates': identification_rates}

        return evaluation_results

    def select_similarity_threshold(self, similarity, false_alarm_rate):
        similarity_thresholds = np.percentile(similarity, (1-false_alarm_rate)*100)
        return similarity_thresholds

    def calc_identification_rate(self, prediction_labels):
        condition = (self.test_labels != -1)
        extracted = np.extract(condition, self.test_labels)
        dir = len(prediction_labels)/len(extracted)
        return dir
