'''
Created on Aug 8, 2016
Processing datasets.

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import scipy.sparse as sp
import numpy as np
import csv

class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        self.trainMatrix = self.load_rating_file_as_matrix(path + ".train.rating")
        self.testMatrix = self.load_file_as_matrix(path + ".test.rating")
        # self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
        # self.testNegatives = self.load_negative_file(path + ".test.negative")
        # assert len(self.testRatings) == len(self.testNegatives)

        self.num_users, self.num_items = self.trainMatrix.shape

    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1:]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_rating_file_as_matrix(self, data):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open("traindata.csv", 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for low in reader:
                u, i = int(low[1]), int(low[3])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
        # Construct matrix
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        with open("traindata.csv", 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for low in reader:
                user, item, rating = int(low[1]), int(low[3]), float(low[10])
                if (rating > 0):
                    mat[user, item] = 1.0
        return mat

    def load_file_as_matrix(self, data):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open("testdata.csv", 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for low in reader:
                u, i = int(low[1]), int(low[3])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
        # Construct matrix
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        with open("testdata.csv", 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for low in reader:
                user, item, rating = int(low[1]), int(low[3]), float(low[10])
                if (rating > 0):
                    mat[user, item] = 1.0
        return mat
