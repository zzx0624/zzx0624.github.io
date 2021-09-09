'''
Created on Aug 9, 2016
Keras Implementation of Multi-Layer Perceptron (GMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.  

@author: Xiangnan He (xiangnanhe@gmail.com)
'''

import numpy as np
from scipy.sparse import data

import theano
import theano.tensor as T
import keras
from keras import backend as K
from keras import initializations
from keras.regularizers import l2, activity_l2
from keras.regularizers import l2
from keras.models import Sequential, Graph, Model
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten, Dropout
from keras.layers import Embedding, Input, Dense, merge, Reshape, Flatten, Dropout
from keras.constraints import maxnorm
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from evaluate import evaluate_model
from dataset import Dataset
from time import time
import sys
import argparse
import multiprocessing as mp


#################### Arguments ####################

def init_normal(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)


def get_model(num_users, num_items, layers=[20, 10], reg_layers=[0, 0]):
    # 6040 3706 [64,32,16,8] [0,0,0,0]
    assert len(layers) == len(reg_layers)
    num_layer = len(layers)  # Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    # item_input = Input(shape=(1,), dtype='int32', name='item_input')
    q_input = Input(shape=(128,), dtype='float32', name='q_input')
    v_input = Input(shape=(128,), dtype='float32', name='v_input')
    MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=layers[0], name='user_embedding',
                                   init=init_normal, W_regularizer=l2(reg_layers[0]), input_length=1)
    # MLP_Embedding_Item = Embedding(input_dim=num_items, output_dim=layers[0] // 2, name='item_embedding',
    #                                init=init_normal, W_regularizer=l2(reg_layers[0]), input_length=1)
    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MLP_Embedding_User(user_input))
    # item_latent = Flatten()(MLP_Embedding_Item(item_input))
    item_latent = merge([q_input, v_input], mode='concat')
    # The 0-th layer is the concatenation of embedding layers
    vector = merge([user_latent, item_latent], mode='concat')
    # MLP layers

    for idx in range(1, num_layer):
        layer = Dense(layers[idx], W_regularizer=l2(reg_layers[idx]), activation='relu', name='layer%d' % idx)
        vector = layer(vector)

    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name='prediction')(vector)

    model = Model(input=[user_input, q_input, v_input],
                  output=prediction)

    return model


def get_train_instances(train, q_idx, v_idx, num_negatives):  # train,4
    user_input, item_input,q_input,v_input, labels = [], [], [], [], []
    num_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        # item_input.append(i)
        q_input.append(q_idx[i])
        v_input.append(v_idx[i])
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train:
                j = np.random.randint(num_items)
            user_input.append(u)
            # item_input.append(j)
            q_input.append(q_idx[j])
            v_input.append(v_idx[j])
            labels.append(0)
    # print('0000000000000000000000000000000000000')
    # print(user_input)
    # print(item_input)
    # print(labels)
    # print('00000000000000000000000000000000000000')
    return user_input, q_input, v_input, labels


def get_test_instances(train):  # train,4
    user_input, item_input, labels = [], [], []
    num_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances

    return user_input, item_input, labels


def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[128,64,32,16,8]',
                        help="Size of each layer. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0,0]',
                        help="Regularization for each layer")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    path = args.path  # default='Data/'
    dataset = args.dataset  # default='ml-1m',
    layers = eval(args.layers)  # default='[64,32,16,8]',
    reg_layers = eval(args.reg_layers)  # default='[0,0,0,0]',
    num_negatives = args.num_neg  # default=4,
    learner = args.learner  # default='adam',
    learning_rate = args.lr  # default=0.001,
    batch_size = args.batch_size  # default=256,
    epochs = args.epochs  # default=100,
    verbose = args.verbose  # default=1,

    topK = 10
    evaluation_threads = 1  # mp.cpu_count()

    print("MLP arguments: %s " % (args))
    model_out_file = 'Pretrain/%s_MLP_%s_%d.h5' % (args.dataset, args.layers, time())

    # Loading data
    t1 = time()
    # dataset = Dataset(args.path + args.dataset)
    dataset = Dataset(args.dataset)
    q = dataset.q
    v = dataset.v
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    test = dataset.testMatrix
    print('---------------------------------------------')
    print(train.shape)
    print(train)

    num_users, num_items = train.shape  # 6040 3706
    q_idx = {i: q[i] for i in range(num_items)}
    v_idx = {i: v[i] for i in range(num_items)}
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time() - t1, num_users, num_items, train.nnz, len(testRatings)))

    # Build model
    model = get_model(num_users, num_items, layers, reg_layers)
    if learner.lower() == "adagrad":
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')

        # Check Init performance
    t1 = time()
    (hits, ndcgs) = evaluate_model(model, q_idx, v_idx, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f [%.1f]' % (hr, ndcg, time() - t1))

    # Train model
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    for epoch in range(epochs):
        t1 = time()
        # Generate training instances
        user_input, q_input, v_input, labels = get_train_instances(train, q_idx, v_idx, num_negatives)

        # Training        
        hist = model.fit([np.array(user_input), np.array(q_input), np.array(v_input)],  # input
                         np.array(labels),  # labels
                         batch_size=batch_size, nb_epoch=1, verbose=0, shuffle=True)
        print(hist.history)
        print(hist.history['loss'])
        print(hist.history.items())
        # user_input1, item_input1, labels1 = get_test_instances(test)
        # testloss = model.evaluate([np.array(user_input1), np.array(item_input1)],  # input
        #                  np.array(labels1),batch_size=batch_size, verbose=0)
        # print('-------------------------------------------------')
        # print(testloss)
        t2 = time()

        # Evaluation
        if epoch % verbose == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
                  % (epoch, t2 - t1, hr, ndcg, loss, time() - t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
    if args.out > 0:
        print("The best MLP model is saved to %s" % (model_out_file))
