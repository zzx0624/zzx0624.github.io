from dataset import Dataset
import argparse
# import torch
# import numpy as np
import math
# from torch.optim.lr_scheduler import ExponentialLR
import time
from xxx import *
from torch.optim.lr_scheduler import ExponentialLR
import heapq  # for retrieval topK

class Experiment:
    def __init__(self, learning_rate=0.0005, layers=[64, 32, 16, 8], reg_layers=[0, 0, 0, 0], verbose=1, num_neg=4,
                 tq=2, tv=2, to=3, epochs=700, batch_size=128, decay_rate=0., cuda=False, label_smoothing=0.):
        self.tq = tq
        self.tv = tv
        self.to = to
        self.epochs = epochs
        self.label_smoothing = label_smoothing
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.layers = layers
        self.reg_layers = reg_layers
        self.verbose = verbose
        self.num_neg = num_neg
        self.cuda = cuda

    def get_batch(self, q_input, v_input, users, labels, idx):
        q_input = np.array(q_input).astype(float)
        v_input = np.array(v_input).astype(float)
        users = np.array(users)
        labels = np.array(labels)
        input_q = torch.tensor(q_input[idx:idx + self.batch_size])
        input_v = torch.tensor(v_input[idx:idx + self.batch_size])
        user_batch = torch.LongTensor(users[idx:idx + self.batch_size])
        label_batch = torch.LongTensor(labels[idx:idx + self.batch_size])
        label_batch = label_batch.to(torch.float32)
        input_q = input_q.to(torch.float32)
        input_v = input_v.to(torch.float32)
        # print(user_batch)
        # print(item_batch)
        # print(label_batch)
        if self.cuda:
            input_q = input_q.cuda()
            input_v = input_v.cuda()
            user_batch = user_batch.cuda()
            label_batch = label_batch.cuda()
        return input_q, input_v, user_batch, label_batch

    def get_train_instances(self, train, q_idx, v_idx, num_negatives):  # train,4
        user_input, item_input, q_input, v_input, labels = [], [], [], [], []
        x = []
        num_items = train.shape[1]
        for (u, i) in train.keys():
            # positive instance
            user_input.append(u)
            item_input.append(i)
            # q_input.append(q_idx[i])
            # v_input.append(v_idx[i])
            labels.append(1)
            # negative instances
            for t in range(num_negatives):
                j = np.random.randint(num_items)
                while (u, j) in train:
                    j = np.random.randint(num_items)
                user_input.append(u)
                item_input.append(j)
                # q_input.append(q_idx[j])
                # v_input.append(v_idx[j])
                labels.append(0)
        for i in range(len(user_input)):
            x.append([user_input[i], item_input[i], labels[i]])
        x = np.array(x)
        np.random.shuffle(x)
        user_input, labels = [], []
        for i in range(len(x)):
            user_input.append(x[i][0])
            q_input.append(q_idx[x[i][1]])
            v_input.append(v_idx[x[i][1]])
            labels.append(x[i][2])
        # print('0000000000000000000000000000000000000')
        # print(user_input)
        # print(item_input)
        # print(labels)
        # print('00000000000000000000000000000000000000')
        return user_input, q_input, v_input, labels
    def get_test_instances(self, train, q_idx, v_idx):  # train,4
        user_input, q_input, v_input, labels = [], [], [], []
        num_items = train.shape[1]
        for (u, i) in train.keys():
            # positive instance
            user_input.append(u)
            # item_input.append(i)
            q_input.append(q_idx[i])
            v_input.append(v_idx[i])
            labels.append(1)
            # negative instances
        # print('0000000000000000000000000000000000000')
        # print(user_input)
        # print(item_input)
        # print(labels)
        # print('00000000000000000000000000000000000000')
        return q_input, v_input, user_input, labels
    def train_and_eval(self):
        max = 0
        mm = 0
        self.testRatings, self.testNegatives = dataset.testRatings, dataset.testNegatives
        num_users, num_items = train.shape  # 6040 3706
        q_idx = {i: q[i] for i in range(num_items)}
        v_idx = {i: v[i] for i in range(num_items)}
        t1 = time.time()
        print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
              % (time.time() - t1, num_users, num_items, train.nnz, len(testRatings)))
        print("Training the model...")
        model = Model(train, self.tq, self.tv, self.to, [64, 32, 16, 8])
        if self.cuda:
            model.cuda()
        model.init()
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)  # 学习率衰减函数
        print("Starting training...")
        for it in range(1, self.epochs + 1):
            print("第%d次迭代：" % it)
            # np.random.shuffle(train)
            user_input, q_input, v_input, labels = self.get_train_instances(train, q_idx, v_idx, 2)
            start_train = time.time()
            model.train()
            losses = []

            for j in range(0, len(labels), self.batch_size):
                q_batch, v_batch, user_batch, targets = self.get_batch(q_input, v_input, user_input, labels, j)
                opt.zero_grad()
                predictions = model.forward(q_batch, v_batch, user_batch)
                # if self.label_smoothing:
                #     targets = ((1.0 - self.label_smoothing) * targets) + 0.05
                loss = model.loss(predictions, targets) \
                       # + 0.5 * model.get_reg()
                loss.backward(retain_graph=True)  # 反向传播，计算当前梯度；
                # sensitivety = 1
                # epsilon = 1
                # model.Tc.grad = self.laplace_mech(model.Tc.grad, sensitivety, epsilon)
                opt.step()  # 根据梯度更新网络参数
                losses.append(loss.item())
            if self.decay_rate:
                scheduler.step()
            print(time.time() - start_train)
            print(np.mean(losses))
            model.eval()
            with torch.no_grad():  # 一个上下文管理器，被该语句内部的语句将不会计算梯度
                if not it % 1:
                    print("Test:")
                    start_test = time.time()
                    self.test(model, q_idx, v_idx)
                    (hits, ndcgs) = self.evaluate(model, q_idx, v_idx)
                    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
                    print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f'
                          % (it, time.time() - start_test, hr, ndcg))
                    if hr > max:
                        max = hr
                    if ndcg > mm:
                        mm = ndcg
                    print('max hr = %.4f, ndcg=%.4f'% (max, mm))
                    # self.test2(model)
                    # self.evaluate(model)

    def noisyCount(self, sensitivety, epsilon):
        beta = sensitivety / epsilon
        u1 = np.random.random()
        u2 = np.random.random()
        if u1 <= 0.5:
            n_value = -beta * np.log(1. - u2)
        else:
            n_value = beta * np.log(u2)
        return n_value

    def laplace_mech(self, data, sensitivety, epsilon):
        for i in range(len(data)):
            data[i] += self.noisyCount(sensitivety, epsilon)
        return data


    def test(self, model, q_idx, v_idx):
        q_input, v_input, user_input, labels = self.get_test_instances(test, q_idx, v_idx)
        print("Starting test...")
        losses = []
        for j in range(0, len(labels), self.batch_size):
            q_batch, v_input, user_batch, targets = self.get_batch(q_input, v_input, user_input, labels, j)
            predictions = model.forward(q_batch, v_input, user_batch)
            # if self.label_smoothing:
            #     targets = ((1.0 - self.label_smoothing) * targets) + 0.05
            loss = model.loss(predictions, targets)\
                   # + 0.5 * model.get_reg()
            losses.append(loss.item())
        print(np.mean(losses))

    def evaluate(self, model, q_idx, v_idx):
        hits, ndcgs = [], []
        for idx in range(len(self.testRatings)):
            q = []
            v = []
            rating = self.testRatings[idx]
            items = self.testNegatives[idx]
            u = rating[0]
            gtItem = rating[1]
            items.append(gtItem)
            for i in range(len(items)):
                q.append(q_idx[int(items[i])])
                v.append(v_idx[int(items[i])])
            # Get prediction scores
            map_item_score = {}
            users = np.full(len(items), u, dtype='int32')
            q_input = torch.tensor(np.array(q).astype(float))
            v_input = torch.tensor(np.array(v).astype(float))
            q_input = q_input.to(torch.float32)
            v_input = v_input.to(torch.float32)
            users = torch.LongTensor(users)
            if self.cuda:
                q_input = q_input.cuda()
                v_input = v_input.cuda()
                users = users.cuda()
            predictions = model.forward(q_input, v_input, users)
            for i in range(len(items)):
                item = items[i]
                map_item_score[item] = predictions[i]
            items.pop()

            # Evaluate top rank list
            ranklist = heapq.nlargest(10, map_item_score, key=map_item_score.get)
            hr = self.getHitRatio(ranklist, gtItem)
            ndcg = self.getNDCG(ranklist, gtItem)
            hits.append(hr)
            ndcgs.append(ndcg)
        return (hits, ndcgs)

    def getHitRatio(self, ranklist, gtItem):
        for item in ranklist:
            if item == gtItem:
                return 1
        return 0

    def getNDCG(self, ranklist, gtItem):
        for i in range(len(ranklist)):
            item = ranklist[i]
            if item == gtItem:
                return math.log(2) / math.log(i + 2)
        return 0
    # def test2(self, model):
    #     # test_user = data.test_user
    #     test_data = data.test_data
    #     test_user = test_data[:, (0, 1)]
    #     test_q = test_data[:, (2, 3, 4)]
    #     test_v = test_data[:, (5, 6, 7)]
    #     test_v = torch.tensor(test_v.astype(np.float32))
    #     test_q = torch.tensor(test_q.astype(np.float32))
    #     test_user = torch.tensor(test_user.astype(np.float32))
    #     if self.cuda:
    #         test_v = test_v.cuda()
    #         test_user = test_user.cuda()
    #         test_q = test_q.cuda()
    #     j = 0
    #     m = 0
    #     x = 0
    #     for i in range(len(test_data)):
    #         input_v = test_v[i].view(1, 3).long()
    #         input_q = test_q[i].view(1, 3).long()
    #         user = test_user[i].view(1, 2).long()
    #         prediction = model.forward(input_v, input_q, user)
    #         if prediction > 0.5:
    #             x = x + 1
    #         if int(test_data[i][8]) == 1:
    #             j = j + 1
    #             prediction1 = model.forward(input_v, input_q, user)
    #             # print(prediction)
    #             if prediction1 > 0.5:
    #                 m = m + 1
    #     P = -1
    #     R = -1
    #     F1 = -1
    #     if x != 0:
    #         P = m / j
    #         R = m / x
    #         F1 = (2 * P * R)/(P + R)
    #     print(m, j, x)
    #     print("召回率: {0}".format(P))
    #     print("准确率: {0}".format(R))
    #     print("F1:  {0}".format(F1))








if __name__== '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", type=str, default="douyin_dataset.csv", nargs="?",)
    parser.add_argument("--epochs", type=int, default=1000, nargs="?",
                        help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=2048, nargs="?",
                        help="Batch size.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument("--lr", type=float, default=0.0005, nargs="?",
                        help="Learning rate.")
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="Size of each layer. Note that the first layer is the concatenation of "
                             "user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each layer")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument("--tq", type=int, default=8, nargs="?",
                        help="The first dimension of the core tensor.")
    parser.add_argument("--tv", type=int, default=8, nargs="?",
                        help="The second dimension of the core tensor.")
    parser.add_argument("--to", type=int, default=8, nargs="?",
                        help="The third dimension of the core tensor.")
    parser.add_argument("--label_smoothing", type=float, default=0.01, nargs="?",
                        help="Amount of label smoothing.")
    parser.add_argument("--cuda", type=bool, default=True, nargs="?",
                        help="Whether to use cuda (GPU) or not (CPU).")
    parser.add_argument("--dr", type=float, default=0.0, nargs="?",
                        help="Decay rate.")
    args = parser.parse_args()
    experiment = Experiment(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr,
                            layers=args.layers, verbose=args.verbose, reg_layers=args.reg_layers,
                            num_neg=args.num_neg, tq=args.tq, tv=args.tv, to=args.to, cuda=args.cuda,
                            label_smoothing=args.label_smoothing, decay_rate=args.dr)
    torch.backends.cudnn.deterministic = True
    seed = 20
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)
    dataset = Dataset(args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    q = dataset.q
    v = dataset.v
    test = dataset.testMatrix
    experiment.train_and_eval()
