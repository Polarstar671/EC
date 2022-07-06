import random
import sys

import MySQLdb
import numpy as np


sys.path.append('..')

import datetime

from torch import nn
from torch.utils.data import DataLoader
from NewdataSet import *


def load_data():
    """Load and shape the dataset"""
    conn = MySQLdb.connect("localhost", "root", "***************", "******", charset='utf8')
    cur = conn.cursor()
    flowSizeList = []


    sql2 = 'select max(LABEL) from iscx_tor where flow_id in (select flow_id from iscx_tor group by flow_id having count(*) >= 30) ' \
           'group by FLOW_ID order by FLOW_ID asc '

    cur.execute(sql2)
    infos = cur.fetchall()
    rawLabelList = []
    for temp_info in infos:
        rawLabelList.append(int(temp_info[0]))
    sampleSize = int(len(rawLabelList))
    labels = np.array(rawLabelList).reshape(sampleSize)


    sql1 = 'select pkt_length, FLOW_ID from iscx_tor where flow_id in (select flow_id from iscx_tor group by flow_id having count(*) >= 30) ' \
           'order by FLOW_ID, PKT_ID asc '
    cur.execute(sql1)
    infos = cur.fetchall()
    curFlow_id = infos[0][1]
    sampleId = 0
    curCount = 0
    zeroCode = 0
    data = np.full((sampleSize, 1000), zeroCode)
    for temp_info in infos:
        tempFlow_id = temp_info[1]
        if curFlow_id == tempFlow_id:
            curCount = curCount + 1
        else:
            flowSizeList.append(curCount)
            sampleId = sampleId + 1
            curCount = 1

        data[sampleId][curCount - 1] = temp_info[0]
        curFlow_id = tempFlow_id

    if curCount<1000:
        flowSizeList.append(curCount)

    cur.close()
    conn.close()



    return data, labels, flowSizeList


def failRawData(rawData, failNum, noiseType, flowSizeList, zeroCode=int(1514 / 32)):
    sampleSize = rawData.shape[0]
    packseq = rawData.shape[1]

    if noiseType == 0:
        for i in range(sampleSize):
            curFlowSize = flowSizeList[i]

            tempList = []
            for j in range(curFlowSize):
                tempList.append(rawData[i][j])

            for j in range(int(failNum * curFlowSize)):
                curSampleLocation = random.randint(0, curFlowSize - j - 1)
                tempList.pop(curSampleLocation)

            for j in range(packseq):
                if j < len(tempList):
                    rawData[i][j] = tempList[j]
                else:
                    rawData[i][j] = 0.
    elif noiseType == 1:
        for i in range(sampleSize):
            curFlowSize = flowSizeList[i]

            tempList = []
            for j in range(curFlowSize):
                tempList.append(rawData[i][j])

            curFlowIdx = [j for j in range(curFlowSize)]
            sampLocations = random.sample(curFlowIdx, int(failNum * curFlowSize))
            AList = sampLocations.copy()
            sampLocations.sort()
            rawData[i][sampLocations] = rawData[i][AList]
    return rawData



class rnn_classify(nn.Module):
    def __init__(self, in_feature=50, hidden_feature=100, num_class=13, num_layers=2):
        super(rnn_classify, self).__init__()
        self.rnn = nn.LSTM(in_feature, hidden_feature, num_layers)
        self.classifier = nn.Linear(hidden_feature, num_class)

    def forward(self, x):
        x = x.reshape((batch_size, 50, 20))
        x = x.permute(2, 0, 1).float()
        out, _ = self.rnn(x)
        out = out[-1, :, :]
        out = self.classifier(out)
        return out


def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total


def train(net, train_data, valid_data, num_epochs, optimizer, criterion):
    if torch.cuda.is_available():
        net = net.cuda()
    prev_time = datetime.datetime.now()
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        net = net.train()
        for im, label in train_data:

            if im.shape[0] < batch_size:
                continue
            if torch.cuda.is_available():
                im = im.cuda()
                label = label.cuda()
            output = net(im)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += get_acc(output, label)
        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)

        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im, label in valid_data:
                if im.shape[0] < batch_size:
                    continue
                if torch.cuda.is_available():
                    im = im.cuda()
                    label = label.cuda()
                output = net(im)
                loss = criterion(output, label)
                valid_loss += loss.item()
                valid_acc += get_acc(output, label)
            epoch_str = (
            "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
            % (epoch, train_loss / len(train_data),
            train_acc / len(train_data), valid_loss / len(valid_data),
            valid_acc / len(valid_data)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
            (epoch, train_loss / len(train_data),
            train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str)

    # 测试集
    predictions = []
    labels = []
    if test_data is not None:
        net = net.eval()
        for im, label in test_data:
            if im.shape[0] < batch_size:
                continue
            if torch.cuda.is_available():
                im = im.cuda()
                label = label.cuda()
            output = net(im)
            predictions.extend(output.detach().cpu().numpy().tolist())
            labels.extend(label.detach().cpu().numpy().tolist())

        predictions = np.array(predictions)
        print(predictions.shape)
        labels = np.array(labels)
        print(labels.shape)
        rightNum = 0
        for temp_idx in range(predictions.shape[0]):
            if np.argmax(predictions[temp_idx]) == labels[temp_idx]:
                rightNum = rightNum + 1

        precision_array = np.zeros((13, ))
        recall_array = np.zeros((13, ))
        f1_array = np.zeros((13, ))
        for i in range(f1_array.shape[0]):
            TP = 0
            FP = 0
            TN = 0
            FN = 0
            ans = 0
            for j in range(predictions.shape[0]):
                label = labels[j]
                if label == i:
                    ans += 1
                pre = np.argmax(predictions[j])
                if label == i and pre == i:
                    TP += 1
                if label == i and pre != i:
                    FN += 1
                if label != i and pre == i:
                    FP += 1
                if label != i and pre != i:
                    TN += 1
            if TP == 0:
                precision_array[i] = 0
                recall_array[i] = 0
            else:
                precision_array[i] = TP / (TP + FP)
                recall_array[i] = TP / (TP + FN)
            print(TP, FP, TN, FN, ans)

        cnt = 0
        for i in range(f1_array.shape[0]):
            if precision_array[i] + recall_array[i] == 0:
                f1_array[i] = 0
                continue
            cnt += 1
            f1_array[i] = (2 * precision_array[i] * recall_array[i]) / (precision_array[i] + recall_array[i])

        for i in range(f1_array.shape[0]):
            print(precision_array[i], recall_array[i], f1_array[i])

        print('accuracy:%.2f', rightNum / y_test.shape[0])
        print('average precision为:%.2f', np.sum(precision_array) / f1_array.shape[0])
        print('average  recall为:%.2f', np.sum(recall_array) / f1_array.shape[0])
        print('average f1_score为:%.2f', np.sum(f1_array) / f1_array.shape[0])



print(torch.cuda.is_available())
X, y, flowSizeList = load_data()

for i in range(len(X)):
    label = y[i]
    if label >= 12:
        y[i] = 12

print(flowSizeList)

X=failRawData(X, 0.2, 1, flowSizeList)

n = X.shape[0]

index = [i for i in range(n)]
random.shuffle(index)
X = X[index, :]
y = y[index]


n_train = n * 1 // 2
n_val = n * 1 // 4
X_train = X[:n_train]
X_val = X[n_train:n_train + n_val]
X_test = X[n_train + n_val:]

y_train = y[:n_train]
y_val = y[n_train:n_train + n_val]
y_test = y[n_train + n_val:]

train_set = dataset_prediction(X_train, y_train)
val_set = dataset_prediction(X_val, y_val)
test_set = dataset_prediction(X_test, y_test)

batch_size = 64
epoch = 100
train_data = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
val_data = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)
test_data = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

net = rnn_classify()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adadelta(net.parameters(), 1e-1)

print("Start Train!")
train(net, train_data, val_data, epoch, optimizer, criterion)




