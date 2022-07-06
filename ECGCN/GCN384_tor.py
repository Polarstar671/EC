# -*-coding:utf-8-*-

from lib import gcn_model_384, graph, coarsening, utils
import numpy as np
import matplotlib.pyplot as plt
import MySQLdb
import math
import random

from utils import *

def loadTrafficData(packetNumsPerFlow=200, regions=80, originA=None, types=12, zeroCode=189.0):  # orignA 没用上，说明它不是活跃变量


    conn = MySQLdb.connect("localhost", "root", "1314wacyj!!!", "data_test", charset='utf8')
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
    print("sampleSize:",sampleSize)
    rawSampleInput = np.full((sampleSize, packetNumsPerFlow, 2), zeroCode)

    sql1 = 'select (pkt_length*flow_seq+1514)/8, FLOW_ID from iscx_tor where flow_id in (select flow_id from iscx_tor group by flow_id having count(*) >= 30) ' \
           'order by FLOW_ID, PKT_ID asc '

    cur.execute(sql1)
    infos = cur.fetchall()
    rawList = []

    curFlow_id = infos[0][1]
    sampleId = 0

    curCount = 0
    for temp_info in infos:
        tempFlow_id = temp_info[1]
        if curFlow_id == tempFlow_id:
            curCount = curCount + 1
            if curCount >= packetNumsPerFlow:
                continue
        else:
            flowSizeList.append(curCount)
            sampleId = sampleId + 1
            curCount = 1

        tempValue = math.modf(float(temp_info[0]))
        rawSampleInput[sampleId][curCount - 1][0] = tempValue[1]
        rawSampleInput[sampleId][curCount - 1][1] = tempValue[0]


        curFlow_id = tempFlow_id

    if curCount < packetNumsPerFlow:
        flowSizeList.append(curCount)

    cur.close()
    conn.close()
    print('len(rawList):', len(rawList))

    return rawSampleInput, np.array(rawLabelList).reshape(sampleSize), flowSizeList


def failRawData(rawData, failNum, noiseType, flowSizeList, zeroCode=int(1514 / 16)):
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
                    rawData[i][j] = zeroCode
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



def standardizeData(rawMatrix):
    for i in range(rawMatrix.shape[3]):
        rawMatrix[:, :, :, i] = (rawMatrix[:, :, :, i] - np.mean(rawMatrix[:, :, :, i])) / np.std(rawMatrix[:, :, :, i])
    return rawMatrix




def calSimilarity(C=40, packetNumsPerFlow=1000):
    conn = MySQLdb.connect("localhost", "root", "********", "*****", charset='utf8')
    cur = conn.cursor()

    baseDataMatrix = np.zeros((C, packetNumsPerFlow))
    sql1 = 'select flow_id, (pkt_length*flow_seq+1514)/8 from iscx_tor where flow_id in (select flow_id from iscx_tor group by flow_id having count(*) >= 30)' \
           'order by flow_id, pkt_id asc '
    cur.execute(sql1)
    infos = cur.fetchall()
    curFlow_id = infos[0][0]

    i = -1
    for temp_info in infos:
        tempFlow_id = temp_info[0]
        if curFlow_id == tempFlow_id:
            tempRegionIdx = int(temp_info[1])
            i = i + 1
            if i >= packetNumsPerFlow:
                continue
        else:
            i = 0
            curFlow_id = tempFlow_id

        baseDataMatrix[tempRegionIdx, i] = baseDataMatrix[tempRegionIdx, i] + 1
    cur.close()
    conn.close()

    corrMatrix = np.corrcoef(baseDataMatrix)
    corrMatrix = np.abs(fill_ndarray(corrMatrix))
    return corrMatrix


def runModel(filterFunctionName, isDAE=1):
    packetNumsPerFlow = 1000
    C = 384
    initialAjacency = calSimilarity(C, packetNumsPerFlow)
    X, y, flowSizeList = loadTrafficData(packetNumsPerFlow, C)

    n = X.shape[0]

    index = [i for i in range(n)]
    random.shuffle(index)
    X = X[index, :]
    y = y[index]

    number = np.zeros((15,))
    for i in range(n):
        label = y[i]
        number[label] += 1
    print(number)

    X_new = X
    y_new = y

    X_train = []
    X_val = []
    X_test = []
    y_train = []
    y_val = []
    y_test = []

    number2 = np.zeros((15,))
    for i in range(n):
        label = y_new[i]
        number2[label] += 1

        if label >= 12:
            y_new[i] = 12

        if number2[label] <= number[label] * 0.8:
            X_train.append(X_new[i])
            y_train.append(y_new[i])
        elif number2[label] <= number[label] * 0.9:
            X_val.append(X_new[i])
            y_val.append(y_new[i])
        else:
            X_test.append(X_new[i])
            y_test.append(y_new[i])

    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    n_train = X_train.shape[0]

    params = dict()
    params['dir_name'] = 'demo'
    params['num_epochs'] = 200
    params['batch_size'] = 50
    params['eval_frequency'] = int(X_train.shape[0] / params['batch_size'])

    params['filter'] = filterFunctionName
    params['brelu'] = 'b1relu'
    params['pool'] = 'apool1'
    params['Cout_temporal'] = [16, 16, 16]
    params['Cout_spatial'] = [4, 4, 4]
    params['K_temporal'] = [4, 4, 4]
    params['K_spatial'] = [4, 4, 4]
    params['p'] = []
    params['M'] = [284, C]
    params['regularization'] = 5e-4
    params['dropout'] = 0.8
    params['learning_rate'] = 3e-3
    params['decay_rate'] = 0.995
    params['momentum'] = 0
    params['decay_steps'] = n_train / params['batch_size']
    params['originL'] = None
    params['scale_'] = 0
    params['mean_'] = 0
    params['AjacentMatrix'] = initialAjacency

    model = gcn_model_384.cgcnn(**params)
    loss, t_step = model.fit(X_train, y_train, X_val, y_val, types=13)

    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    axes.plot(loss, 'g.-')
    axes.set_ylabel('training loss', color='g')
    string, loss, predictions = model.evaluate(X_test, y_test, types=13)

    rightNum = 0
    for temp_idx in range(y_test.shape[0]):
        if np.argmax(predictions[temp_idx]) == y_test[temp_idx]:
            rightNum = rightNum + 1

    precision_array = np.zeros((predictions.shape[1]))
    recall_array = np.zeros((predictions.shape[1]))
    f1_array = np.zeros((predictions.shape[1]))
    for i in range(f1_array.shape[0]):
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        ans = 0
        for j in range(y_test.shape[0]):
            label = y_test[j]
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
        precision_array[i] = TP/(TP+FP+0.01)
        recall_array[i] = TP/(TP+FN+0.01)
        print(TP, FP, TN, FN, ans)

    cnt = 0
    for i in range(f1_array.shape[0]):
        if precision_array[i]+recall_array[i] == 0:
            continue
        cnt += 1
        f1_array[i] = (2*precision_array[i]*recall_array[i])/(precision_array[i]+recall_array[i])

    print('测试分类准确率为:%.2f', rightNum / y_test.shape[0])
    print('测试分类平均precision为:%.2f', np.sum(precision_array)/cnt)
    print('测试分类平均recall为:%.2f', np.sum(recall_array)/cnt)
    print('测试分类平均f1_score为:%.2f', np.sum(f1_array)/cnt)

    plt.legend()
    plt.show()


if __name__ == '__main__':
    runModel('temporalSpatialBlock')
