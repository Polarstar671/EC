#-*-coding:utf-8-*-

from lib import gcn_model_192, graph, coarsening, utils
import numpy as np
import matplotlib.pyplot as plt
import MySQLdb
import math
import random


def loadTrafficData(packetNumsPerFlow=200, regions=80, originA=None, types=30, zeroCode=189.0):
    conn = MySQLdb.connect("localhost", "root", "*******", "*******", charset='utf8')
    cur=conn.cursor()
    flowSizeList=[]

    sql2='select max(LABEL) from ALL_PACKETS_1000 group by FLOW_ID order by FLOW_ID asc'
    cur.execute(sql2)
    infos = cur.fetchall()
    rawLabelList=[]
    for temp_info in infos:
        rawLabelList.append(int(temp_info[0]))

    sampleSize=int(len(rawLabelList))        
    rawSampleInput = np.full((sampleSize, packetNumsPerFlow, 2), zeroCode)

    sql1='select (pkt_length*flow_seq+1514)/8, round(time,6), FLOW_ID  from ALL_PACKETS_1000 order by FLOW_ID, PKT_ID asc'
    cur.execute(sql1)
    infos = cur.fetchall()
    rawList=[]
    
    curFlow_id=infos[0][2]
    sampleId=0

    curCount=0
    for temp_info in infos:
        tempFlow_id=temp_info[2]
        if curFlow_id==tempFlow_id:
            curCount=curCount+1            
        else:
            flowSizeList.append(curCount)
            sampleId=sampleId+1
            curCount=1
        
        tempValue= math.modf(float(temp_info[0]))
        rawSampleInput[sampleId][curCount-1][0]=tempValue[1]
        rawSampleInput[sampleId][curCount-1][1]=tempValue[0]

        curFlow_id=tempFlow_id
        
    if curCount<packetNumsPerFlow:
        flowSizeList.append(curCount)                  
        
    cur.close()
    conn.close()
    print('len(rawList):',len(rawList))
    
    return rawSampleInput, np.array(rawLabelList).reshape(sampleSize), flowSizeList

    

def standardizeData(rawMatrix):

    
    for i in range(rawMatrix.shape[3]):
        rawMatrix[:,:,:,i] = (rawMatrix[:,:,:,i] - np.mean(rawMatrix[:,:,:,i]))/np.std(rawMatrix[:,:,:,i])
    return rawMatrix


def fill_ndarray(t1):
    for i in range(t1.shape[1]):
        temp_col = t1[:, i]
        nan_num = np.count_nonzero(temp_col != temp_col)
        if nan_num != 0:
            temp_col[np.isnan(temp_col)] = 0
    return t1

def calSimilarity(C=40, packetNumsPerFlow=1000):
    conn = MySQLdb.connect("localhost", "root", "******", "*****", charset='utf8')
    cur=conn.cursor()

    baseDataMatrix = np.zeros((C,packetNumsPerFlow))
    sql1='select flow_id, (pkt_length*flow_seq+1514)/8 from ALL_PACKETS_1000 order by flow_id, pkt_id asc'
    cur.execute(sql1)
    infos = cur.fetchall()
    curFlow_id=infos[0][0]

    i=-1
    for temp_info in infos:
        tempFlow_id=temp_info[0]

        if curFlow_id==tempFlow_id:
            tempRegionIdx=int(temp_info[1])
            i=i+1
        else:
            i=0
            curFlow_id=tempFlow_id

        baseDataMatrix[tempRegionIdx,i]=baseDataMatrix[tempRegionIdx,i]+1
    cur.close()
    conn.close()

    corrMatrix=np.corrcoef(baseDataMatrix)
    corrMatrix = np.abs(fill_ndarray(corrMatrix))
    return corrMatrix

def runModel(filterFunctionName, isDAE=1):
    packetNumsPerFlow=1000
    C=379
    
    initialAjacency=calSimilarity(C)
    X, y, flowSizeList = loadTrafficData(packetNumsPerFlow, C)
    
    n=X.shape[0]   

    index = [i for i in range(n)] 
    random.shuffle(index)
    X = X[index, :]
    y = y[index]

    n_train = n*1 // 2
    n_val = n*1 // 4
    X_train = X[:n_train]
    X_val   = X[n_train:n_train+n_val]
    X_test  = X[n_train+n_val:]
    
    y_train = y[:n_train]
    y_val   = y[n_train:n_train+n_val]
    y_test  = y[n_train+n_val:]


    params = dict()
    params['dir_name']       = 'demo'
    params['num_epochs']     = 10
    params['batch_size']     = 25
    params['eval_frequency'] = int(X_train.shape[0] / params['batch_size'])
    params['filter']         = filterFunctionName
    params['brelu']          = 'b1relu'
    params['pool']           = 'apool1'
    params['Cout_temporal']              = [16, 16, 16]
    params['Cout_spatial']              = [4, 4, 4]
    params['K_temporal']              = [4, 4, 4]
    params['K_spatial']              = [4, 4, 4]
    params['p']              = []
    params['M']              = [284, C]
    params['regularization'] = 5e-4
    params['dropout']        = 0.8
    params['learning_rate']  = 1e-3
    params['decay_rate']     = 0.7
    params['momentum']       = 0
    params['decay_steps']    = n_train / params['batch_size']
    params['originL']    = None
    params['scale_']    = 0
    params['mean_']    = 0
    params['AjacentMatrix'] = initialAjacency
    
    model = gcn_model_192.cgcnn(**params)
    loss, t_step = model.fit(X_train, y_train, X_val, y_val, types=30)
    
    fig, axes = plt.subplots(1,1,figsize=(10, 10))
    axes.plot(loss, 'g.-')
    axes.set_ylabel('training loss', color='g')
    string, loss, predictions = model.evaluate(X_test, y_test, types=30)
    
    rightNum=0
    for temp_idx in range(y_test.shape[0]):
        if np.argmax(predictions[temp_idx])==y_test[temp_idx]:
            rightNum=rightNum+1   
        
    print('测试分类准确率为:%.2f',rightNum/y_test.shape[0])
    plt.legend()
    plt.show()          
    
if __name__=='__main__':
    runModel('temporalSpatialBlock')
