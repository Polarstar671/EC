import numpy as np
import scipy.sparse

def convertParentsToMapMatrixs(graphs,parents,perms,xAverage):

    mapLists=[]
    
    addList=[]
    for i in range(graphs[len(graphs)-1].shape[0]):
        addList.append(i) 
    parents.append(addList)
    mapLists.append(parents[0])
    for i in range(1,len(parents)):
        tempMap=parents[0].copy()
        for j in range(1,i+1):
            for k in range(0,len(parents[0])):
                tempMap[k]=parents[j][tempMap[k]] 
        mapLists.append(tempMap)
        
    mapArrays=[]
    for temp_idx in range(len(mapLists)):
        temp_list=mapLists[temp_idx]

        curArray=np.array([[0 for i in range(142)] for j in range(len(parents[temp_idx]))]) 
        for t_idx in range(len(temp_list)):
            curArray[temp_list[t_idx]][t_idx]=np.float32(1)
        curArray=perm_mapMatrix(curArray, perms[temp_idx]) 
        mapArrays.append(curArray)
       

    remapArrays=[]

    for temp_idx in range(len(mapLists)):
        temp_list=mapLists[temp_idx]

        curArray=np.array([[0 for i in range(142)] for j in range(len(parents[temp_idx]))]) 
        for t_idx in range(len(temp_list)):
            curArray[temp_list[t_idx]][t_idx]=np.float32(1)
        
        curRow,curCol=curArray.shape
        for rIdx in range(curRow):
            rowSum=0
            for cIdx in range(curCol):
                if curArray[rIdx][cIdx]==1:
                    rowSum+=xAverage[cIdx]
                    
            if rowSum!=0:
                for cIdx in range(curCol):
                    curArray[rIdx][cIdx]=xAverage[cIdx]/rowSum

        curArray=perm_mapMatrix(curArray, perms[temp_idx]) 
        remapArrays.append(np.transpose(curArray))
    return mapArrays,remapArrays            
 
def convertListToMatrix(originArray):

     M,K=originArray.shape
     rowList=[]
     colList=[]
     valList=[]
     for i in range(M):
         for j in range(K):
             if originArray[i][j]==np.float32(1):
                 rowList.append(i)
                 colList.append(j)
                 valList.append(np.float32(1))
     targetMatrix=scipy.sparse.csr_matrix((valList,(rowList,colList)), shape=(M,K))

     return targetMatrix
 
def coarsen(A, levels, self_connections=False):

    graphs, parents = metis(A, levels)

    perms = compute_perm(parents)

    for i, A in enumerate(graphs):
        M, M = A.shape

        if not self_connections:
            A = A.tocoo()
            A.setdiag(0)

        if i < levels:
            A = perm_adjacency(A, perms[i])

        A = A.tocsr()
        A.eliminate_zeros()
        graphs[i] = A

        Mnew, Mnew = A.shape

        print('Layer {0}: M_{0} = |V| = {1} nodes ({2} added),'
              '|E| = {3} edges'.format(i, Mnew, Mnew-M, A.nnz//2))
    return graphs, perms if levels > 0 else None, parents


def metis(W, levels, rid=None):
    N, N = W.shape
    if rid is None:
        rid = np.random.permutation(range(N))

    parents = []
    degree = W.sum(axis=0) - W.diagonal()
    graphs = []
    graphs.append(W)

    for _ in range(levels):


        weights = degree
        weights = np.array(weights).squeeze()


        idx_row, idx_col, val = scipy.sparse.find(W)

        perm = np.argsort(idx_row)
        rr = idx_row[perm]
        cc = idx_col[perm]
        vv = val[perm]
        cluster_id = metis_one_level(rr,cc,vv,rid,weights)

        parents.append(cluster_id)


        nrr = cluster_id[rr]
        ncc = cluster_id[cc]
        nvv = vv
        Nnew = cluster_id.max() + 1

        W = scipy.sparse.csr_matrix((nvv,(nrr,ncc)), shape=(Nnew,Nnew))
        W.eliminate_zeros()

        graphs.append(W)
        N, N = W.shape


        degree = W.sum(axis=0)

        ss = np.array(W.sum(axis=0)).squeeze()
        rid = np.argsort(ss)

    return graphs, parents



def metis_one_level(rr,cc,vv,rid,weights):

    nnz = rr.shape[0]
    N = rr[nnz-1] + 1

    marked = np.zeros(N, np.bool)
    rowstart = np.zeros(N, np.int32)
    rowlength = np.zeros(N, np.int32)
    cluster_id = np.zeros(N, np.int32)

    oldval = rr[0]
    count = 0
    clustercount = 0

    for ii in range(nnz):
        rowlength[count] = rowlength[count] + 1
        if rr[ii] > oldval:
            oldval = rr[ii]
            rowstart[count+1] = ii
            count = count + 1

    for ii in range(N):
        tid = rid[ii]
        if not marked[tid]:
            wmax = 0.0
            rs = rowstart[tid]
            marked[tid] = True
            bestneighbor = -1
            for jj in range(rowlength[tid]):
                nid = cc[rs+jj]
                if marked[nid]:
                    tval = 0.0
                else:
                    tval = vv[rs+jj] * (1.0/weights[tid] + 1.0/weights[nid])
                if tval > wmax:
                    wmax = tval
                    bestneighbor = nid

            cluster_id[tid] = clustercount

            if bestneighbor > -1:
                cluster_id[bestneighbor] = clustercount
                marked[bestneighbor] = True

            clustercount += 1

    return cluster_id

def compute_perm(parents):

    indices = []
    if len(parents) > 0:
        M_last = max(parents[-1]) + 1
        indices.append(list(range(M_last)))

    for parent in parents[::-1]:

        pool_singeltons = len(parent)

        indices_layer = []
        for i in indices[-1]:
            indices_node = list(np.where(parent == i)[0])
            assert 0 <= len(indices_node) <= 2

            if len(indices_node) is 1:
                indices_node.append(pool_singeltons)
                pool_singeltons += 1

            elif len(indices_node) is 0:
                indices_node.append(pool_singeltons+0)
                indices_node.append(pool_singeltons+1)
                pool_singeltons += 2


            indices_layer.extend(indices_node)
        indices.append(indices_layer)


    for i,indices_layer in enumerate(indices):
        M = M_last*2**i

        assert len(indices[0] == M)

        assert sorted(indices_layer) == list(range(M))

    return indices[::-1]

assert (compute_perm([np.array([4,1,1,2,2,3,0,0,3]),np.array([2,1,0,1,0])])
        == [[3,4,0,9,1,2,5,8,6,7,10,11],[2,4,1,3,0,5],[0,1,2]])

def perm_data(x, indices):

    if indices is None:
        return x

    N, M, K = x.shape
    Mnew = len(indices)
    assert Mnew >= M
    xnew = np.empty((N, Mnew, K))
    for i,j in enumerate(indices):

        if j < M:
            xnew[:,i,:] = x[:,j,:]

        else:
            xnew[:,i,:] = np.array([[0 for i in range(6)] for j in range(N)])
    return xnew

def perm_mapMatrix(x, indices):

    if indices is None:
        return x

    M, K = x.shape
    Mnew = len(indices)
    print('Mnew:',Mnew)
    print('M:',M)
    assert Mnew >= M
    xnew = np.empty((Mnew, K))
    for i,j in enumerate(indices):
        if j < M:
            xnew[i,:] = x[j,:]
        else:
            xnew[i,:] = np.array([[0 for i in range(K)]])
    return xnew

def perm_adjacency(A, indices):

    if indices is None:
        return A

    M, M = A.shape
    Mnew = len(indices)
    assert Mnew >= M
    A = A.tocoo()


    if Mnew > M:
        rows = scipy.sparse.coo_matrix((Mnew-M,    M), dtype=np.float32)
        cols = scipy.sparse.coo_matrix((Mnew, Mnew-M), dtype=np.float32)
        A = scipy.sparse.vstack([A, rows])
        A = scipy.sparse.hstack([A, cols])


    perm = np.argsort(indices)
    A.row = np.array(perm)[A.row]
    A.col = np.array(perm)[A.col]


    assert type(A) is scipy.sparse.coo.coo_matrix
    return A
