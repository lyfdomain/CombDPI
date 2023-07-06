import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sklearn.metrics
from sklearn.metrics import *

prenum=1996

class IKNN:
    def __init__(self, k) -> None:
        self._k = k

    def fit(self, _X):
        self.X = _X.copy()
        return self

    def neighbors(self, i) -> np.ndarray:
        drugs_sim = self.X[i]
        values = np.sort(drugs_sim)[::-1][1:self._k + 1]
        indexes = np.argsort(drugs_sim)[::-1][1:self._k + 1]
        return (indexes, values)


def screening(K, drug_mat, target_mat, eta):
    m = len(drug_mat)
    n = len(target_mat)
    sr = np.zeros((m, m))
    sp = np.zeros((n, n))

    iknn = IKNN(K)

    iknn.fit(drug_mat)
    for d in range(m):
        (indexes, values) = iknn.neighbors(d)
        z = np.sum(values)
        if z == 0:
            continue
        for i in range(K):
            sr[d, indexes[i]] = (eta ** i) * values[i]

    iknn.fit(target_mat)
    for t in range(n):
        (indexes, values) = iknn.neighbors(t)
        z = np.sum(values)
        if z == 0:
            continue
        for i in range(K):
            sp[indexes[i], t] = (eta ** i) * values[i]
    return sr, sp

def drug_rowsimm(dd):
    mm = np.zeros(dd.shape)
    for i in range(dd.shape[0]):
        # for j in range(mm.shape[1]):
            if np.sum(dd[i, :])!=0:
               mm[i, :] = dd[i, :] / np.sum(dd[i, :])
    return mm


def dis_colsimm(dd):
    mm = np.zeros(dd.shape)
    for j in range(dd.shape[1]):
            if np.sum(dd[:, j]) != 0:
                mm[:, j] = dd[:, j] / np.sum(dd[:, j])
    return mm


index_1 = np.loadtxt("./dataset/drugbank/crossval/index_1.txt")
index_0 = np.loadtxt("./dataset/drugbank/crossval/index_0.txt")
index = np.hstack((index_1, index_0))
A_real=np.loadtxt("./dataset/drugbank/DPI_drugbank.txt")  #载入药物疾病数据

SR = np.loadtxt("./dataset/drugbank/SR.txt")
SP = np.loadtxt("./dataset/drugbank/SP.txt")

auc=[]
ap=[]
jieguo=[]

KN=20
eta=0.7
lam=10000


for f in range(10):

    A = np.loadtxt("./dataset/drugbank/crossval/DTI"+str(f)+".txt")

    dr_simdti  = cosine_similarity(A,A)
    pre_simdti = cosine_similarity(A.T, A.T)


    SR_1,SP_1=screening(KN, SR, SP, 0.7)
    SR_2,SP_2=screening(KN, dr_simdti, pre_simdti, 0.7)
    print(SR_1.shape,SP_1.shape)

    SR_1=drug_rowsimm(SR_1)
    SP_1=dis_colsimm(SP_1)

    SR_2=drug_rowsimm(SR_2)
    SP_2=dis_colsimm(SP_2)

    SR=drug_rowsimm(SR)
    SP=dis_colsimm(SP)

    dr_simdti=drug_rowsimm(dr_simdti )
    pre_simdti=dis_colsimm(pre_simdti)

    ee=[]
    ff=[]

    score_1=[]
    score_2=[]
    score_3=[]
    score_4=[]
    score_5=[]
    score_6=[]
    score_7=[]
    score_8=[]
    score_true=[]

    mat_1=np.matmul(A,SP)
    mat_2=np.matmul(SR,A)
    mat_3=np.matmul(A,pre_simdti)
    mat_4=np.matmul(dr_simdti,A)
    mat_5=np.matmul(A,SP_1)
    mat_6=np.matmul(A,SP_2)
    mat_7=np.matmul(SR_1,A)
    mat_8=np.matmul(SR_2,A)
    mat_real=A

    idx0 = []
    idx1 = []

    idx=[i1 for i2 in index_1[0:f] for i1 in i2]+[i1 for i2 in index_1[f+1:] for i1 in i2]
    print(len(idx))

    for i in range(len(idx)):
        d=int(idx[i]/prenum)
        p=int(idx[i]%prenum)
        idx1.append([d,p])

    idx=[i1 for i2 in index_0[f+1:] for i1 in i2]+[i1 for i2 in index_0[0:f] for i1 in i2]
    for i in range(len(idx1)):
        d=int(idx[i]/prenum)
        p=int(idx[i]%prenum)
        idx0.append([d,p])

    for i in range(len(idx0)):
        score_1.append(mat_1[idx0[i][0],idx0[i][1]])
        score_1.append(mat_1[idx1[i][0],idx0[i][1]])
        score_2.append(mat_2[idx0[i][0],idx0[i][1]])
        score_2.append(mat_2[idx1[i][0],idx0[i][1]])
        score_3.append(mat_3[idx0[i][0],idx0[i][1]])
        score_3.append(mat_3[idx1[i][0],idx0[i][1]])
        score_4.append(mat_4[idx0[i][0],idx0[i][1]])
        score_4.append(mat_4[idx1[i][0],idx0[i][1]])
        score_5.append(mat_5[idx0[i][0],idx0[i][1]])
        score_5.append(mat_5[idx1[i][0],idx0[i][1]])
        score_6.append(mat_6[idx0[i][0],idx0[i][1]])
        score_6.append(mat_6[idx1[i][0],idx0[i][1]])
        score_7.append(mat_7[idx0[i][0],idx0[i][1]])
        score_7.append(mat_7[idx1[i][0],idx0[i][1]])
        score_8.append(mat_8[idx0[i][0],idx0[i][1]])
        score_8.append(mat_8[idx1[i][0],idx0[i][1]])
        score_true.append(mat_real[idx0[i][0],idx0[i][1]])
        score_true.append(mat_real[idx1[i][0],idx0[i][1]])
    print(len(score_1),len(idx1),mat_1[idx0[0][0],idx0[0][1]])

    matrix_X=np.mat([score_1,score_2,score_3,score_4,score_5,score_6,score_7,score_8]).T
    print(matrix_X.shape,np.mat(score_true).shape)

    # print((matrix_X.T @ matrix_X), np.ones((1,len(matrix_X.T))))
    alpha=(np.ones((1,len(matrix_X.T))) @ (2*matrix_X.T @ matrix_X + 2*lam*np.eye(len(matrix_X.T))).I @ (2*matrix_X.T @ np.mat(score_true).T ) -1)/(np.ones((1,len(matrix_X.T))) @ (2*matrix_X.T @ matrix_X + 2*lam*np.eye(len(matrix_X.T))).I @ (np.ones((len(matrix_X.T),1))))
    weight = (2*matrix_X.T @ matrix_X + 2*lam*np.eye(len(matrix_X.T)) ).I @ (2*matrix_X.T @ np.mat(score_true).T + - alpha[0,0] * (np.ones((len(matrix_X.T),1))))


    a1=float(weight[0])
    a2=float(weight[1])
    a3=float(weight[2])
    a4=float(weight[3])
    a5=float(weight[4])
    a6=float(weight[5])
    a7=float(weight[6])
    a8=float(weight[7])

    A_pred= a1*np.matmul(A,SP)+a2*np.matmul(SR,A)+a3*np.matmul(A,pre_simdti)+a4*np.matmul(dr_simdti,A)+a5*np.matmul(A,SP_1)+a6*np.matmul(A,SP_2)\
            +a7*np.matmul(SR_1,A)+a8*np.matmul(SR_2,A)


    # idx=index[f,:1404]
    idx=index[f]

    for i in range(len(idx)):
        d=int(idx[i]/prenum)
        p=int(idx[i]%prenum)
        # realvalue[d,p]=A_real[d,p]
        ee.append(A_real[d,p])
    #
    # RR=np.zeros(R.shape)
    for i in range(len(idx)):
        d=int(idx[i]/prenum)
        p=int(idx[i]%prenum)
        # RR[d,p]=A_pred[d,p]
        ff.append(A_pred[d,p])

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(ee, ff)
    area = sklearn.metrics.auc(fpr, tpr)
    print("the Area Under the PRCurve is:", area)
    auc.append(area)

    aps = average_precision_score(ee, ff)
    print("the AP score is:", aps)
    ap.append(aps)

print("the average Area Under the PRCurve is:", sum(auc) / 10, "\n", "the average Area Under the PRCurve is:", sum(ap) / 10)
