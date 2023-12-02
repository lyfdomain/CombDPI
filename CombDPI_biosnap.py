import numpy as np
import csv
from tqdm import trange
import sklearn
import sklearn.metrics
import math

jihe = "./dataset/biosnap/unseen drug"

sr_t = np.loadtxt(jihe+"/SR_test.txt")
sr_v = np.loadtxt(jihe+"/SR_val.txt")
sp_t = np.loadtxt(jihe+"/SP_test.txt")
sp_v = np.loadtxt(jihe+"/SP_val.txt")
dti=np.loadtxt(jihe+"/dti.txt")


print("=============")


def nor(sim_mat):
    mm = np.zeros(sim_mat.shape)
    for i in trange(sim_mat.shape[0]):
        for j in range(sim_mat.shape[1]):
            if sim_mat[i,j]>0:
                mm[i, j] = (sim_mat[i, j]) / max(sim_mat[i, :])#(len(pp[j]))
            else:
                mm[i, j]=0
    return mm


def sim_recon(sim_drug, sim_protein):
    sorted_drug = (sim_drug).argsort(axis=1).argsort(axis=1)
    sorted_drug=(len(sim_drug[0])-1)*np.ones(sim_drug.shape)-sorted_drug
    sorted_drug[sorted_drug == 0] = 1
    sorted_drug = 1/(sorted_drug)

    sorted_pro = (sim_protein).argsort(axis=0).argsort(axis=0)
    sorted_pro=(len(sim_protein)-1)*np.ones(sim_protein.shape)-sorted_pro
    sorted_pro[sorted_pro == 0] = 1
    sorted_pro = 1/(sorted_pro)
    return sorted_drug, sorted_pro


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


def get_dpiinformation(fpath):
    data = []
    with open(fpath, newline='') as csvfile:
        # 使用csv模块读取CSV文件
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)
        # 遍历CSV文件中的每一行数据
        for row in reader:
            # 处理每一行数据
            # print(', '.join(row))
            data.append(row)
    data = np.array(data)

    drug = []
    drug_smi = []
    for i, j in enumerate(data[:, 3]):
        if j not in drug:
            drug.append(j)
            drug_smi.append(data[i, 6])

    protein = []
    protein_seq = []
    for ii, jj in enumerate(data[:, 4]):
        if jj not in protein:
            protein.append(jj)
            protein_seq.append(data[ii, 7])
    return data, drug, drug_smi, protein, protein_seq

val_data, drug_val, drug_val_smi, protein_val, protein_val_seq \
    = get_dpiinformation("./"+jihe+"/val.csv")

test_data, drug_test, drug_test_smi, protein_test, protein_test_seq \
    = get_dpiinformation("./"+jihe+"/test.csv")


print("+++++++++++++")

sp_v= nor(sp_v)
sp_t= nor(sp_t)


sorted_drug_t, sorted_pro_t = sim_recon(sr_t, sp_t)
sorted_drug_v, sorted_pro_v = sim_recon(sr_v, sp_v)


sr_v_nor = drug_rowsimm(sr_v*sorted_drug_v)
sr_t_nor = drug_rowsimm(sr_t*sorted_drug_t)

sp_v_nor = dis_colsimm(sp_v*sorted_pro_v)
sp_t_nor = dis_colsimm(sp_t*sorted_pro_t)


sr_v=drug_rowsimm(sr_v)
sr_t=drug_rowsimm(sr_t)


sp_v=dis_colsimm(sp_v)
sp_t=dis_colsimm(sp_t)

print("+++++++++++++")

score_1 = []
score_2 = []
score_true = []

mat_real=dti
mat_1 = sr_v@dti@sp_v
mat_2 = sr_v_nor@dti@sp_v_nor

lam=10000

idx = []
for i in trange(len(val_data)):
    if val_data[i,3] in drug_val and val_data[i,4] in protein_val:
        idx.append([drug_val.index(val_data[i,3]), protein_val.index(val_data[i,4])])
# aa=np.array(idx)

for i in range(len(idx)):
    score_1.append(mat_1[idx[0][0], idx[0][1]])
    score_2.append(mat_2[idx[0][0], idx[0][1]])
    score_true.append(int(float(val_data[i,5])))
print(len(score_1), len(idx))

matrix_X = np.mat([score_1, score_2]).T

alpha = (np.ones((1, len(matrix_X.T))) @ (2 * matrix_X.T @ matrix_X + 2
                                          * lam * np.eye(len(matrix_X.T))).I @ (2 * matrix_X.T @ np.mat(score_true).T) - 1) / (np.ones((1, len(matrix_X.T))) @ (2 * matrix_X.T @ matrix_X + 2 * lam * np.eye(len(matrix_X.T))).I @ (np.ones((len(matrix_X.T), 1))))
weight = (2 * matrix_X.T @ matrix_X + 2 * lam * np.eye(len(matrix_X.T))).I @ (
        2 * matrix_X.T @ np.mat(score_true).T + - alpha[0, 0] * (np.ones((len(matrix_X.T), 1))))

a1 = float(weight[0])
a2 = float(weight[1])

dtii=a1*sr_t@dti@sp_t+a2*sr_t_nor@dti@sp_t_nor

aa=[]
for i in trange(len(test_data)):
    if test_data[i,3] in drug_test and test_data[i,4] in protein_test:
        aa.append([dtii[drug_test.index(test_data[i,3]), protein_test.index(test_data[i,4])], int(float(test_data[i,5]))])
aa=np.array(aa)

fpr, tpr, thresholds = sklearn.metrics.roc_curve(aa[:,1], aa[:,0])
print(aa[:,0], aa[:,1])
area = sklearn.metrics.auc(fpr, tpr)
print("the Area Under the ROCCurve is:", area)

precision, recall, _ = sklearn.metrics.precision_recall_curve(aa[:,1], aa[:,0])
area = sklearn.metrics.auc(recall, precision)
print("the Area Under the PRCurve is:", area)
