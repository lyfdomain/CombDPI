import random
import numpy as np
from tqdm import trange


fold=10
dr_pre=np.loadtxt('./dataset/drugbank/DPI_drugbank.txt')
simdr=np.loadtxt('./dataset/drugbank/SR.txt')
simpre=np.loadtxt('./dataset/drugbank/SP.txt')

prenum=len(simpre)

dd_dim1=dr_pre.flatten()

i=0
list_1=[]
while i<len(dd_dim1):
    if dd_dim1[i]==1:
        list_1.append(i)
    i=i+1
num1=len(list_1)
group_size1=int(num1/fold)
random.seed(10)
random.shuffle(list_1)
#
array_1=np.array(list_1)[0:fold*group_size1]
group_data1=np.reshape(array_1,(fold,group_size1))
np.savetxt('./dataset/drugbank/corssval/index_1.txt',group_data1)


i=0
list_0=[]
while i<len(dd_dim1):
    if dd_dim1[i]==0:
        list_0.append(i)
    i=i+1
num0=len(list_0)
group_size0=int(num0/fold)
random.seed(10)
random.shuffle(list_0)

array_0=np.array(list_0)[0:fold*group_size0]
group_data0=np.reshape(array_0,(fold,group_size0))
np.savetxt('./dataset/drugbank/corssval/index_0.txt',group_data0)


f = 0
for f in trange(fold):
    DTI = np.copy(dr_pre)
    i=0
    while i < group_size1:
        r = int(group_data1[f, i] / prenum)
        c = int(group_data1[f, i] % prenum)
        DTI[r, c] = 0
        i += 1  # 得到每次交叉验证中所使用的A矩阵
    np.savetxt('./dataset/drugbank/corssval/DTI'+str(f)+".txt",DTI)



