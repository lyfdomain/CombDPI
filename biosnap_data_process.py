import copy
import csv
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np
from tqdm import trange
from Bio.Align import PairwiseAligner
from Bio.Align import substitution_matrices
# from Bio.SubsMat import MatrixInfo as matlist


jihe="dataset/biosnap/unseen drug"

train_data=[]
with open("./"+jihe+"/train.csv", newline='') as csvfile:
    # 使用csv模块读取CSV文件
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(reader)
    # 遍历CSV文件中的每一行数据
    for row in reader:
        # 处理每一行数据
        # print(', '.join(row))
        train_data.append(row)
train_data= np.array(train_data)

drug_train=[]
drug_train_smi=[]
for i, j in enumerate(train_data[:,3]):
    if j not in drug_train:
        drug_train.append(j)
        drug_train_smi.append(train_data[i,6])


protein_train=[]
protein_train_seq=[]
for ii, jj in enumerate(train_data[:,4]):
    if jj not in protein_train:
        protein_train.append(jj)
        protein_train_seq.append(train_data[ii,7])
        
print(len(drug_train), len(protein_train))

dti=np.zeros((len(drug_train),len(protein_train)))
for i in range(len(train_data)):
    if train_data[i,3] in drug_train and train_data[i,4] in protein_train:
        dti[drug_train.index(train_data[i,3]) ,protein_train.index(train_data[i,4])] = int(float(train_data[i,5]))
np.savetxt("./"+jihe+"/dti.txt", dti )


test_data=[]
with open("./"+jihe+"/test.csv", newline='') as csvfile:
    # 使用csv模块读取CSV文件
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(reader)
    # 遍历CSV文件中的每一行数据
    for row in reader:
        # 处理每一行数据
        # print(', '.join(row))
        test_data.append(row)
test_data= np.array(test_data)


# print(drug_data_p)
drug_test=[]
drug_test_smi=[]
for i, j in enumerate(test_data[:,3]):
    if j not in drug_test:
        drug_test.append(j)
        drug_test_smi.append(test_data[i,6])

proteint=[]
for j in test_data[:,4]:
    if j not in proteint:
        proteint.append(j)


protein_test=[]
protein_test_seq=[]
for ii, jj in enumerate(test_data[:,4]):
    if jj not in protein_test:
        protein_test.append(jj)
        protein_test_seq.append(test_data[ii,7])



set1 = drug_train_smi
set2 = drug_test_smi

# 将SMILES字符串转换为分子对象列表
mol_list1 = [Chem.MolFromSmiles(smiles) for smiles in set1]
mol_list2 = [Chem.MolFromSmiles(smiles) for smiles in set2]

# 计算分子指纹
fp_list1 = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in mol_list1]
fp_list2 = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in mol_list2]
# print(fp_list2)
# 计算相似性矩阵
similarity_matrix = np.zeros((len(fp_list1), len(fp_list2)))
for i in range(len(fp_list1)):
    for j in range(len(fp_list2)):
        similarity = DataStructs.DiceSimilarity(fp_list1[i], fp_list2[j])
        similarity_matrix[i, j] = similarity

print(similarity_matrix,similarity_matrix.shape)
np.savetxt("./"+jihe+"/SR.txt", similarity_matrix )




# 定义两个蛋白质集合
records1 = protein_train_seq
records2  = protein_test_seq
print(len(records1))


# 定义史密斯-沃特曼比对器
aligner = PairwiseAligner()
aligner.mode = 'global'
aligner.open_gap_score = -5
aligner.extend_gap_score = -1
aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")


# 计算比对得分
score_matrix = np.zeros((len(records1), len(records2)))
for i in trange(len(records1)):
    for j in range(len(records2)):
        char_to_remove = "U"
        records1[i] = records1[i].replace(char_to_remove, "")
        records2[j] = records2[j].replace(char_to_remove, "")
        alignments = aligner.align(str(records1[i]), str(records2[j])).score
        # score = max([alignment.score for alignment in alignments])[0]
        score_matrix[i, j] = alignments

print(score_matrix)
np.savetxt("./"+jihe+"/SP.txt",score_matrix)


