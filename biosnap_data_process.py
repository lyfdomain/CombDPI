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


def get_drug_sim(set1,set2):

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

    print(similarity_matrix, similarity_matrix.shape)
    return similarity_matrix

def get_protein_sim(records1,records2):

    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -1
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")

    # 计算比对得分
    print(len(records1), len(records2))
    score_matrix = np.zeros((len(records1), len(records2)))
    for i in trange(len(records1)):
        for j in range(len(records2)):
            char_to_remove = "U"
            records1[i] = records1[i].replace(char_to_remove, "")
            records2[j] = records2[j].replace(char_to_remove, "")
            alignments = aligner.align(str(records1[i]), str(records2[j])).score
            # score = max([alignment.score for alignment in alignments])[0]
            score_matrix[i, j] = alignments

    return score_matrix


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


jihe="dataset/biosnap/unseen drug"


train_data, drug_train, drug_train_smi, protein_train, protein_train_seq \
    = get_dpiinformation("./"+jihe+"/train.csv")

test_data, drug_test, drug_test_smi, protein_test, protein_test_seq \
    = get_dpiinformation("./"+jihe+"/test.csv")

val_data, drug_val, drug_val_smi, protein_val, protein_val_seq \
    = get_dpiinformation("./"+jihe+"/val.csv")


dti=np.zeros((len(drug_train),len(protein_train)))
for i in range(len(train_data)):
    if train_data[i,3] in drug_train and train_data[i,4] in protein_train:
        dti[drug_train.index(train_data[i,3]) ,protein_train.index(train_data[i,4])] = int(float(train_data[i,5]))
np.savetxt("./"+jihe+"/dti.txt", dti )



drug_simmat_test= get_drug_sim(drug_test_smi,drug_train_smi)
# print(drug_similarity_matrix, drug_similarity_matrix.shape)
np.savetxt("./"+jihe+"/SR_test.txt", drug_simmat_test)

drug_simmat_val= get_drug_sim(drug_val_smi,drug_train_smi)
# print(drug_similarity_matrix, drug_similarity_matrix.shape)
np.savetxt("./"+jihe+"/SR_val.txt", drug_simmat_val)

protein_simmat_test=get_protein_sim(protein_train_seq, protein_test_seq)
np.savetxt("./"+jihe+"/SP_test.txt", protein_simmat_test)

protein_simmat_val=get_protein_sim(protein_train_seq, protein_val_seq)
np.savetxt("./"+jihe+"/SP_val.txt", protein_simmat_val)
