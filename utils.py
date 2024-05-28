import os
import random
import numpy as np
import torch
import dgl
import logging

CHARPROTSET = {
    "A": 1,
    "C": 2,
    "B": 3,
    "E": 4,
    "D": 5,
    "G": 6,
    "F": 7,
    "I": 8,
    "H": 9,
    "K": 10,
    "M": 11,
    "L": 12,
    "O": 13,
    "N": 14,
    "Q": 15,
    "P": 16,
    "S": 17,
    "R": 18,
    "U": 19,
    "T": 20,
    "W": 21,
    "V": 22,
    "Y": 23,
    "X": 24,
    "Z": 25,
}

CHARPROTLEN = 25


def set_seed(seed=1000):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def graph_collate_func(x):
    d, t, p, y = zip(*x)
    d = dgl.batch(d)
    return d, t, torch.tensor(np.array(p)), torch.tensor(y)


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)


def integer_label_protein(sequence, max_length=1200):
    """
    Integer encoding for protein string sequence.
    Args:
        sequence (str): Protein string sequence.
        max_length: Maximum encoding length of input protein string.
    """
    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            letter = letter.upper()
            encoding[idx] = CHARPROTSET[letter]
        except KeyError:
            logging.warning(
                f"character {letter} does not exists in sequence category encoding, skip and treat as " f"padding."
            )
    return encoding




import rdkit
from rdkit import Chem
import rdkit.Chem.GraphDescriptors as gd

def mol_with_atom_index(mol:rdkit.Chem.rdchem.Mol)->rdkit.Chem.rdchem.Mol:
    """
    画出带原子序号的分子图
    """
    atoms = mol.GetNumAtoms()
    for idx in range(atoms):
        mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(mol.GetAtomWithIdx(idx).GetIdx()))
    return mol
# # Draw molecule with atom index (see RDKitCB_0)
# def mol_with_atom_index(mol):
#     for atom in mol.GetAtoms():
#         atom.SetAtomMapNum(atom.GetIdx())
#     return mol
# mol_with_atom_index(mol)


def phi(mol):
    """
    计算分子的柔性指数，
    Kier L B. An index of molecular flexibility from kappa shape attributes [J]. 
    Quantitative Structure‐Activity Relationships, 1989, 8(3): 221-4.
    """
    A = mol.GetNumHeavyAtoms()
    kappa1 = gd.Kappa1(mol)
    kappa2 = gd.Kappa2(mol)
    phi = kappa1*kappa2/A
    
    return phi


# 判断键是否可旋转 -> bool
## 这里我们定义可旋转键：
## Rotatable bonds were defined as any single bond, 
## not in a ring, bound to a heavy (i.e., non-hydrogen) atom. 
## Excluded from the count were amide C-N bonds because of their high rotational energy barrier.
def is_single_bond(bond:rdkit.Chem.rdchem.Bond)->bool:
    """
    判断键是否是单键
    """
    flag = bond.GetBondType() == Chem.BondType.SINGLE
    return flag


def is_in_ring(bond:rdkit.Chem.rdchem.Bond)->bool:
    """
    判断键是否在环内
    """
    flag = bond.IsInRing()
    return flag


def is_ch_bond(bond:rdkit.Chem.rdchem.Bond)->bool:
    """
    判断是否是碳-氢键
    """
    # 获取成键原子对象
    begin_atom = bond.GetBeginAtom()
    end_atom = bond.GetEndAtom()
    flag = False
    # 获取成键原子的原子编号（元素）
    begin_atom_num = begin_atom.GetAtomicNum()
    end_atom_num = end_atom.GetAtomicNum()
    # 判断键的两端原子是否为碳（C）和氮（N）
    if {begin_atom_num, end_atom_num} == {6, 1}:
        flag = True
        
    return flag


def is_amide_bond(mol:rdkit.Chem.rdchem.Mol, bond:rdkit.Chem.rdchem.Bond)->bool:
    """
    判断键是否为酰胺键
    """
    # 获取成键原子对象
    begin_atom = bond.GetBeginAtom()
    end_atom = bond.GetEndAtom()
    flag = False
    # 获取成键原子的原子编号（元素）
    begin_atom_num = begin_atom.GetAtomicNum()
    end_atom_num = end_atom.GetAtomicNum()
    # 判断键的两端原子是否为碳（C）和氮（N）
    if {begin_atom_num, end_atom_num} == {6, 7}:
        if begin_atom_num == 6:
            catom = begin_atom
        else:
            catom = end_atom
        neighbors = catom.GetNeighbors()
        for neighbor in neighbors:
            if neighbor.GetAtomicNum() == 8 and mol.GetBondBetweenAtoms(catom.GetIdx(), neighbor.GetIdx()).GetBondType() == Chem.BondType.DOUBLE:
                flag = True
                break
    return flag
    
    
def is_rotable(mol:rdkit.Chem.rdchem.Mol, bond:rdkit.Chem.rdchem.Bond)->bool:
    """
    判断键是否是可旋转键
    """
    # 判断是否是单键
    single_flag = is_single_bond(bond)
    # 判断是否在环内
    ring_flag = is_in_ring(bond)
    # 判断是否是碳-氢键
    ch_flag = is_ch_bond(bond)
    # 判断是否为酰胺键
    amide_flag = is_amide_bond(mol, bond)
    
    rotable_flag = single_flag and not ring_flag and not amide_flag and not ch_flag
    
    return rotable_flag


########
class UnionFind:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

    
def merge_sets(mol:rdkit.Chem.rdchem.Mol):
    atoms = mol.GetAtoms()
    atom_sets = []
    for atom in atoms:
        atom_set = [atom.GetIdx()]
        neighbors = atom.GetNeighbors()
        for neighbor in neighbors:
            bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            if not is_rotable(mol, bond):
                atom_set.append(neighbor.GetIdx())
        atom_sets.append(atom_set)
    
    element_to_index = {}  # 映射元素到集合索引的字典
    next_index = 0

    # 初始化并查集
    uf = UnionFind(len(atom_sets))

    # 构建映射和合并集合
    for s in atom_sets:
        for elem in s:
            if elem in element_to_index:
                uf.union(element_to_index[elem], next_index)
            else:
                element_to_index[elem] = next_index
        next_index += 1

    # 构建新的集合
    merged_sets = {}
    for elem, index in element_to_index.items():
        root_index = uf.find(index)
        if root_index not in merged_sets:
            merged_sets[root_index] = []
        merged_sets[root_index].append(elem)

    return list(merged_sets.values())


def transform_from_rotable_bonds(smiles:str, explicit_hydrogens=False):
    """
    根据键是否可旋转聚合原子，求聚合矩阵
    """
    mol = Chem.MolFromSmiles(smiles)
    if explicit_hydrogens:
        mol = Chem.AddHs(mol)
    merged_sets = merge_sets(mol)
    transform_rotable = torch.zeros([len(merged_sets), mol.GetNumAtoms()], dtype=torch.float32)
    for index, i in enumerate(merged_sets):
        for j in i:
            transform_rotable[index,j] = 1.0

    return transform_rotable


# def graph_collate_func(x):
#     d, smiles, y = zip(*x)
#     d = dgl.batch(d)
#     smiles = dgl.batch(smiles)
#     return d, smiles, y