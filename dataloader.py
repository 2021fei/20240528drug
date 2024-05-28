import pandas as pd
import torch.utils.data as data
import torch
import numpy as np
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from utils import integer_label_protein
from utils import is_rotable, transform_from_rotable_bonds, mol_with_atom_index, phi

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import rdkit.Chem.GraphDescriptors as gd


class DTIDataset(data.Dataset):
    def __init__(self, list_IDs, df):
        self.list_IDs = list_IDs
        self.df = df

        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=False)
        self.fc = partial(smiles_to_bigraph, explicit_hydrogens=True, add_self_loop=False)
        self.atom_rotable_transform = partial(transform_from_rotable_bonds, explicit_hydrogens=True)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        mol_smiles = self.df.iloc[index]['SMILES']
        v_d = self.fc(smiles=mol_smiles, 
                      node_featurizer=self.atom_featurizer, 
                      edge_featurizer=self.bond_featurizer)

        coords = self.df.iloc[index]['xyz']
        v_d.ndata['pos'] = coords
        
        # 计算rotable-transform矩阵
        t = self.atom_rotable_transform(mol_smiles)

        v_p = self.df.iloc[index]['Protein']
        v_p = integer_label_protein(v_p)
        y = self.df.iloc[index]["Y"]
        # y = torch.Tensor([y])
        return v_d, t, v_p, y


class MultiDataLoader(object):
    def __init__(self, dataloaders, n_batches):
        if n_batches <= 0:
            raise ValueError("n_batches should be > 0")
        self._dataloaders = dataloaders
        self._n_batches = np.maximum(1, n_batches)
        self._init_iterators()

    def _init_iterators(self):
        self._iterators = [iter(dl) for dl in self._dataloaders]

    def _get_nexts(self):
        def _get_next_dl_batch(di, dl):
            try:
                batch = next(dl)
            except StopIteration:
                new_dl = iter(self._dataloaders[di])
                self._iterators[di] = new_dl
                batch = next(new_dl)
            return batch

        return [_get_next_dl_batch(di, dl) for di, dl in enumerate(self._iterators)]

    def __iter__(self):
        for _ in range(self._n_batches):
            yield self._get_nexts()
        self._init_iterators()

    def __len__(self):
        return self._n_batches
