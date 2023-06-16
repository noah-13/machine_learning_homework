import pandas as pd
import torch
from tqdm import tqdm
from Task1.data_preparation import transform_to_fingerprint
from collections import OrderedDict
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from rdchiral.main import rdchiralRunText

class RolloutPolicyNet(nn.Module):
    def __init__(self, n_rules, idx2rule, fp_dim=2048, dim=512,
                 dropout_rate=0.4):
        super(RolloutPolicyNet, self).__init__()
        self.fp_dim = fp_dim
        self.n_rules = n_rules
        self.dropout_rate = dropout_rate
        self.fc1 = nn.Linear(fp_dim,dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(dim,n_rules)
        self.idx2rules = idx2rule


    def forward(self,x, y=None, loss_fn =nn.CrossEntropyLoss()):
        x = self.dropout1(F.elu(self.bn1(self.fc1(x))))
        x = self.fc3(x)
        if y is not None :
            return loss_fn(x, y)
        else :
            return x
        return x
    def run(self, mol, topk=1):
        arr = transform_to_fingerprint(mol)
        arr = np.reshape(arr, [-1, arr.shape[0]])
        arr = torch.tensor(arr, dtype=torch.float32)
        preds = self(arr)
        preds = F.softmax(preds, dim=1)
        probs, idx = torch.topk(preds, k=topk)
        rule_k = [self.idx2rules[id] for id in idx[0].numpy().tolist()]
        reactants = []
        for i, rule in enumerate(rule_k):
            out1 = []
            try:
                out1 = rdchiralRunText(rule, mol)
                # out1 = rdchiralRunText(rule, Chem.MolToSmiles(Chem.MolFromSmarts(x)))
                if len(out1) == 0:
                    continue
                # if len(out1) > 1: print("more than two reactants."),print(out1)
                out1 = sorted(out1)
                for reactant in out1:
                    reactants.append(reactant)
            # out1 = rdchiralRunText(x, rule)
            except ValueError:
                pass
        if len(reactants) == 0:
            return None
        if "." in reactants[0]:
            reactants = reactants[0].split(".")
        return reactants


def dfs(x, s, max_itr,rollout):
    itr = 0
    stack = rollout.run(x)
    while stack:
        node = stack.pop()
        itr += 1
        if node in s:
            continue
        children = rollout.run(node)
        if children is None:
            return False, itr
        if itr > max_itr:
            return False ,itr
        stack.extend(children)
    return True, itr

starting_mol = pd.read_pickle("starting_mols.pkl")
target_mol_route = pd.read_pickle("target_mol_route.pkl")
test_mols = pd.read_pickle("test_mols.pkl")

template_rule_path = r"template_rules_1.dat"
template_rules = {}
with open(template_rule_path, 'r') as f:
    for i, l in tqdm(enumerate(f), desc='rollout'):
        rule = l.strip()
        template_rules[rule] = i

idx2rule = {}
for rule, idx in template_rules.items():
    idx2rule[idx] = rule
rollout = RolloutPolicyNet(len(template_rules),idx2rule)

model_path = r"saved_rollout_state_1_2048.ckpt"
checkpoint = torch.load(model_path,map_location='cpu')
new_state_dict = OrderedDict()
for k, v in checkpoint.items():
    name = k[7:]
    new_state_dict[name] = v
rollout.load_state_dict(new_state_dict)
rollout.eval()
results = []
itrs = []
for i,test_mol in enumerate(test_mols):
    result ,itr = dfs(test_mol,starting_mol,max_itr=100,rollout=rollout)
    print("{}/190".format(i+1)+"success:{}".format(len([True for i in results if i == True]))+"aver_itr:{}".format(np.average(itrs)))
    results.append(result)
    itrs.append(itr)
print(np.average(itrs))
print(len([True for i in results if i == True]))