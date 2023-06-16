import csv
import pandas as pd
from rdchiral.template_extractor import extract_from_reaction
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def get_template(reaction):
    "get corresponding template for the reaction"
    reactants, product = reaction.split(">>")
    inputRec = {'_id': None, 'reactants': reactants, 'products': product}
    ans = extract_from_reaction(inputRec)
    if 'reaction_smarts' in ans.keys():
        return [product, ans["reaction_smarts"]]
    else:
        return None


def transform_reactions(reactions):
    """make sure that each reaction has only one product, if not, split it into more reactions"""
    reaction_oneproducts = []
    for reaction in reactions:
        reactants, products = reaction.split(">>")
        if "." in products:
            for product in products.split("."):
                reaction_oneproducts.append(reactants + ">>" + product)
        else:
            reaction_oneproducts.append(reaction)
    return reaction_oneproducts


def get_reaction(filename):
    """get reactions from the file"""
    data = pd.read_csv(filename)
    reactions = []
    for index, row in data.iterrows():
        reactions.append(row["reactants>reagents>production"])
    return reactions


def transform_to_fingerprint(product):
    "transform product into Morgan FingerPrintvector"
    mol = Chem.MolFromSmiles(product)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits(), dtype=np.bool_)
    arr[onbits] = 1
    return arr


def write_csv(path, row):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def read_csv(path):
    with open(path, "r") as f:
        reader = csv.reader(f)
        list = []
        for row in reader:
            list += row
    return list


if __name__ == '__main__':
    data_path = "raw_test.csv"
    reactions = get_reaction(data_path)
    reactions_oneproducts = transform_reactions(reactions)
    product_template = []
    for reaction in reactions_oneproducts:
        product_template_pair = get_template(reaction)
        product_template.append(product_template_pair)
    template_class= read_csv("all_templates.csv")
    template_class_dict = {template:num for num, template in enumerate(template_class)}
    onehot_templates = []
    fingerprints = []
    for product, template in product_template:
        onehot_template = list(np.zeros(shape=len(template_class), dtype=int))
        try:
            onehot_template[template_class_dict[template]] = 1
        except:
            pass
        onehot_templates.append(onehot_template)
        fingerprint = list(transform_to_fingerprint(product))
        fingerprints.append(fingerprint)
    np_fingerprints = np.array(fingerprints)
    np_onehot_template = np.array(onehot_templates)
    np.save("test_data_fingerprint.npy", fingerprints)
    np.save("test_data_onehot.npy", onehot_templates)
