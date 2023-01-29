import tempfile
import deepchem as dc
from deepchem.models import GCNModel
import pandas as pd
import numpy as np

# # option 1 - load the data and featurize
# loader = dc.data.CSVLoader(["task1"], feature_field="smiles",featurizer=dc.feat.MolGraphConvFeaturizer())
# dataset = loader.create_dataset('data_training.csv')

# load the data - option 2
data_pd = pd.read_csv('data_full.csv')
# select the training data
data_pd_training = data_pd[data_pd['assignment_DNN_taut']=='training']
# select the first 500 rows
data_pd_training = data_pd_training[0:500]
# extract the smiles and labels
smiles = data_pd_training['SMILES'].to_list()
labels = data_pd_training['logP'].to_numpy()
# featurize
featurizer = dc.feat.MolGraphConvFeaturizer()
X = featurizer.featurize(smiles)
# build the dataset
dataset = dc.data.NumpyDataset(X=X, y=labels)
# train model
model = GCNModel(mode='regression',graph_conv_layers=[64,128],dense_layer_size=256,dropout=0.1,
	n_tasks=1,batch_size=16,learning_rate=0.001)
loss = model.fit(dataset, nb_epoch=100)
print(loss)


# test the model

# select the validation data
data_pd_validation = data_pd[data_pd['assignment_DNN_taut']=='validation']
# select the first 500 rows
data_pd_validation = data_pd_validation[0:10]
# extract the smiles and labels
smiles = data_pd_validation['SMILES'].to_list()
labels = data_pd_validation['logP'].to_numpy()
# featurize
featurizer = dc.feat.MolGraphConvFeaturizer()
X = featurizer.featurize(smiles)
# build the dataset
dataset = dc.data.NumpyDataset(X=X, y=labels)
# run prediction
prediction = model.predict(dataset)
ground_truth = data_pd_validation['logP'].to_numpy()
ground_truth = ground_truth[:,np.newaxis]
print(np.hstack((ground_truth,prediction)))