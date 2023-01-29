#Copyright 2017 PandeLab

#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import tensorflow as tf
import numpy as np
import deepchem as dc
from deepchem.feat.mol_graphs import ConvMol
from deepchem.metrics import to_one_hot
from deepchem.models.losses import L2Loss
from deepchem.models.layers import GraphConv, GraphPool, Dense, Dropout, GraphGather, Stack
import csv
from tensorflow.keras.layers import Input, Reshape, Conv2D, Flatten, Dense, Softmax
from tensorflow.keras.layers import BatchNormalization

#Reads data and converts data to ConvMol objects

def read_data(input_file_path):
    featurizer = dc.feat.ConvMolFeaturizer(use_chirality=True)
    loader = dc.data.CSVLoader(tasks=prediction_tasks, smiles_field="SMILES", featurizer=featurizer)
    dataset = loader.featurize(input_file_path, shard_size=8192)
    return dataset

def data_generator(dataset,batch_size,epochs=1):
  for ind, (X_b, y_b, w_b, ids_b) in enumerate(dataset.iterbatches(batch_size, epochs,deterministic=True, pad_batches=True)):
    print(ind)
    # print(X_b)
    # print('----------')
    multiConvMol = ConvMol.agglomerate_mols(X_b)
    inputs = [multiConvMol.get_atom_features(), multiConvMol.deg_slice, np.array(multiConvMol.membership)]
    for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
      inputs.append(multiConvMol.get_deg_adjacency_lists()[i])
    labels = [y_b]
    weights = [w_b]
    yield (inputs, labels, weights)

#Used for loading of trained model

def layer_reference(model, layer):
  if isinstance(layer, list):
    return [model.layers[x.name] for x in layer]
  return model.layers[layer.name]

#Get array for the calculation of rms and r² for dataset
def reshape_y_pred(y_true, y_pred):
 
  n_samples = len(y_true)
  retval = np.vstack(y_pred)
  return retval[:n_samples]

#Define working directory

model_dir = '.'

#Define prediction task and directory of .csv file
#csv file should include SMILES and the prediction task

prediction_tasks = ['logP']
train_dataset = read_data('data.csv')
test_dataset = read_data('data.csv')

# https://github.com/deepchem/deepchem/blob/master/examples/tutorials/Introduction_to_Graph_Convolutions.ipynb
import tensorflow.keras.layers as layers

class MyGraphConvModel(tf.keras.Model):

  def __init__(self,batch_size):
    super(MyGraphConvModel, self).__init__()
    self.gc1 = GraphConv(64, activation_fn=tf.nn.tanh)
    self.batch_norm1 = layers.BatchNormalization()
    self.gp1 = GraphPool()

    self.gc2 = GraphConv(128, activation_fn=tf.nn.tanh)
    self.batch_norm2 = layers.BatchNormalization()
    self.gp2 = GraphPool()

    self.dense1 = layers.Dense(256, activation=tf.nn.tanh)
    self.batch_norm3 = layers.BatchNormalization()
    self.dropout = layers.Dropout(0.1)
    self.readout = GraphGather(batch_size=batch_size, activation_fn=tf.nn.tanh)

    self.regression = layers.Dense(1,activation=None)

  def call(self, inputs):
    gc1_output = self.gc1(inputs)
    batch_norm1_output = self.batch_norm1(gc1_output)
    gp1_output = self.gp1([batch_norm1_output] + inputs[1:])

    gc2_output = self.gc2([gp1_output] + inputs[1:])
    batch_norm2_output = self.batch_norm2(gc2_output)
    gp2_output = self.gp2([batch_norm2_output] + inputs[1:])

    dense1_output = self.dense1(gp2_output)
    batch_norm3_output = self.batch_norm3(dense1_output)
    dropout = self.dropout(batch_norm3_output)
    readout_output = self.readout([dropout] + inputs[1:])

    return self.regression(readout_output)


batch_size = 3
model = dc.models.KerasModel(MyGraphConvModel(batch_size=batch_size), loss=dc.models.losses.L2Loss(), model_dir='model')
model.fit_generator(data_generator(train_dataset,batch_size=batch_size,epochs=1000))
print('Training set score:', model.evaluate_generator(data_generator(train_dataset,batch_size=batch_size),[dc.metrics.mean_squared_error]))
# model.save()

print(model.predict_on_generator(data_generator(train_dataset,batch_size=batch_size)))