# Named_Entity_Recognition_ML_experiments

## Description
An exploration of machine learning approaches to Named Entity Recognition (NER) including classical classifiers as well as Neural Network (NN) and embedding techniques

This was a portfolio project for a course on Machine Learning for NLP. 


Neural net consists of a feed forward model with:
1 Dense layer- relu, 1 Dense output - softmax, 1 hidden layer Dense - relu
batch size = 50
epochs = 6
optimizer = 'adam'
loss = categorical_crossentropy


## Run
Run from commandline: 

python .\main.py 

3 arguments can be specified (but need not be): argv[1] argv[2] argv[3]

use arguments to pass in path to:
- training data; 
- test data; 
- embeddings (word2vec GoogleNews-vectors-negative300)

defaults assume everything is stored in local folder. E.g.:
- train_file = 'reuters-train-tab.en'
- test_file = 'gold_stripped.conll'
- embed_file = './GoogleNews-vectors-negative300.bin'

These can also be set inside the petercaine_program.py file in the first section.

## Alternative paths

3 possibilites are available:
- use embeddings (y/n)
- use neural net (y/n)
- select algorithm (initials given at prompt)
press Enter to indicate selection is finished

3 options are available 2 with embeddings; 1 without.

(1) tokens and features as data - using machine learning algorithms
- recommended features are (copy/paste into input):
token pos chunk stem caps prev_caps short_shape prev_short_shape next_short_shape shape prev_shape

(2) embeddings and features as data - using svm
- recommended features are (copy/paste into input):
pos chunk caps prev_caps short_shape prev_short_shape next_short_shape shape prev_shape

(3) embeddings and features as a neural net.
- Features are (no user choice involved):
pos chunk caps prev_caps short_shape prev_short_shape next_short_shape shape prev_shape

