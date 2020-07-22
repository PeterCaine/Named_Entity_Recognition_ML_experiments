import sys
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import pandas as pd
from utils import file_cleaner, label_aligner, feature_maker, \
    dicts_n_labels, dict_vectorizer, MNB_predictions, logistic_predictions, \
    SVM_predictions, SVM_predictions_e, table_maker, dict_to_dataframe, \
    concat_arrays, dict_vectorizer_embed, knn_predictions, NER_to_array\

#########################################################################

# input the key variables for the program
train_file = 'reuters-train-tab.en'
test_file = 'gold_stripped.conll'
embed_file = 'GoogleNews-vectors-negative300.bin'

if len(sys.argv) == 1:
    train_file = train_file
    test_file = test_file
    embed_file = embed_file

elif len(sys.argv) == 2:
    train_file = sys.argv[1]
    test_file = test_file
    embed_file = embed_file
elif len(sys.argv) == 3:
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    embed_file = embed_file
else:
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    embed_file = sys.argv[3]

# user selects whether to use embeddings + whether to use NN (default is to use neither)
select_embed = input("Type y + Enter if you want to use embeddings instead of tokens; \
 hit Enter to continue. ")
# set default value for neural net option (default -no)
NN_Y_N = 'n'

# user can select features - see READ ME for recommended lists of features to copy and paste in. Default is none.
if select_embed == 'y':
    NN_Y_N = input('Type y + Enter if you wish to use a Neural Net:\
    Hit enter to continue: \
    ')
    if NN_Y_N == 'y':
        user_features = 'pos chunk caps prev_caps short_shape prev_short_shape next_short_shape shape prev_shape'
    else:
        user_features = input('''Available features - (pos, chunk, caps, prev_caps, short_shape, prev_short_shape, next_short_shape, shape, prev_shape, prefix, suffix).
Select the features you wish to use by typing them exactly as written, separated by a space. E.g. pos chunk caps ...
See ReadMe for suggestions to copy/paste - default = none
Press enter to confirm selection:
    ''')
else:
    user_features = input("""Available features - (token, pos, chunk, lemma, stem, caps, prev_caps, short_shape, prev_short_shape, next_short_shape, shape, prev_shape, prefix, suffix).
Select the features you wish to use by typing them exactly as written, separated by a space. E.g. token lemma caps...
See ReadMe for suggestions to copy/paste - default = none
Press enter to confirm selection:
    """)


features = user_features.lower().split()

if NN_Y_N == 'y':
    pass
elif select_embed == 'y':
    model = 'SVM_embed'
else:
    model = input("Which algorithm would you prefer to use? \
    (Type: svm for SVM; lr for Logistic Regression; nb for Naive Bayes; or knn for KNearest Neighbors)\
     ")

#########################Pre-processing#####################################

# this preprocesses the training and test_dataset
clean_train = file_cleaner(train_file)
clean_test = file_cleaner(test_file)

# in this step we align the labels
aligned_train_df = label_aligner(clean_train)
aligned_test_df = label_aligner(clean_test)

# an intermediate step to store a cleaned version of test data
aligned_test_df.to_csv('cleaned_gold.csv', index=False)

###############################Features################################

# feature maker adds information about capitalisation
# previous token capitalisation; shape and short shape etc.
if select_embed == 'y':
    train_features, train_embeddings = feature_maker(embed_file, aligned_train_df, select_embed)
    test_features, test_embeddings = feature_maker(embed_file, aligned_test_df, select_embed)
else:
    train_features = feature_maker(embed_file, aligned_train_df, select_embed)
    test_features = feature_maker(embed_file, aligned_test_df, select_embed)

###############################Vectorisation################################


# we prepare the dicts and lists for vectorization
gold_dict, gold_labels, gold_tokens = dicts_n_labels(test_features, features)
training_dict, training_labels, training_tokens = dicts_n_labels(train_features, features)


# path without using Neural Net
if NN_Y_N != 'y':

    # training_vec, test_vec, training_array, test_array = dict_vectorizer(training_dict, gold_dict)
    # vectorize training data
    if select_embed == 'y':
        training_vec, test_vec, training_array, test_array = dict_vectorizer_embed(
            training_dict, gold_dict)
        concat_training = concat_arrays(training_array, train_embeddings)
        concat_test = concat_arrays(test_array, test_embeddings)
    else:
        training_vec, test_vec = dict_vectorizer(training_dict, gold_dict)
# alternative path for various user selected options:
#embeddings; algorithms
    if select_embed == 'y':
        SVM_predictions_e(training_vec, test_vec, training_labels,
                          gold_tokens, concat_training, concat_test)
    elif model == 'svm':
        SVM_predictions(training_vec, test_vec, training_labels,
                        gold_tokens)
    elif model == 'nb':
        MNB_predictions(training_vec, test_vec, training_labels, gold_tokens)

    elif model == 'knn':
        knn_predictions(training_vec, test_vec, training_labels, gold_tokens)

    else:
        logistic_predictions(training_vec, test_vec, training_labels, gold_tokens)

##########################################################################
# output prediction file and perform analysis

    cleaned_gold = 'cleaned_gold.csv'
    predictions_file = f'predicted_{model}.csv'
    # predictions_file = output_file
    print('\n features are: \n', features, '\n')
    dict_to_dataframe(features, cleaned_gold, predictions_file)

######################### NN alternative path###########################
# alternative path using Neural net
else:
    training_array, label2Idx = NER_to_array(training_labels)
    test_for_df, label2Idx = NER_to_array(gold_labels)
    y_train = keras.utils.np_utils.to_categorical(training_array)
    y_test = keras.utils.np_utils.to_categorical(test_for_df)
    v = DictVectorizer()
    training_vec = v.fit_transform(training_dict)
    test_vec = v.transform(gold_dict)
    test_array = test_vec.toarray()
    training_array = training_vec.toarray()
    training_concat = concat_arrays(training_array, train_embeddings)
    test_concat = concat_arrays(test_array, test_embeddings)
    train_y = y_train
    test_y = y_test
    train_X = np.array(training_concat)
    test_X = np.array(test_concat)
# NN = feed forward w/ 1 hidden layer
    model = Sequential()
    model.add(Dense(units=64, input_dim=test_vec.shape[1]+300, activation='relu'))
    model.add(Dense(units=32, input_dim=20, activation='relu'))
    model.add(Dense(units=5, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_X, train_y, epochs=6, batch_size=50)
    prediction = model.predict(test_X)
    rounded_prediction = model.predict_classes(test_X)
    system_predictions = pd.DataFrame(rounded_prediction, columns=['Predicted'])
# perform prediction comparison with gold.
    gold_series = pd.DataFrame(test_for_df, columns=['Gold'])
    system_predictions = pd.DataFrame(rounded_prediction, columns=['Predicted'])
    result = pd.concat([gold_series, system_predictions], axis=1, sort=False)
    inv_map = {v: k for k, v in label2Idx.items()}
    result['Gold'] = result.Gold.map(inv_map)
    predictions = result.Predicted.map(inv_map)
    predictions_list = []
    for i, j in zip(gold_tokens, predictions):
        predictions_list.append([str(i), str(j)])
# publish results
    with open('predicted_NN.csv', 'w') as outfile:
        for line in predictions_list:
            element = '\t'.join(line)
            outfile.write(element+'\n')

    cleaned_gold = 'cleaned_gold.csv'
    predictions_file = 'predicted_NN.csv'
    features = 'NN + pos chunk caps prev_caps short_shape prev_short_shape next_short_shape shape prev_shape'

    dict_to_dataframe(features, cleaned_gold, predictions_file)
