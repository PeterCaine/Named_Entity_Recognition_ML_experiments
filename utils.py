from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from nltk import PorterStemmer, WordNetLemmatizer
from collections import defaultdict
from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np
import gensim
import csv
import re
import os

######################## pre-processing ##############################


def file_cleaner(filepath):
    ''' takes filepath to CoNLL formatted file as input to read in file and
    remove: blank lines, "LRB" "RRB" - bracket notation and "-DOCSTART-" from
    the file contents
    Outputs a cleaned list of row data
    '''
    with open(filepath, 'r') as infile:
        line_list = infile.read().splitlines()

    cleaned_lines = []
    for line1 in line_list:
        if line1:
            templ_1 = line1.split('\t')
            if any(["DOCSTART" in templ_1[0],
                    templ_1[0] == '(',
                    templ_1[0] == ')',
                    templ_1[0] == '"',
                    "LRB" in templ_1[0],
                    "RRB" in templ_1[0]]
                   ):
                continue
            else:
                cleaned_lines.append(templ_1[:])
            continue

    return cleaned_lines


def label_aligner(cleaned_file):
    ''' takes a list of CoNLL formatted rows as input,
    returns a dataframe which aligns NER labels to one standard.
    '''

    column_names = ['token', 'pos', 'chunk', 'NER']
    map_dict = {'I-ORG': 'ORG', 'B-ORG': 'ORG', 'I-LOC': 'LOC', 'B-LOC': 'LOC',
                'B-MISC': 'MISC', 'I-MISC': 'MISC', 'I-PER': 'PER', 'O': 'O'}

    aligned_df = pd.DataFrame(cleaned_file, columns=column_names)
    aligned_df['NER'] = aligned_df.NER.map(map_dict)

    return aligned_df

######################## feature construction ##############################


def feature_maker(embed_file, dataframe, embed_signal='n'):
    '''takes a path to embeddings file, dataframe as input - default keyword
    embed-signal means that embeddings are not encoded by default
    returns an expanded dataframe with:
    a column of lemmatised words; a column of stemmed words; a column indicating
    capitalisation status; a column indicating capilatisation status of previous
    token; columns indicating shape, previous shape, short shape, previous
    short shape, following token short shape.
    If kwarg embed_signal is 'y', a list of embeddings is also generated.

    '''

    wnl = WordNetLemmatizer()
    prtr = PorterStemmer()
    stringed_list = [str(x) for x in dataframe['token']]
    wn_lemma_list = [wnl.lemmatize(t) for t in stringed_list]
    dataframe['lemma'] = wn_lemma_list
    prtr_stemmer_list = [prtr.stem(t) for t in stringed_list]
    dataframe['stem'] = prtr_stemmer_list

    dataframe['caps'] = 'no caps'
    dataframe.loc[dataframe['token'].str.contains('^[A-Z][a-z]'), ['caps']] = 'begin_cap'
    dataframe.loc[dataframe['token'].str.contains('[A-Z][A-Z]'), ['caps']] = 'all_caps'
    dataframe.loc[dataframe['token'].str.contains('[a-z][A-Z]]'), ['caps']] = 'caps_inside'

    temp_list = dataframe['caps'].to_list()
    temp_list.insert(0, 'no_cap')
    temp_list.pop()
    dataframe['prev_caps'] = temp_list

    dataframe['short_shape'] = 'x'
    dataframe.loc[dataframe['token'].str.contains('^[A-Z][a-z]'), ['short_shape']] = 'Xx'
    dataframe.loc[dataframe['token'].str.contains('[A-Z][A-Z]'), ['short_shape']] = 'XX'
    dataframe.loc[dataframe['token'].str.contains('[a-z][A-Z]]'), ['short_shape']] = 'xXx'
    dataframe.loc[dataframe['token'].str.contains('\W'), ['short_shape']] = '-'

    prev_short_shape_list = []
    prev_short_shape_list = dataframe['short_shape'].to_list()
    prev_short_shape_list.insert(0, '-')
    prev_short_shape_list.pop()
    dataframe['prev_short_shape'] = prev_short_shape_list

    next_short_shape_list = []
    next_short_shape_list = dataframe['short_shape'].to_list()
    next_short_shape_list.pop(0)
    next_short_shape_list.append('-')
    dataframe['next_short_shape'] = next_short_shape_list

    shape_list = []
    pre_list = []
    suf_list = []
    for text in dataframe['token']:

        prefix = text[:3]
        suffix = text[-3:]
        pre_list.append(prefix)
        suf_list.append(suffix)
        replace_caps = re.sub('[A-Z]', 'X', text)
        replace_lowers = re.sub('[a-z]', 'x', replace_caps)
        replace_digits = re.sub('\d', 'd', replace_lowers)

        shape_list.append(replace_digits)

    dataframe['shape'] = shape_list

    prev_shape_list = []
    prev_shape_list = dataframe['shape'].to_list()
    prev_shape_list.insert(0, '-')
    prev_shape_list.pop()
    dataframe['prev_shape'] = prev_shape_list

    dataframe['prefix'] = pre_list
    dataframe['suffix'] = suf_list

    if embed_signal == 'y':
        word_embedding_model = gensim.models.KeyedVectors.load_word2vec_format(
            embed_file, binary=True)
        embeddings = []
        for token in dataframe['token']:
            if token in word_embedding_model:
                vector = word_embedding_model[token]
            else:
                vector = [0]*300
            embeddings.append(vector)
        return dataframe, embeddings
    else:
        return dataframe


def dicts_n_labels(df_w_caps, features):
    '''
    takes a dataframe and a user-generated list of features as input
    returns a dictionary of features ready for vecotization, a list of NER labels
    and a list of token from the dataframe
    '''
    # deal with any NaN values
    values = 'NA'
    cleaned_df = df_w_caps.fillna(value=values)
    no_NER = cleaned_df[features]
    dict_for_vec = no_NER.to_dict('records')
    # create training labels
    labels = list(cleaned_df['NER'])
    # create list of tokens
    tokens = list(cleaned_df['token'])

    return dict_for_vec, labels, tokens


############################## NN Path only ###################################
def NER_to_array(labels):
    '''
    creates an array which maps labels to a dictionary of vectors
    returns labels as an array and the mapping dictionary for decoding

    '''
    label_set = set()
    for label in labels:
        label_set.add(label)
    label2Idx = {}
    for label in label_set:
        label2Idx[label] = len(label2Idx)
    map_prep = pd.DataFrame(labels)
    mapped = list(map_prep[0].map(label2Idx))
    new_array = np.asarray(mapped)
    return new_array, label2Idx


######################## vectorization ##############################


def dict_vectorizer(training_dict, gold_dict):
    '''
    a function that takes two dictionaries of CoNLL data (training and test)
    Returns vectors usable as input for machine learning calculations
    '''
    v = DictVectorizer()
    training_vec = v.fit_transform(training_dict)
    test_vec = v.transform(gold_dict)

    return training_vec, test_vec


def dict_vectorizer_embed(training_dict, gold_dict):
    '''
    a function that takes two dictionaries of CoNLL data (training and test)
    allows for embeddings
    Returns vectors usable as input for machine learning calculations
    '''
    v = DictVectorizer()
    training_vec = v.fit_transform(training_dict)
    test_vec = v.transform(gold_dict)
    test_array = test_vec.toarray()
    training_array = training_vec.toarray()

    return training_vec, test_vec, training_array, test_array

######################## concatenation ##############################


def concat_arrays(feature_array, embedding_list):
    '''
    a feature for embeddings path which concatenates an array of features and
    a list of feature vectors
    returns one list of concatenated feature vectors and associated embeddings

    '''
    num_words = feature_array.shape[0]
    concat_input = []  # for storing the result of concatenating
    for index in range(num_words):
        # concatenate features per word
        representation = list(feature_array[index]) + list(embedding_list[index])
        concat_input.append(representation)
    return concat_input


######################## predictions ##############################
def knn_predictions(training_vec, test_vec, training_labels, gold_tokens):
    '''
    a function that takes training X and y
    outputs a file of predicted labels for test data using the KNearest neighbors
    classifier algorithm
    '''
    neigh = KNeighborsClassifier(n_neighbors=5).fit(training_vec, training_labels)
    MNB_var = MultinomialNB().fit(training_vec, training_labels)
    predictions = list(neigh.predict(test_vec))
    predictions_list = []
    for i, j in zip(gold_tokens, predictions):
        predictions_list.append([str(i), str(j)])

    with open('predicted_knn.csv', 'w') as outfile:
        for line in predictions_list:
            element = '\t'.join(line)
            outfile.write(element+'\n')


def MNB_predictions(training_vec, test_vec, training_labels, gold_tokens):
    '''
    a function that takes training X and y
    outputs a file of predicted labels for test data using the Naive Bayes
    classifier algorithm
    '''
    MNB_var = MultinomialNB().fit(training_vec, training_labels)
    predictions = list(MNB_var.predict(test_vec))
    predictions_list = []
    for i, j in zip(gold_tokens, predictions):
        predictions_list.append([str(i), str(j)])

    with open('predicted_nb.csv', 'w') as outfile:
        for line in predictions_list:
            element = '\t'.join(line)
            outfile.write(element+'\n')


def logistic_predictions(training_vec, test_vec, training_labels, gold_tokens):
    '''
    a function that takes training X and y
    outputs a file of predicted labels for test data using the Logistic Regression
    classifier algorithm
    '''
    log_var = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000)
    log_var.fit(training_vec, training_labels)
    predictions = list(log_var.predict(test_vec))
    predictions_list = []
    for i, j in zip(gold_tokens, predictions):
        predictions_list.append([str(i), str(j)])

    with open('predicted_lr.csv', 'w') as outfile:
        for line in predictions_list:
            element = '\t'.join(line)
            outfile.write(element+'\n')


def SVM_predictions_e(training_vec, test_vec, training_labels, gold_tokens, concat_training, concat_test):
    '''
    a function that takes training X and y, concatenated training and test
    data for embeddings
    outputs a file of predicted labels for test data using a liner SVC
    classifier algorithm
    '''
    svm_var = LinearSVC(random_state=0, tol=1e-5)
    svm_var.fit(concat_training, training_labels)
    predictions = list(svm_var.predict(concat_test))
    predictions_list = []
    for i, j in zip(gold_tokens, predictions):
        predictions_list.append([str(i), str(j)])

    with open('predicted_SVM_embed.csv', 'w') as outfile:
        for line in predictions_list:
            element = '\t'.join(line)
            outfile.write(element+'\n')


def SVM_predictions(training_vec, test_vec, training_labels, gold_tokens):
    '''
    a function that takes training X and y
    outputs a file of predicted labels for test data using a liner SVC
    classifier algorithm
    '''
    svm_var = LinearSVC(random_state=0, tol=1e-5, max_iter=1500)
    svm_var.fit(training_vec, training_labels)
    predictions = list(svm_var.predict(test_vec))
    predictions_list = []
    for i, j in zip(gold_tokens, predictions):
        predictions_list.append([str(i), str(j)])

    with open('predicted_SVM.csv', 'w') as outfile:
        for line in predictions_list:
            element = '\t'.join(line)
            outfile.write(element+'\n')

######################## analysis ##############################


def table_maker(tally_dict_gold, tally_dict_file, confusion_dict, basename, features):
    '''
    takes the outputs of dict_to_dataframe as inputs to create an evaluation
    tables publishes full evaluation tables (including f-score, precision and
    recall) to latex file.
    '''
    # data in the form of list of tuples
    data = [tally_dict_gold, tally_dict_file, confusion_dict]

    df = pd.DataFrame(data, index=['Gold', 'System', 'True Positive'])
    df = df.T
    df['Precision'] = round(df['True Positive']/df['System'], 3)
    df['Recall'] = round(df['True Positive']/df['Gold'], 3)
    df['f-score'] = round(2*(df['Precision']*df['Recall'])/(df['Precision']+df['Recall']), 3)
    stripped_basename = os.path.basename(basename)
    print(basename, '\n', df.head())
    with open('evaluation_outcome.txt', 'a') as outfile:
        outfile.write(f'\n\nfeatures used ={str(features)}\n')
        for row in df.T:
            text = f'{stripped_basename} {row} precision {df.at[row,"Precision"]}; recall {df.at[row,"Recall"]}; f-score {df.at[row,"f-score"]} \n'
            outfile.write(text)

    with open('final_tables.tex', 'a') as purdy_tabs:
        purdy_tabs.write('\n'+stripped_basename+'\n'+str(features)+'\n')
        purdy_tabs.write(df.to_latex())


def dict_to_dataframe(features, file1, file2):
    '''
    takes NER tags from two CSV files and creates dictionaries to give statistics
    for tags as well as direct comparison between the files.
    the output dictionaries are used as inputs for function table_maker.
    '''
    confusion_dict = defaultdict(int)
    tally_dict_gold = defaultdict(int)
    tally_dict_file = defaultdict(int)

    with open(file1, 'r') as infile_g, open(file2) as infile_2:
        a_csv = csv.reader(infile_g)
        a_header = next(a_csv)
        b_csv = csv.reader(infile_2, delimiter='\t')

        a = []
        b = []

        for tag in a_csv:
            tally_dict_gold[tag[3]] += 1
            a.append(tag[3])
        for lineb in b_csv:
            b.append(lineb[1])
            tally_dict_file[lineb[1]] += 1
        for i, j in zip(a, b):
            if i == j:
                confusion_dict[i] += 1
    basename = file2
    table_maker(tally_dict_gold, tally_dict_file, confusion_dict, basename, features)
