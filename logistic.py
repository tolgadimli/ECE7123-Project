from sklearn.linear_model import LogisticRegression
from embedding_methods import *
from sklearn.preprocessing import StandardScaler
import pandas as pd



dataset_dictionaries1 = [get_elmo(), get_unirep(), get_transformer(), get_elmo_and_unirep_concat(),
                        get_elmo_and_transformer_concat(),   get_transformer_and_unirep_concat() , get_all_concat() ]

dataset_dictionaries2 = [get_elmo_and_unirep_padded_avg(), get_elmo_and_transformer_padded_avg(),
                             get_transformer_and_unirep_padded_avg(), get_all_padded_avg() ]

dataset_dictionaries3 = [get_elmo_and_unirep_inter_avg(), get_elmo_and_transformer_inter_avg(),
                             get_transformer_and_unirep_inter_avg(), get_all_inter_avg() ]
                             
padded_and_int_avgs = dataset_dictionaries2 + dataset_dictionaries3

#padded_and_int_avgs = [get_all_inter_avg()]

pandas_ls = []

for dataset_dict in padded_and_int_avgs:

    train_X, train_Y = dataset_dict['train']
    scaler = StandardScaler().fit(train_X)
    train_X = scaler.transform(train_X)

    val_X, val_Y = dataset_dict['valid']
    val_X = scaler.transform(val_X)

    fold_X, fold_Y = dataset_dict['test_fold']
    fold_X = scaler.transform(fold_X)

    family_X, family_Y = dataset_dict['test_family']
    family_X = scaler.transform(family_X)


    superfamily_X, superfamily_Y = dataset_dict['test_superfamily']
    superfamily_X = scaler.transform(superfamily_X)

    clf = LogisticRegression(random_state=0, max_iter=5000).fit(train_X, train_Y)

    train_acc = clf.score(train_X, train_Y)
    val_acc = clf.score(val_X, val_Y)
    fold_acc = clf.score(fold_X, fold_Y)
    family_acc = clf.score(family_X, family_Y)
    superfamily_acc = clf.score(superfamily_X, superfamily_Y)

    results = [train_acc, val_acc, fold_acc, family_acc, superfamily_acc] 

    pandas_ls.append(results)

    #df = pd.DataFrame(data = pandas_ls, columns = ['train', 'val', 'fold', 'family', 'superfamily'])
    #df.to_csv('logistic_results_padded_and_concat.csv')
    #df.to_csv('interpolated_avg_all.csv')
    break

