import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2

from main import n_fold_split, rf_max_depth
from main import under_sampling_c1, under_sampling_c2, under_sampling_c3
from main import over_sampling_c1, over_sampling_c2, over_sampling_c3


def RandomForest_on_entire_dataset():
    rf = RandomForestClassifier(criterion="entropy")
    score_array = []
    accuracy = []
    num_leaves = []
    num_nodes = []
    for i in range(1, n_fold_split):
        X_train = np.load(f"split/Xtr_fold_{i}.npy")
        X_test = np.load(f"split/Xte_fold_{i}.npy")
        y_train = np.load(f"split/ytr_fold_{i}.npy")
        y_test = np.load(f"split/yte_fold_{i}.npy")
        trained_rf = rf.fit(X_train, y_train)
        print(len(trained_rf.estimators_))
        num_leaves_test = 0
        num_nodes_test = 0
        for tree in trained_rf.estimators_:
            num_leaves_test += tree.get_n_leaves()
            num_nodes_test += tree.tree_.node_count
        print(num_leaves_test)
        num_leaves.append(num_leaves_test)
        num_nodes.append(num_nodes_test)

        y_pred = trained_rf.predict(X_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        score_array.append(precision_recall_fscore_support(y_test, y_pred, average=None))

    print("rf total data num leaves")
    print(np.mean(num_leaves, axis=0))

    print("rf total data num nodes")
    print(np.mean(num_nodes, axis=0))

    avg_accuracy = np.mean(accuracy, axis=0)
    print("Accuracy score for total RF")
    print(avg_accuracy)

    avg_score = np.mean(score_array, axis=0)
    print("Prec - Recall - F1 values for RF on entire DataSet")
    print(avg_score)

    # Save CSV file
    a = np.array(avg_score)
    np.savetxt('CSV Result/RF/rf_entire.csv', a, delimiter=';')


def RandomForest_on_entire_dataset_feature_selection():
    rf = RandomForestClassifier(criterion="entropy")
    score_array = []
    accuracy = []
    num_leaves = []
    num_nodes = []
    for i in range(1, n_fold_split):
        X_train = np.load(f"split/Xtr_fold_{i}.npy")
        X_test = np.load(f"split/Xte_fold_{i}.npy")
        y_train = np.load(f"split/ytr_fold_{i}.npy")
        y_test = np.load(f"split/yte_fold_{i}.npy")
        selection = SelectKBest(chi2, k=20).fit(X_train, y_train)
        X_new_train = selection.transform(X_train)
        trained_rf = rf.fit(X_new_train, y_train)
        X_new_test = selection.transform(X_test)
        y_pred = trained_rf.predict(X_new_test)

        num_leaves_test = 0
        num_nodes_test = 0
        for tree in trained_rf.estimators_:
            num_leaves_test += tree.get_n_leaves()
            num_nodes_test += tree.tree_.node_count
        print(num_leaves_test)
        num_leaves.append(num_leaves_test)
        num_nodes.append(num_nodes_test)

        accuracy.append(accuracy_score(y_test, y_pred))
        score_array.append(precision_recall_fscore_support(y_test, y_pred, average=None))

    print("rf total data num leaves")
    print(np.mean(num_leaves, axis=0))

    print("rf total data num nodes")
    print(np.mean(num_nodes, axis=0))

    avg_accuracy = np.mean(accuracy, axis=0)
    print("Accuracy score for total RF feature selection")
    print(avg_accuracy)

    avg_score = np.mean(score_array, axis=0)
    print("Prec - Recall - F1 values for RF on entire DataSet feature selection")
    print(avg_score)

    # Save CSV file
    a = np.array(avg_score)
    np.savetxt('CSV Result/RF/rf_entire_feature.csv', a, delimiter=';')


def RandomForest_on_undersampled_dataset():
    ros = RandomUnderSampler(sampling_strategy={0: under_sampling_c1, 1: under_sampling_c2, 2: under_sampling_c3}, random_state=50, replacement=True)
    rf = RandomForestClassifier(criterion="entropy")
    pipeline = make_pipeline(ros, rf)

    score_array = []
    accuracy = []
    num_leaves = []
    num_nodes = []
    for i in range(1, n_fold_split):
        X_train = np.load(f"split/Xtr_fold_{i}.npy")
        X_test = np.load(f"split/Xte_fold_{i}.npy")
        y_train = np.load(f"split/ytr_fold_{i}.npy")
        y_test = np.load(f"split/yte_fold_{i}.npy")
        y_pred = pipeline.fit(X_train, y_train).predict(X_test)

        num_leaves_test = 0
        num_nodes_test = 0
        for tree in rf.estimators_:
            num_leaves_test += tree.get_n_leaves()
            num_nodes_test += tree.tree_.node_count
        print(num_leaves_test)
        num_leaves.append(num_leaves_test)
        num_nodes.append(num_nodes_test)


        accuracy.append(accuracy_score(y_test, y_pred))
        score_array.append(precision_recall_fscore_support(y_test, y_pred, average=None))

    print("rf total data num leaves")
    print(np.mean(num_leaves, axis=0))

    print("rf total data num nodes")
    print(np.mean(num_nodes, axis=0))

    avg_accuracy = np.mean(accuracy, axis=0)
    print("Accuracy score for undersampled RF")
    print(avg_accuracy)

    avg_score = np.mean(score_array, axis=0)
    print("Prec - Recall - F1 values for undersampled RF")
    print(avg_score)

    # Save CSV file
    a = np.array(avg_score)
    np.savetxt('CSV Result/RF/rf_under.csv', a, delimiter=';')

def RandomForest_on_undersampled_dataset_feature_selection():
    ros = RandomUnderSampler(sampling_strategy={0: under_sampling_c1, 1: under_sampling_c2, 2: under_sampling_c3}, random_state=50, replacement=True)
    rf = RandomForestClassifier(criterion="entropy")
    pipeline = make_pipeline(ros, rf)

    score_array = []
    accuracy = []
    num_leaves = []
    num_nodes = []
    for i in range(1, n_fold_split):
        X_train = np.load(f"split/Xtr_fold_{i}.npy")
        X_test = np.load(f"split/Xte_fold_{i}.npy")
        y_train = np.load(f"split/ytr_fold_{i}.npy")
        y_test = np.load(f"split/yte_fold_{i}.npy")
        selection = SelectKBest(chi2, k=20).fit(X_train, y_train)
        X_new_train = selection.transform(X_train)
        X_new_test = selection.transform(X_test)
        y_pred = pipeline.fit(X_new_train, y_train).predict(X_new_test)

        num_leaves_test = 0
        num_nodes_test = 0
        for tree in rf.estimators_:
            num_leaves_test += tree.get_n_leaves()
            num_nodes_test += tree.tree_.node_count
        print(num_leaves_test)
        num_leaves.append(num_leaves_test)
        num_nodes.append(num_nodes_test)

        accuracy.append(accuracy_score(y_test, y_pred))
        score_array.append(precision_recall_fscore_support(y_test, y_pred, average=None))

    print("rf total data num leaves")
    print(np.mean(num_leaves, axis=0))

    print("rf total data num nodes")
    print(np.mean(num_nodes, axis=0))

    avg_accuracy = np.mean(accuracy, axis=0)
    print("Accuracy score for undersampled RF feature selection")
    print(avg_accuracy)

    avg_score = np.mean(score_array, axis=0)
    print("Prec - Recall - F1 values for undersampled RF feature selection")
    print(avg_score)

    # Save CSV file
    a = np.array(avg_score)
    np.savetxt('CSV Result/RF/rf_under_feature.csv', a, delimiter=';')

def RandomForest_on_oversampled_dataset():
    ros = RandomOverSampler(sampling_strategy={0:over_sampling_c1, 1:over_sampling_c2, 2:over_sampling_c3}, random_state=10)
    rf = RandomForestClassifier(criterion="entropy")
    pipeline = make_pipeline(ros, rf)

    score_array = []
    accuracy = []
    num_leaves = []
    num_nodes = []
    for i in range(1, n_fold_split):
        X_train = np.load(f"split/Xtr_fold_{i}.npy")
        X_test = np.load(f"split/Xte_fold_{i}.npy")
        y_train = np.load(f"split/ytr_fold_{i}.npy")
        y_test = np.load(f"split/yte_fold_{i}.npy")
        y_pred = pipeline.fit(X_train, y_train).predict(X_test)

        num_leaves_test = 0
        num_nodes_test = 0
        for tree in rf.estimators_:
            num_leaves_test += tree.get_n_leaves()
            num_nodes_test += tree.tree_.node_count
        print(num_leaves_test)
        num_leaves.append(num_leaves_test)
        num_nodes.append(num_nodes_test)

        accuracy.append(accuracy_score(y_test, y_pred))
        score_array.append(precision_recall_fscore_support(y_test, y_pred, average=None))

    print("rf total data num leaves")
    print(np.mean(num_leaves, axis=0))

    print("rf total data num nodes")
    print(np.mean(num_nodes, axis=0))
    avg_accuracy = np.mean(accuracy, axis=0)
    print("Accuracy score for oversampled RF")
    print(avg_accuracy)

    avg_score = np.mean(score_array, axis=0)
    print("Prec - Recall - F1 values for oversampled RF")
    print(avg_score)

    # Save CSV file
    a = np.array(avg_score)
    np.savetxt('CSV Result/RF/rf_over.csv', a, delimiter=';')

def RandomForest_on_oversampled_dataset_feature_selection():
    ros = RandomOverSampler(sampling_strategy={0:over_sampling_c1, 1:over_sampling_c2, 2:over_sampling_c3}, random_state=10)
    rf = RandomForestClassifier(criterion="entropy")
    pipeline = make_pipeline(ros, rf)

    score_array = []
    accuracy = []
    num_leaves = []
    num_nodes = []

    for i in range(1, n_fold_split):
        X_train = np.load(f"split/Xtr_fold_{i}.npy")
        X_test = np.load(f"split/Xte_fold_{i}.npy")
        y_train = np.load(f"split/ytr_fold_{i}.npy")
        y_test = np.load(f"split/yte_fold_{i}.npy")
        selection = SelectKBest(chi2, k=20).fit(X_train, y_train)
        X_new_train = selection.transform(X_train)
        X_new_test = selection.transform(X_test)
        y_pred = pipeline.fit(X_new_train, y_train).predict(X_new_test)

        num_leaves_test = 0
        num_nodes_test = 0
        for tree in rf.estimators_:
            num_leaves_test += tree.get_n_leaves()
            num_nodes_test += tree.tree_.node_count
        print(num_leaves_test)
        num_leaves.append(num_leaves_test)
        num_nodes.append(num_nodes_test)

        accuracy.append(accuracy_score(y_test, y_pred))
        score_array.append(precision_recall_fscore_support(y_test, y_pred, average=None))

    print("rf total data num leaves")
    print(np.mean(num_leaves, axis=0))

    print("rf total data num nodes")
    print(np.mean(num_nodes, axis=0))
    avg_accuracy = np.mean(accuracy, axis=0)
    print("Accuracy score for oversampled RF feature selection")
    print(avg_accuracy)

    avg_score = np.mean(score_array, axis=0)
    print("Prec - Recall - F1 values for oversampled RF feature selection")
    print(avg_score)

    # Save CSV file
    a = np.array(avg_score)
    np.savetxt('CSV Result/RF/rf_over_feature.csv', a, delimiter=';')

def randomForest_with_SMOTENN():
    smot = SMOTEENN(sampling_strategy="auto", random_state=10, n_jobs=4)
    rf = RandomForestClassifier(criterion="entropy")
#    pipeline = make_pipeline(smot, clf_tree, verbose=True)
    score_array = []
    accuracy = []
    num_leaves = []
    num_nodes = []
    for i in range(1, n_fold_split):
        X_train = np.load(f"split/Xtr_fold_{i}.npy")
        X_test = np.load(f"split/Xte_fold_{i}.npy")
        y_train = np.load(f"split/ytr_fold_{i}.npy")
        y_test = np.load(f"split/yte_fold_{i}.npy")
        X_train_smtk, y_train_smtk = smot.fit_resample(X_train, y_train)
#        y_pred = pipeline.fit(X_train, y_train).predict(X_test)
        trained_rf = rf.fit(X_train_smtk, y_train_smtk)
        y_pred = trained_rf.predict(X_test)

        num_leaves_test = 0
        num_nodes_test = 0
        for tree in trained_rf.estimators_:
            num_leaves_test += tree.get_n_leaves()
            num_nodes_test += tree.tree_.node_count
        print(num_leaves_test)
        num_leaves.append(num_leaves_test)
        num_nodes.append(num_nodes_test)

        accuracy.append(accuracy_score(y_test, y_pred))
        score_array.append(precision_recall_fscore_support(y_test, y_pred, average=None))

    print("rf total data num leaves")
    print(np.mean(num_leaves, axis=0))

    print("rf total data num nodes")
    print(np.mean(num_nodes, axis=0))
    avg_accuracy = np.mean(accuracy, axis=0)
    print(avg_accuracy)
    print("Accuracy score for SMOTE RF")

    avg_score = np.mean(score_array, axis=0)
    print("Prec - Recall - F1 values for SMOTEEN RF")
    print(avg_score)

    # Save CSV file
    a = np.array(avg_score)
    np.savetxt('CSV Result/RF/rf_SMOTE.csv', a, delimiter=';')