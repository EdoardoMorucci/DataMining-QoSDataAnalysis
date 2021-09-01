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

from main import n_fold_split, bdt_max_depth
from main import over_sampling_c1, over_sampling_c2, over_sampling_c3
from main import under_sampling_c1, under_sampling_c2, under_sampling_c3

def tree_on_entire_dataset():
    clf_tree = DecisionTreeClassifier(criterion="entropy", max_depth=bdt_max_depth)
    accuracy = []
    score_array = []
    sumLeaves = []
    sumNodes = []
    for i in range(1, n_fold_split):
        X_train = np.load(f"split/Xtr_fold_{i}.npy")
        X_test = np.load(f"split/Xte_fold_{i}.npy")
        y_train = np.load(f"split/ytr_fold_{i}.npy")
        y_test = np.load(f"split/yte_fold_{i}.npy")
        clf = clf_tree.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        score_array.append(precision_recall_fscore_support(y_test, y_pred, average=None))
        sumLeaves.append(clf.get_n_leaves())
        sumNodes.append(clf.tree_.node_count)

    print("Average leaves")
    print(np.mean(sumLeaves, axis=0))

    print("Average nodes")
    print(np.mean(sumNodes, axis=0))

    avg_accuracy = np.mean(accuracy, axis=0)
    print("Accuracy score for BDT")
    print(avg_accuracy)

    avg_score = np.mean(score_array, axis=0)
    print("Prec - Recall - F1 values for BDT")
    print(avg_score)

    # Save CSV file
    a = np.array(avg_score)
    np.savetxt("CSV Result/BDT/tree_entire.csv", a, delimiter=';')

def tree_on_entire_dataset_feature_selection():
    clf_tree = DecisionTreeClassifier(criterion="entropy", max_depth=bdt_max_depth)
    accuracy = []
    score_array = []
    sumLeaves = []
    sumNodes = []
    for i in range(1, n_fold_split):
        X_train = np.load(f"split/Xtr_fold_{i}.npy")
        X_test = np.load(f"split/Xte_fold_{i}.npy")
        y_train = np.load(f"split/ytr_fold_{i}.npy")
        y_test = np.load(f"split/yte_fold_{i}.npy")

        selection = SelectKBest(chi2, k=20).fit(X_train, y_train)
        X_new_train = selection.transform(X_train)
        clf = clf_tree.fit(X_new_train, y_train)
        X_new_test = selection.transform(X_test)
        y_pred = clf.predict(X_new_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        score_array.append(precision_recall_fscore_support(y_test, y_pred, average=None))
        sumLeaves.append(clf.get_n_leaves())
        sumNodes.append(clf.tree_.node_count)

    print("Average leaves")
    print(np.mean(sumLeaves, axis=0))

    print("Average nodes")
    print(np.mean(sumNodes, axis=0))

    avg_accuracy = np.mean(accuracy, axis=0)
    print("Accuracy score for BDT with feature selection")
    print(avg_accuracy)

    avg_score = np.mean(score_array, axis=0)
    print("Prec - Recall - F1 values for normal tree with feature selection")
    print(avg_score)

    # Save CSV file
    a = np.array(avg_score)
    np.savetxt("CSV Result/BDT/tree_entire_feature.csv", a, delimiter=';')


def tree_with_undersampling():
    rus = RandomUnderSampler(sampling_strategy={0: under_sampling_c1, 1: under_sampling_c2, 2: under_sampling_c3}, random_state=50, replacement=True)
    clf_tree = DecisionTreeClassifier(criterion="entropy", max_depth=bdt_max_depth)
    pipeline = make_pipeline(rus, clf_tree)

    score_array = []
    accuracy = []
    sumLeaves = []
    sumNodes = []
    for i in range(1, n_fold_split):
        X_train = np.load(f"split/Xtr_fold_{i}.npy")
        X_test = np.load(f"split/Xte_fold_{i}.npy")
        y_train = np.load(f"split/ytr_fold_{i}.npy")
        y_test = np.load(f"split/yte_fold_{i}.npy")
        y_pred = pipeline.fit(X_train, y_train).predict(X_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        score_array.append(precision_recall_fscore_support(y_test, y_pred, average=None))
        sumLeaves.append(clf_tree.get_n_leaves())
        sumNodes.append(clf_tree.tree_.node_count)

    print("Average leaves")
    print(np.mean(sumLeaves, axis=0))

    print("Average nodes")
    print(np.mean(sumNodes, axis=0))

    avg_score = np.mean(score_array, axis=0)
    avg_accuracy = np.mean(accuracy, axis=0)
    print("Accuracy score for undersampled tree")
    print(avg_accuracy)
    print("Prec - Recall - F1 values for undersampled tree")
    print(avg_score)

    # Save CSV file
    a = np.array(avg_score)
    np.savetxt("CSV Result/BDT/tree_under.csv", a, delimiter=';')


def tree_with_undersampling_feature_selection():
    rus = RandomUnderSampler(sampling_strategy={0: under_sampling_c1, 1: under_sampling_c2, 2: under_sampling_c3}, random_state=50, replacement=True)
    clf_tree = DecisionTreeClassifier(criterion="entropy", max_depth=bdt_max_depth)
    pipeline = make_pipeline(rus, clf_tree)

    score_array = []
    accuracy = []
    sumLeaves = []
    sumNodes = []
    for i in range(1, n_fold_split):
        X_train = np.load(f"split/Xtr_fold_{i}.npy")
        X_test = np.load(f"split/Xte_fold_{i}.npy")
        y_train = np.load(f"split/ytr_fold_{i}.npy")
        y_test = np.load(f"split/yte_fold_{i}.npy")

        selection = SelectKBest(chi2, k=20).fit(X_train, y_train)
        X_new_train = selection.transform(X_train)
        X_new_test = selection.transform(X_test)

        y_pred = pipeline.fit(X_new_train, y_train).predict(X_new_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        score_array.append(precision_recall_fscore_support(y_test, y_pred, average=None))
        sumLeaves.append(clf_tree.get_n_leaves())
        sumNodes.append(clf_tree.tree_.node_count)

    print("Average leaves")
    print(np.mean(sumLeaves, axis=0))

    print("Average nodes")
    print(np.mean(sumNodes, axis=0))

    avg_score = np.mean(score_array, axis=0)
    avg_accuracy = np.mean(accuracy, axis=0)
    print("Accuracy score for undersampled tree with feature selection")
    print(avg_accuracy)
    print("Prec - Recall - F1 values for undersampled tree with feature selection ")
    print(avg_score)

    # Save CSV file
    a = np.array(avg_score)
    np.savetxt("CSV Result/BDT/tree_under_feature.csv", a, delimiter=';')


def tree_with_oversampling():
    ros = RandomOverSampler(sampling_strategy={0:over_sampling_c1, 1:over_sampling_c2, 2:over_sampling_c3}, random_state=10)
    clf_tree = DecisionTreeClassifier(criterion="entropy", max_depth=bdt_max_depth)
    pipeline = make_pipeline(ros, clf_tree)

    score_array = []
    accuracy = []
    sumLeaves = []
    sumNodes = []
    for i in range(1, n_fold_split):
        X_train = np.load(f"split/Xtr_fold_{i}.npy")
        X_test = np.load(f"split/Xte_fold_{i}.npy")
        y_train = np.load(f"split/ytr_fold_{i}.npy")
        y_test = np.load(f"split/yte_fold_{i}.npy")
        y_pred = pipeline.fit(X_train, y_train).predict(X_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        score_array.append(precision_recall_fscore_support(y_test, y_pred, average=None))
        sumLeaves.append(clf_tree.get_n_leaves())
        sumNodes.append(clf_tree.tree_.node_count)

    print("Average leaves")
    print(np.mean(sumLeaves, axis=0))

    print("Average nodes")
    print(np.mean(sumNodes, axis=0))

    avg_score = np.mean(score_array, axis=0)
    avg_accuracy = np.mean(accuracy, axis=0)
    print("Accuracy score for oversampled tree")
    print(avg_accuracy)
    print("Prec - Recall - F1 values for oversampled tree")
    print(avg_score)

    # Save CSV file
    a = np.array(avg_score)
    np.savetxt("CSV Result/BDT/tree_over.csv", a, delimiter=';')


def tree_with_oversampling_feature_selection():
    ros = RandomOverSampler(sampling_strategy={0:over_sampling_c1, 1:over_sampling_c2, 2:over_sampling_c3}, random_state=10)
    clf_tree = DecisionTreeClassifier(criterion="entropy", max_depth=bdt_max_depth)
    pipeline = make_pipeline(ros, clf_tree)

    score_array = []
    accuracy = []
    sumLeaves = []
    sumNodes = []
    for i in range(1, n_fold_split):
        X_train = np.load(f"split/Xtr_fold_{i}.npy")
        X_test = np.load(f"split/Xte_fold_{i}.npy")
        y_train = np.load(f"split/ytr_fold_{i}.npy")
        y_test = np.load(f"split/yte_fold_{i}.npy")
        selection = SelectKBest(chi2, k=20).fit(X_train, y_train)
        X_new_train = selection.transform(X_train)
        X_new_test = selection.transform(X_test)
        y_pred = pipeline.fit(X_new_train, y_train).predict(X_new_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        score_array.append(precision_recall_fscore_support(y_test, y_pred, average=None))
        sumLeaves.append(clf_tree.get_n_leaves())
        sumNodes.append(clf_tree.tree_.node_count)

    print("Average leaves")
    print(np.mean(sumLeaves, axis=0))

    print("Average nodes")
    print(np.mean(sumNodes, axis=0))

    avg_score = np.mean(score_array, axis=0)
    avg_accuracy = np.mean(accuracy, axis=0)
    print("Accuracy score for oversampled tree with feature selection")
    print(avg_accuracy)
    print("Prec - Recall - F1 values for oversampled tree feature selection")
    print(avg_score)

    # Save CSV file
    a = np.array(avg_score)
    np.savetxt("CSV Result/BDT/tree_over_feature.csv", a, delimiter=';')


def tree_with_SMOTENN():
    smot = SMOTEENN(sampling_strategy="auto", random_state=10, n_jobs=4)
    clf_tree = DecisionTreeClassifier(criterion="entropy", max_depth=bdt_max_depth)
#    pipeline = make_pipeline(smot, clf_tree, verbose=True)
    score_array = []
    accuracy = []
    sumLeaves = []
    sumNodes = []
    for i in range(1, n_fold_split):
        X_train = np.load(f"split/Xtr_fold_{i}.npy")
        X_test = np.load(f"split/Xte_fold_{i}.npy")
        y_train = np.load(f"split/ytr_fold_{i}.npy")
        y_test = np.load(f"split/yte_fold_{i}.npy")
        X_train_smtk, y_train_smtk = smot.fit_resample(X_train, y_train)
#        y_pred = pipeline.fit(X_train, y_train).predict(X_test)
        clf = clf_tree.fit(X_train_smtk, y_train_smtk)
        y_pred = clf.predict(X_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        score_array.append(precision_recall_fscore_support(y_test, y_pred, average=None))
        sumLeaves.append(clf.get_n_leaves())
        sumNodes.append(clf.tree_.node_count)

    print("Average leaves")
    print(np.mean(sumLeaves, axis=0))

    print("Average nodes")
    print(np.mean(sumNodes, axis=0))

    avg_score = np.mean(score_array, axis=0)
    avg_accuracy = np.mean(accuracy, axis=0)
    print("Accuracy score for SMOTE tree")
    print(avg_accuracy)
    print("Prec - Recall - F1 values for SMOTEEN tree")
    print(avg_score)

    # Save CSV file
    a = np.array(avg_score)
    np.savetxt("CSV Result/BDT/tree_SMOTE.csv", a, delimiter=';')

def tree_with_SMOTENN_feature_selection():
    smot = SMOTEENN(sampling_strategy="auto", random_state=10, n_jobs=4)
    clf_tree = DecisionTreeClassifier(criterion="entropy", max_depth=bdt_max_depth)
#    pipeline = make_pipeline(smot, clf_tree, verbose=True)
    score_array = []
    accuracy = []
    sumLeaves = []
    sumNodes = []
    for i in range(1, n_fold_split):
        print("iterazione")
        X_train = np.load(f"split/Xtr_fold_{i}.npy")
        X_test = np.load(f"split/Xte_fold_{i}.npy")
        y_train = np.load(f"split/ytr_fold_{i}.npy")
        y_test = np.load(f"split/yte_fold_{i}.npy")

        selection = SelectKBest(chi2, k=20).fit(X_train, y_train)
        X_new_train = selection.transform(X_train)
        X_new_test = selection.transform(X_test)



        X_train_smtk, y_train_smtk = smot.fit_resample(X_new_train, y_train)
#        y_pred = pipeline.fit(X_train, y_train).predict(X_test)
        clf = clf_tree.fit(X_train_smtk, y_train_smtk)
        y_pred = clf.predict(X_new_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        score_array.append(precision_recall_fscore_support(y_test, y_pred, average=None))
        sumLeaves.append(clf.get_n_leaves())
        sumNodes.append(clf.tree_.node_count)

    print("Average leaves")
    print(np.mean(sumLeaves, axis=0))

    print("Average nodes")
    print(np.mean(sumNodes, axis=0))

    avg_score = np.mean(score_array, axis=0)
    avg_accuracy = np.mean(accuracy, axis=0)
    print("Accuracy score for SMOTE tree")
    print(avg_accuracy)
    print("Prec - Recall - F1 values for SMOTEEN tree feature")
    print(avg_score)

    # Save CSV file
    a = np.array(avg_score)
    np.savetxt("CSV Result/BDT/tree_SMOTE_feature.csv", a, delimiter=';')
