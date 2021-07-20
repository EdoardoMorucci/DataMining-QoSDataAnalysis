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


def tree_on_entire_dataset():
    clf_tree = DecisionTreeClassifier(criterion="entropy", max_depth=6)
    accuracy = []
    score_array = []
    for i in range(1, 6):
        X_train = np.load(f"split/Xtr_fold_{i}.npy")
        X_test = np.load(f"split/Xte_fold_{i}.npy")
        y_train = np.load(f"split/ytr_fold_{i}.npy")
        y_test = np.load(f"split/yte_fold_{i}.npy")
        clf = clf_tree.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        score_array.append(precision_recall_fscore_support(y_test, y_pred, average=None))

    avg_accuracy = np.mean(accuracy, axis=0)
    avg_score = np.mean(score_array, axis=0)
    print("Prec - Recall - F1 values for normal tree")
    print(avg_score)

    print(avg_accuracy)

def tree_with_undersampling():
    rus = RandomUnderSampler(sampling_strategy={0: 2000, 1: 2000, 2: 635}, random_state=10, replacement=True)
    clf_tree = DecisionTreeClassifier(criterion="entropy", max_depth=6)
    pipeline = make_pipeline(rus, clf_tree)

    score_array = []
    accuracy = []
    for i in range(1, 6):
        X_train = np.load(f"split/Xtr_fold_{i}.npy")
        X_test = np.load(f"split/Xte_fold_{i}.npy")
        y_train = np.load(f"split/ytr_fold_{i}.npy")
        y_test = np.load(f"split/yte_fold_{i}.npy")
        y_pred = pipeline.fit(X_train, y_train).predict(X_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        score_array.append(precision_recall_fscore_support(y_test, y_pred, average=None))

    avg_score = np.mean(score_array, axis=0)
    avg_accuracy = np.mean(accuracy, axis=0)
    print("Prec - Recall - F1 values for undersampled tree")
    print(avg_score)
    print(avg_accuracy)

def tree_with_oversampling():
    ros = RandomOverSampler(sampling_strategy="all", random_state=10)
    clf_tree = DecisionTreeClassifier(criterion="entropy", max_depth=6)
    pipeline = make_pipeline(ros, clf_tree, verbose=True)

    score_array = []
    accuracy = []
    for i in range(1, 6):
        X_train = np.load(f"split/Xtr_fold_{i}.npy")
        X_test = np.load(f"split/Xte_fold_{i}.npy")
        y_train = np.load(f"split/ytr_fold_{i}.npy")
        y_test = np.load(f"split/yte_fold_{i}.npy")
        y_pred = pipeline.fit(X_train, y_train).predict(X_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        score_array.append(precision_recall_fscore_support(y_test, y_pred, average=None))

    avg_score = np.mean(score_array, axis=0)
    avg_accuracy = np.mean(accuracy, axis=0)
    print("Prec - Recall - F1 values for oversampled tree")
    print(avg_score)
    print(avg_accuracy)

def tree_with_SMOTENN():
    smot = SMOTETomek(sampling_strategy="auto", random_state=10, n_jobs=4)
    clf_tree = DecisionTreeClassifier(criterion="entropy", max_depth=6)
#    pipeline = make_pipeline(smot, clf_tree, verbose=True)
    print("prima del for")
    score_array = []
    accuracy = []
    for i in range(1, 6):
        X_train = np.load(f"split/Xtr_fold_{i}.npy")
        X_test = np.load(f"split/Xte_fold_{i}.npy")
        y_train = np.load(f"split/ytr_fold_{i}.npy")
        y_test = np.load(f"split/yte_fold_{i}.npy")
        print("prima del predict")
        X_train_smtk, y_train_smtk = smot.fit_resample(X_train, y_train)
#        y_pred = pipeline.fit(X_train, y_train).predict(X_test)
        print("dopo il sampling")
        clf = clf_tree.fit(X_train_smtk, y_train_smtk)
        y_pred = clf.predict(X_test)
        print("dopo il predict")
        accuracy.append(accuracy_score(y_test, y_pred))
        score_array.append(precision_recall_fscore_support(y_test, y_pred, average=None))
    print("dopo il for")
    avg_score = np.mean(score_array, axis=0)
    avg_accuracy = np.mean(accuracy, axis=0)
    print("Prec - Recall - F1 values for SMOTEEN tree")
    print(avg_score)
    print(avg_accuracy)

def knn_on_entire_dataset():
    knn = KNeighborsClassifier(n_neighbors=10)
    score_array = []
    accuracy = []
    for i in range(1, 6):
        X_train = np.load(f"split/Xtr_fold_{i}.npy")
        X_test = np.load(f"split/Xte_fold_{i}.npy")
        y_train = np.load(f"split/ytr_fold_{i}.npy")
        y_test = np.load(f"split/yte_fold_{i}.npy")
        trained_knn = knn.fit(X_train, y_train)
        y_pred = trained_knn.predict(X_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        score_array.append(precision_recall_fscore_support(y_test, y_pred, average=None))

    avg_accuracy = np.mean(accuracy, axis=0)
    print("Accuracy score for total KNN")
    print(avg_accuracy)

    avg_score = np.mean(score_array, axis=0)
    print("Prec - Recall - F1 values for KNN on entire DataSet")
    print(avg_score)

def knn_on_undersampled_dataset():
    rus = RandomUnderSampler(sampling_strategy={0: 6000, 1: 6000, 2: 635}, random_state=50, replacement=False)
    knn = KNeighborsClassifier(n_neighbors=10)
    pipeline = make_pipeline(rus, knn)

    score_array = []
    accuracy = []
    for i in range(1, 6):
        X_train = np.load(f"split/Xtr_fold_{i}.npy")
        X_test = np.load(f"split/Xte_fold_{i}.npy")
        y_train = np.load(f"split/ytr_fold_{i}.npy")
        y_test = np.load(f"split/yte_fold_{i}.npy")
        y_pred = pipeline.fit(X_train, y_train).predict(X_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        score_array.append(precision_recall_fscore_support(y_test, y_pred, average=None))


    avg_accuracy = np.mean(accuracy, axis=0)
    print("Accuracy score for undersampled KNN")
    print(avg_accuracy)

    avg_score = np.mean(score_array, axis=0)
    print("Prec - Recall - F1 values for undersampled tree")
    print(avg_score)

def knn_on_oversampled_dataset():
    ros = RandomOverSampler(sampling_strategy="not majority", random_state=10)
    knn = KNeighborsClassifier(n_neighbors=10)
    pipeline = make_pipeline(ros, knn)

    score_array = []
    accuracy = []
    for i in range(1, 6):
        X_train = np.load(f"split/Xtr_fold_{i}.npy")
        X_test = np.load(f"split/Xte_fold_{i}.npy")
        y_train = np.load(f"split/ytr_fold_{i}.npy")
        y_test = np.load(f"split/yte_fold_{i}.npy")
        y_pred = pipeline.fit(X_train, y_train).predict(X_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        score_array.append(precision_recall_fscore_support(y_test, y_pred, average=None))

    avg_accuracy = np.mean(accuracy, axis=0)
    print("Accuracy score for oversampled KNN")
    print(avg_accuracy)

    avg_score = np.mean(score_array, axis=0)
    print("Prec - Recall - F1 values for oversampled tree")
    print(avg_score)


def naive_GaussianBayesClassifier_on_entire_dataset():
    gaussianBayes = GaussianNB()
    score_array = []
    accuracy = []
    for i in range(1, 6):
        X_train = np.load(f"split/Xtr_fold_{i}.npy")
        X_test = np.load(f"split/Xte_fold_{i}.npy")
        y_train = np.load(f"split/ytr_fold_{i}.npy")
        y_test = np.load(f"split/yte_fold_{i}.npy")
        trained_GB = gaussianBayes.fit(X_train, y_train)
        y_pred = trained_GB.predict(X_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        score_array.append(precision_recall_fscore_support(y_test, y_pred, average=None))

    avg_accuracy = np.mean(accuracy, axis=0)
    print("Accuracy score for total GaussianB")
    print(avg_accuracy)

    avg_score = np.mean(score_array, axis=0)
    print("Prec - Recall - F1 values for GaussianB on entire DataSet")
    print(avg_score)

def naive_GaussianBayesClassifier_on_undersampled_dataset():
    rus = RandomUnderSampler(sampling_strategy={0:6000, 1:6000, 2:635}, random_state=10, replacement=True)
    gaussianBayes = GaussianNB()
    pipeline = make_pipeline(rus, gaussianBayes)

    score_array = []
    accuracy = []
    for i in range(1, 6):
        X_train = np.load(f"split/Xtr_fold_{i}.npy")
        X_test = np.load(f"split/Xte_fold_{i}.npy")
        y_train = np.load(f"split/ytr_fold_{i}.npy")
        y_test = np.load(f"split/yte_fold_{i}.npy")
        y_pred = pipeline.fit(X_train, y_train).predict(X_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        score_array.append(precision_recall_fscore_support(y_test, y_pred, average=None))

    avg_accuracy = np.mean(accuracy, axis=0)
    print("Accuracy score for undesampled GaussianB")
    print(avg_accuracy)

    avg_score = np.mean(score_array, axis=0)
    print("Prec - Recall - F1 values for undesampled GaussianB")
    print(avg_score)

def naive_GaussianBayesClassifier_on_oversampled_dataset():
    ros = RandomOverSampler(sampling_strategy="not majority", random_state=10)
    gaussianBayes = GaussianNB()
    pipeline = make_pipeline(ros, gaussianBayes)

    score_array = []
    accuracy = []
    for i in range(1, 6):
        X_train = np.load(f"split/Xtr_fold_{i}.npy")
        X_test = np.load(f"split/Xte_fold_{i}.npy")
        y_train = np.load(f"split/ytr_fold_{i}.npy")
        y_test = np.load(f"split/yte_fold_{i}.npy")
        y_pred = pipeline.fit(X_train, y_train).predict(X_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        score_array.append(precision_recall_fscore_support(y_test, y_pred, average=None))

    avg_accuracy = np.mean(accuracy, axis=0)
    print("Accuracy score for oversampled GaussianB")
    print(avg_accuracy)

    avg_score = np.mean(score_array, axis=0)
    print("Prec - Recall - F1 values for oversampled GaussianB")
    print(avg_score)


if __name__ == '__main__':
#    tree_on_entire_dataset()
#    tree_with_undersampling()
#    tree_with_oversampling()
    tree_with_SMOTENN()
#    knn_on_entire_dataset()
#    knn_on_undersampled_dataset()
#    knn_on_oversampled_dataset()
#    naive_GaussianBayesClassifier_on_entire_dataset()
#    naive_GaussianBayesClassifier_on_undersampled_dataset()
#    naive_GaussianBayesClassifier_on_oversampled_dataset()
