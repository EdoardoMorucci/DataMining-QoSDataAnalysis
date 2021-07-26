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

from main import knn_neighbors, n_fold_split
from main import under_sampling_c1, under_sampling_c2, under_sampling_c3
from main import over_sampling_c1, over_sampling_c2, over_sampling_c3


def knn_on_entire_dataset():
    knn = KNeighborsClassifier(n_neighbors=knn_neighbors)
    score_array = []
    accuracy = []
    for i in range(1, n_fold_split):
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

    # Save CSV file
    a = np.array(avg_score)
    np.savetxt('CSV Result/KNN/knn_entire.csv', a, delimiter=';')


def knn_on_entire_dataset_feature_selection():
    knn = KNeighborsClassifier(n_neighbors=knn_neighbors)
    score_array = []
    accuracy = []
    for i in range(1, n_fold_split):
        X_train = np.load(f"split/Xtr_fold_{i}.npy")
        X_test = np.load(f"split/Xte_fold_{i}.npy")
        y_train = np.load(f"split/ytr_fold_{i}.npy")
        y_test = np.load(f"split/yte_fold_{i}.npy")
        selection = SelectKBest(chi2, k=20).fit(X_train, y_train)
        X_new_train = selection.transform(X_train)
        trained_knn = knn.fit(X_new_train, y_train)
        X_new_test = selection.transform(X_test)
        y_pred = trained_knn.predict(X_new_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        score_array.append(precision_recall_fscore_support(y_test, y_pred, average=None))

    avg_accuracy = np.mean(accuracy, axis=0)
    print("Accuracy score for total KNN feature selection feature selection")
    print(avg_accuracy)

    avg_score = np.mean(score_array, axis=0)
    print("Prec - Recall - F1 values for KNN on entire DataSet feature selection feature selection")
    print(avg_score)

    # Save CSV file
    a = np.array(avg_score)
    np.savetxt('CSV Result/KNN/knn_entire_feature.csv', a, delimiter=';')

def knn_on_undersampled_dataset():
    rus = RandomUnderSampler(sampling_strategy={0: under_sampling_c1, 1: under_sampling_c2, 2: under_sampling_c3}, random_state=50, replacement=True)
    knn = KNeighborsClassifier(n_neighbors=knn_neighbors)
    pipeline = make_pipeline(rus, knn)

    score_array = []
    accuracy = []
    for i in range(1, n_fold_split):
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
    print("Prec - Recall - F1 values for undersampled KNN")
    print(avg_score)

    # Save CSV file
    a = np.array(avg_score)
    np.savetxt('CSV Result/KNN/knn_under.csv', a, delimiter=';')

def knn_on_undersampled_dataset_feature_selection():
    rus = RandomUnderSampler(sampling_strategy={0: under_sampling_c1, 1: under_sampling_c2, 2: under_sampling_c3}, random_state=50, replacement=True)
    knn = KNeighborsClassifier(n_neighbors=knn_neighbors)
    pipeline = make_pipeline(rus, knn)

    score_array = []
    accuracy = []
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


    avg_accuracy = np.mean(accuracy, axis=0)
    print("Accuracy score for undersampled KNN feature selection")
    print(avg_accuracy)

    avg_score = np.mean(score_array, axis=0)
    print("Prec - Recall - F1 values for undersampled KNN feature selection")
    print(avg_score)

    # Save CSV file
    a = np.array(avg_score)
    np.savetxt('CSV Result/KNN/knn_under_feature.csv', a, delimiter=';')

def knn_on_oversampled_dataset():
    ros = RandomOverSampler(sampling_strategy={0:over_sampling_c1, 1:over_sampling_c2, 2:over_sampling_c3}, random_state=10)
    knn = KNeighborsClassifier(n_neighbors=knn_neighbors)
    pipeline = make_pipeline(ros, knn)

    score_array = []
    accuracy = []
    for i in range(1, n_fold_split):
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
    print("Prec - Recall - F1 values for oversampled KNN")
    print(avg_score)

    # Save CSV file
    a = np.array(avg_score)
    np.savetxt('CSV Result/KNN/knn_over.csv', a, delimiter=';')

def knn_on_oversampled_dataset_feature_selection():
    ros = RandomOverSampler(sampling_strategy={0:40924, 1:20000, 2:10000}, random_state=10)
    knn = KNeighborsClassifier(n_neighbors=knn_neighbors)
    pipeline = make_pipeline(ros, knn)

    score_array = []
    accuracy = []
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

    avg_accuracy = np.mean(accuracy, axis=0)
    print("Accuracy score for oversampled KNN feature selection")
    print(avg_accuracy)

    avg_score = np.mean(score_array, axis=0)
    print("Prec - Recall - F1 values for oversampled KNN feature selection")
    print(avg_score)

    # Save CSV file
    a = np.array(avg_score)
    np.savetxt('CSV Result/KNN/knn_over_feature.csv', a, delimiter=';')

def knn_with_SMOTENN():
    smot = SMOTEENN(sampling_strategy="auto", random_state=10, n_jobs=4)
    knn = KNeighborsClassifier(n_neighbors=knn_neighbors)
#    pipeline = make_pipeline(smot, clf_tree, verbose=True)
    score_array = []
    accuracy = []
    for i in range(1, n_fold_split):
        X_train = np.load(f"split/Xtr_fold_{i}.npy")
        X_test = np.load(f"split/Xte_fold_{i}.npy")
        y_train = np.load(f"split/ytr_fold_{i}.npy")
        y_test = np.load(f"split/yte_fold_{i}.npy")
        X_train_smtk, y_train_smtk = smot.fit_resample(X_train, y_train)
#        y_pred = pipeline.fit(X_train, y_train).predict(X_test)
        trained_knn = knn.fit(X_train_smtk, y_train_smtk)
        y_pred = trained_knn.predict(X_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        score_array.append(precision_recall_fscore_support(y_test, y_pred, average=None))
    avg_score = np.mean(score_array, axis=0)
    avg_accuracy = np.mean(accuracy, axis=0)
    a = np.array(avg_score)
    np.savetxt('knnSMOTE.csv', a, delimiter=';')
    print("Accuracy score for SMOTE KNN")
    print(avg_accuracy)

    print("Prec - Recall - F1 values for SMOTEEN knn")
    print(avg_score)

    # Save CSV file
    a = np.array(avg_score)
    np.savetxt('CSV Result/KNN/knn_SMOTE.csv', a, delimiter=';')

def knn_with_SMOTENN_feature_selection():
    smot = SMOTEENN(sampling_strategy="auto", random_state=10, n_jobs=4)
    knn = KNeighborsClassifier(n_neighbors=knn_neighbors)
#    pipeline = make_pipeline(smot, clf_tree, verbose=True)
    score_array = []
    accuracy = []
    for i in range(1, n_fold_split):
        X_train = np.load(f"split/Xtr_fold_{i}.npy")
        X_test = np.load(f"split/Xte_fold_{i}.npy")
        y_train = np.load(f"split/ytr_fold_{i}.npy")
        y_test = np.load(f"split/yte_fold_{i}.npy")

        selection = SelectKBest(chi2, k=20).fit(X_train, y_train)
        X_new_train = selection.transform(X_train)
        X_new_test = selection.transform(X_test)

        X_train_smtk, y_train_smtk = smot.fit_resample(X_new_train, y_train)
#        y_pred = pipeline.fit(X_train, y_train).predict(X_test)
        trained_knn = knn.fit(X_train_smtk, y_train_smtk)
        y_pred = trained_knn.predict(X_new_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        score_array.append(precision_recall_fscore_support(y_test, y_pred, average=None))
    avg_score = np.mean(score_array, axis=0)
    avg_accuracy = np.mean(accuracy, axis=0)
    a = np.array(avg_score)
    np.savetxt('knnSMOTE.csv', a, delimiter=';')
    print("Accuracy score for SMOTE KNN")
    print(avg_accuracy)

    print("Prec - Recall - F1 values for SMOTEEN knn feature")
    print(avg_score)

    # Save CSV file
    a = np.array(avg_score)
    np.savetxt('CSV Result/KNN/knn_SMOTE_feature.csv', a, delimiter=';')