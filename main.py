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


def tree_on_entire_dataset():
    clf_tree = DecisionTreeClassifier(criterion="entropy", max_depth=11)
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

def tree_on_entire_dataset_feature_selection():
    clf_tree = DecisionTreeClassifier(criterion="entropy", max_depth=11)
    accuracy = []
    score_array = []
    for i in range(1, 6):
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

    avg_accuracy = np.mean(accuracy, axis=0)
    avg_score = np.mean(score_array, axis=0)
    print("Prec - Recall - F1 values for normal tree with frature selection")
    a = np.array(avg_score)
    np.savetxt('tree.csv', a, delimiter=';')
    print(avg_score)

    print(avg_accuracy)


def tree_with_undersampling():
    rus = RandomUnderSampler(sampling_strategy={0: 6000, 1: 6000, 2: 635}, random_state=50, replacement=True)
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

def tree_with_undersampling_feature_selection():
    rus = RandomUnderSampler(sampling_strategy={0: 6000, 1: 6000, 2: 635}, random_state=50, replacement=True)
    clf_tree = DecisionTreeClassifier(criterion="entropy", max_depth=6)
    pipeline = make_pipeline(rus, clf_tree)

    score_array = []
    accuracy = []
    for i in range(1, 6):
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

    avg_score = np.mean(score_array, axis=0)
    avg_accuracy = np.mean(accuracy, axis=0)
    print("Prec - Recall - F1 values for undersampled tree with feature selection ")
    a = np.array(avg_score)
    np.savetxt('treeUnder.csv', a, delimiter=';')
    print(avg_score)
    print(avg_accuracy)

def tree_with_oversampling():
    ros = RandomOverSampler(sampling_strategy={0:40924, 1:20000, 2:10000}, random_state=10)
    clf_tree = DecisionTreeClassifier(criterion="entropy", max_depth=6)
    pipeline = make_pipeline(ros, clf_tree)

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
    np.savetxt("test.csv", avg_score, delimiter=' ')
    print(avg_score)
    print(avg_accuracy)

def tree_with_oversampling_feature_selection():
    ros = RandomOverSampler(sampling_strategy={0:40924, 1:20000, 2:10000}, random_state=10)
    clf_tree = DecisionTreeClassifier(criterion="entropy", max_depth=6)
    pipeline = make_pipeline(ros, clf_tree)

    score_array = []
    accuracy = []
    for i in range(1, 6):
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

    avg_score = np.mean(score_array, axis=0)
    avg_accuracy = np.mean(accuracy, axis=0)
    print("Prec - Recall - F1 values for oversampled tree feature selection")
    a = np.array(avg_score)
    np.savetxt('treeOver.csv', a, delimiter=';')
    print(avg_score)
    print(avg_accuracy)

def tree_with_SMOTENN():
    smot = SMOTEENN(sampling_strategy="auto", random_state=10, n_jobs=4)
    clf_tree = DecisionTreeClassifier(criterion="entropy", max_depth=6)
#    pipeline = make_pipeline(smot, clf_tree, verbose=True)
    score_array = []
    accuracy = []
    for i in range(1, 6):
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
    avg_score = np.mean(score_array, axis=0)
    avg_accuracy = np.mean(accuracy, axis=0)
    print("Prec - Recall - F1 values for SMOTEEN tree")
    print(avg_score)
    print(avg_accuracy)

def knn_on_entire_dataset():
    knn = KNeighborsClassifier(n_neighbors=5)
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

def knn_on_entire_dataset_feature_selection():
    knn = KNeighborsClassifier(n_neighbors=5)
    score_array = []
    accuracy = []
    for i in range(1, 6):
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
    a = np.array(avg_score)
    np.savetxt('knn.csv', a, delimiter=';')
    print("Prec - Recall - F1 values for KNN on entire DataSet feature selection feature selection")
    print(avg_score)

def knn_on_undersampled_dataset():
    rus = RandomUnderSampler(sampling_strategy={0: 6000, 1: 6000, 2: 635}, random_state=50, replacement=True)
    knn = KNeighborsClassifier(n_neighbors=5)
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

def knn_on_undersampled_dataset_feature_selection():
    rus = RandomUnderSampler(sampling_strategy={0: 6000, 1: 6000, 2: 635}, random_state=50, replacement=True)
    knn = KNeighborsClassifier(n_neighbors=5)
    pipeline = make_pipeline(rus, knn)

    score_array = []
    accuracy = []
    for i in range(1, 6):
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
    a = np.array(avg_score)
    np.savetxt('knnUnder.csv', a, delimiter=';')
    print("Prec - Recall - F1 values for undersampled tree feature selection")
    print(avg_score)

def knn_on_oversampled_dataset():
    ros = RandomOverSampler(sampling_strategy={0:40924, 1:20000, 2:10000}, random_state=10)
    knn = KNeighborsClassifier(n_neighbors=5)
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

def knn_on_oversampled_dataset_feature_selection():
    ros = RandomOverSampler(sampling_strategy={0:40924, 1:20000, 2:10000}, random_state=10)
    knn = KNeighborsClassifier(n_neighbors=5)
    pipeline = make_pipeline(ros, knn)

    score_array = []
    accuracy = []
    for i in range(1, 6):
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
    a = np.array(avg_score)
    np.savetxt('knnOver.csv', a, delimiter=';')
    print("Prec - Recall - F1 values for oversampled tree feature selection")
    print(avg_score)

def knn_with_SMOTENN():
    smot = SMOTEENN(sampling_strategy="auto", random_state=10, n_jobs=4)
    knn = KNeighborsClassifier(n_neighbors=5)
#    pipeline = make_pipeline(smot, clf_tree, verbose=True)
    score_array = []
    accuracy = []
    for i in range(1, 6):
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
    print("Prec - Recall - F1 values for SMOTEEN knn")
    print(avg_score)
    print(avg_accuracy)


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

def naive_GaussianBayesClassifier_on_entire_dataset_feature_selection():
    gaussianBayes = GaussianNB()
    score_array = []
    accuracy = []
    for i in range(1, 6):
        X_train = np.load(f"split/Xtr_fold_{i}.npy")
        X_test = np.load(f"split/Xte_fold_{i}.npy")
        y_train = np.load(f"split/ytr_fold_{i}.npy")
        y_test = np.load(f"split/yte_fold_{i}.npy")
        selection = SelectKBest(chi2, k=20).fit(X_train, y_train)
        X_new_train = selection.transform(X_train)
        trained_GB = gaussianBayes.fit(X_new_train, y_train)
        X_new_test = selection.transform(X_test)
        y_pred = trained_GB.predict(X_new_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        score_array.append(precision_recall_fscore_support(y_test, y_pred, average=None))

    avg_accuracy = np.mean(accuracy, axis=0)
    print("Accuracy score for total GaussianB feature selection")
    print(avg_accuracy)

    avg_score = np.mean(score_array, axis=0)
    a = np.array(avg_score)
    np.savetxt('gaussian.csv', a, delimiter=';')
    print("Prec - Recall - F1 values for GaussianB on entire DataSet feature selection")
    print(avg_score)

def naive_GaussianBayesClassifier_on_undersampled_dataset():
    rus = RandomUnderSampler(sampling_strategy={0: 6000, 1: 6000, 2: 635}, random_state=50, replacement=True)
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

def naive_GaussianBayesClassifier_on_undersampled_dataset_feature_selection():
    rus = RandomUnderSampler(sampling_strategy={0: 6000, 1: 6000, 2: 635}, random_state=50, replacement=True)
    gaussianBayes = GaussianNB()
    pipeline = make_pipeline(rus, gaussianBayes)

    score_array = []
    accuracy = []
    for i in range(1, 6):
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
    print("Accuracy score for undesampled GaussianB feature selection")
    print(avg_accuracy)

    avg_score = np.mean(score_array, axis=0)
    a = np.array(avg_score)
    np.savetxt('gaussianUnder.csv', a, delimiter=';')
    print("Prec - Recall - F1 values for undesampled GaussianB feature selection")
    print(avg_score)

def naive_GaussianBayesClassifier_on_oversampled_dataset():
    ros = RandomOverSampler(sampling_strategy={0:40924, 1:20000, 2:10000}, random_state=10)
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


def naive_GaussianBayesClassifier_on_oversampled_dataset_feature_selection():
    ros = RandomOverSampler(sampling_strategy={0:40924, 1:20000, 2:10000}, random_state=10)
    gaussianBayes = GaussianNB()
    pipeline = make_pipeline(ros, gaussianBayes)

    score_array = []
    accuracy = []
    for i in range(1, 6):
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
    print("Accuracy score for oversampled GaussianB feature selection")
    print(avg_accuracy)

    avg_score = np.mean(score_array, axis=0)
    a = np.array(avg_score)
    np.savetxt('gaussianOver.csv', a, delimiter=';')
    print("Prec - Recall - F1 values for oversampled GaussianB feature selection")
    print(avg_score)

def naive_GaussianBayesClassifier_with_SMOTENN():
    smot = SMOTEENN(sampling_strategy="auto", random_state=10, n_jobs=4)
    gaussianBayes = GaussianNB()
#    pipeline = make_pipeline(smot, clf_tree, verbose=True)
    score_array = []
    accuracy = []
    for i in range(1, 6):
        X_train = np.load(f"split/Xtr_fold_{i}.npy")
        X_test = np.load(f"split/Xte_fold_{i}.npy")
        y_train = np.load(f"split/ytr_fold_{i}.npy")
        y_test = np.load(f"split/yte_fold_{i}.npy")
        X_train_smtk, y_train_smtk = smot.fit_resample(X_train, y_train)
#        y_pred = pipeline.fit(X_train, y_train).predict(X_test)
        trained_GB = gaussianBayes.fit(X_train_smtk, y_train_smtk)
        y_pred = trained_GB.predict(X_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        score_array.append(precision_recall_fscore_support(y_test, y_pred, average=None))
    avg_score = np.mean(score_array, axis=0)
    avg_accuracy = np.mean(accuracy, axis=0)
    print("Prec - Recall - F1 values for SMOTEEN GB")
    print(avg_score)
    print(avg_accuracy)


def RandomForest_on_entire_dataset():
    rf = RandomForestClassifier(criterion="entropy", max_depth=6)
    score_array = []
    accuracy = []
    for i in range(1, 6):
        X_train = np.load(f"split/Xtr_fold_{i}.npy")
        X_test = np.load(f"split/Xte_fold_{i}.npy")
        y_train = np.load(f"split/ytr_fold_{i}.npy")
        y_test = np.load(f"split/yte_fold_{i}.npy")
        trained_rf = rf.fit(X_train, y_train)
        y_pred = trained_rf.predict(X_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        score_array.append(precision_recall_fscore_support(y_test, y_pred, average=None))

    avg_accuracy = np.mean(accuracy, axis=0)
    print("Accuracy score for total RF")
    print(avg_accuracy)

    avg_score = np.mean(score_array, axis=0)
    print("Prec - Recall - F1 values for RF on entire DataSet")
    print(avg_score)


def RandomForest_on_entire_dataset_feature_selection():
    rf = RandomForestClassifier(criterion="entropy", max_depth=6)
    score_array = []
    accuracy = []
    for i in range(1, 6):
        X_train = np.load(f"split/Xtr_fold_{i}.npy")
        X_test = np.load(f"split/Xte_fold_{i}.npy")
        y_train = np.load(f"split/ytr_fold_{i}.npy")
        y_test = np.load(f"split/yte_fold_{i}.npy")

        selection = SelectKBest(chi2, k=20).fit(X_train, y_train)
        X_new_train = selection.transform(X_train)
        trained_rf = rf.fit(X_new_train, y_train)
        X_new_test = selection.transform(X_test)
        y_pred = trained_rf.predict(X_new_test)

        accuracy.append(accuracy_score(y_test, y_pred))
        score_array.append(precision_recall_fscore_support(y_test, y_pred, average=None))

    avg_accuracy = np.mean(accuracy, axis=0)
    print("Accuracy score for total RF feature selection")
    print(avg_accuracy)

    avg_score = np.mean(score_array, axis=0)
    a = np.array(avg_score)
    np.savetxt('rf.csv', a, delimiter=';')
    print("Prec - Recall - F1 values for RF on entire DataSet feature selection")
    print(avg_score)


def RandomForest_on_undersampled_dataset():
    ros = RandomUnderSampler(sampling_strategy={0: 2000, 1: 2000, 2: 635}, random_state=50, replacement=True)
    rf = RandomForestClassifier(criterion="entropy", max_depth=6)
    pipeline = make_pipeline(ros, rf)

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
    print("Accuracy score for undersampled RF")
    print(avg_accuracy)

    avg_score = np.mean(score_array, axis=0)
    print("Prec - Recall - F1 values for undersampled RF")
    print(avg_score)

def RandomForest_on_undersampled_dataset_feature_selection():
    ros = RandomUnderSampler(sampling_strategy={0: 2000, 1: 2000, 2: 635}, random_state=50, replacement=True)
    rf = RandomForestClassifier(criterion="entropy", max_depth=6)
    pipeline = make_pipeline(ros, rf)

    score_array = []
    accuracy = []
    for i in range(1, 6):
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
    print("Accuracy score for undersampled RF feature selection")
    print(avg_accuracy)

    avg_score = np.mean(score_array, axis=0)
    a = np.array(avg_score)
    np.savetxt('rfUnder.csv', a, delimiter=';')
    print("Prec - Recall - F1 values for undersampled RF feature selection")
    print(avg_score)

def RandomForest_on_oversampled_dataset():
    ros = RandomOverSampler(sampling_strategy={0:40924, 1:20000, 2:3000}, random_state=10)
    rf = RandomForestClassifier(criterion="entropy", max_depth=6)
    pipeline = make_pipeline(ros, rf)

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
    print("Accuracy score for oversampled RF")
    print(avg_accuracy)

    avg_score = np.mean(score_array, axis=0)
    print("Prec - Recall - F1 values for oversampled RF")
    print(avg_score)

def RandomForest_on_oversampled_dataset_feature_selection():
    ros = RandomOverSampler(sampling_strategy={0:40924, 1:20000, 2:3000}, random_state=10)
    rf = RandomForestClassifier(criterion="entropy", max_depth=6)
    pipeline = make_pipeline(ros, rf)

    score_array = []
    accuracy = []
    for i in range(1, 6):
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
    print("Accuracy score for oversampled RF feature selection")
    print(avg_accuracy)

    avg_score = np.mean(score_array, axis=0)
    a = np.array(avg_score)
    np.savetxt('rfOver.csv', a, delimiter=';')
    print("Prec - Recall - F1 values for oversampled RF feature selection")
    print(avg_score)

def randomForest_with_SMOTENN():
    smot = SMOTEENN(sampling_strategy="auto", random_state=10, n_jobs=4)
    rf = RandomForestClassifier(criterion="entropy", max_depth=6)
#    pipeline = make_pipeline(smot, clf_tree, verbose=True)
    score_array = []
    accuracy = []
    for i in range(1, 6):
        X_train = np.load(f"split/Xtr_fold_{i}.npy")
        X_test = np.load(f"split/Xte_fold_{i}.npy")
        y_train = np.load(f"split/ytr_fold_{i}.npy")
        y_test = np.load(f"split/yte_fold_{i}.npy")
        X_train_smtk, y_train_smtk = smot.fit_resample(X_train, y_train)
#        y_pred = pipeline.fit(X_train, y_train).predict(X_test)
        trained_rf = rf.fit(X_train_smtk, y_train_smtk)
        y_pred = trained_rf.predict(X_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        score_array.append(precision_recall_fscore_support(y_test, y_pred, average=None))
    avg_score = np.mean(score_array, axis=0)
    avg_accuracy = np.mean(accuracy, axis=0)
    print("Prec - Recall - F1 values for SMOTEEN RF")
    print(avg_score)
    print(avg_accuracy)


if __name__ == '__main__':
#    tree_on_entire_dataset()
    tree_on_entire_dataset_feature_selection()
#    tree_with_undersampling()
    tree_with_undersampling_feature_selection()
#    tree_with_oversampling()
    tree_with_oversampling_feature_selection()
#    tree_with_SMOTENN()
#    knn_on_entire_dataset()
    knn_on_entire_dataset_feature_selection()
#    knn_on_undersampled_dataset()
    knn_on_undersampled_dataset_feature_selection()
#    knn_on_oversampled_dataset()
    knn_on_oversampled_dataset_feature_selection()
#    knn_with_SMOTENN()
#    naive_GaussianBayesClassifier_on_entire_dataset()
    naive_GaussianBayesClassifier_on_entire_dataset_feature_selection()
#    naive_GaussianBayesClassifier_on_undersampled_dataset()
    naive_GaussianBayesClassifier_on_undersampled_dataset_feature_selection()
#    naive_GaussianBayesClassifier_on_oversampled_dataset()
    naive_GaussianBayesClassifier_on_oversampled_dataset_feature_selection()
#    naive_GaussianBayesClassifier_with_SMOTENN()
#    RandomForest_on_entire_dataset()
    RandomForest_on_entire_dataset_feature_selection()
#    RandomForest_on_undersampled_dataset()
    RandomForest_on_undersampled_dataset_feature_selection()
#    RandomForest_on_oversampled_dataset()
    RandomForest_on_oversampled_dataset_feature_selection()
#    randomForest_with_SMOTENN()
