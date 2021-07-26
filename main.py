# K-fold split number
n_fold_split = 6

# Resampling configuration parameters
under_sampling_c1 = 2000
under_sampling_c2 = 2000
under_sampling_c3 = 635

over_sampling_c1 = 40924
over_sampling_c2 = 20000
over_sampling_c3 = 3000

# Feature selection
feature_selection = 0

# Classifier parameters
knn_neighbors = 10
bdt_max_depth = 11
rf_max_depth = 6

# Choose the rebalancing method
entire = 0
undersampling = 0
oversampling = 0
SMOTE = 1

# Choose classifier
KNN = 1
BDT = 1
naiveGaussian = 0
RF = 0

import Classifier.BinaryDecisionTree as bdt
import Classifier.KNN as knn
import Classifier.NaiveGaussianBayes as naive_gaus
import Classifier.RandomForest as rf

if __name__ == '__main__':

#   Binary Tree Decision Classifier
    if BDT == 1:
        if feature_selection == 1:
            if entire == 1:
                bdt.tree_on_entire_dataset_feature_selection()
            if undersampling == 1:
                bdt.tree_with_undersampling_feature_selection()
            if oversampling == 1:
                bdt.tree_with_oversampling_feature_selection()
            if SMOTE == 1:
                bdt.tree_with_SMOTENN_feature_selection()
        else:
            if entire == 1:
                bdt.tree_on_entire_dataset()
            if undersampling == 1:
                bdt.tree_with_undersampling()
            if oversampling == 1:
                bdt.tree_with_oversampling()
            if SMOTE == 1:
                bdt.tree_with_SMOTENN()

#   KNN Classifier
    if KNN == 1:
        if feature_selection == 1:
            if entire == 1:
                knn.knn_on_entire_dataset_feature_selection()
            if undersampling == 1:
                knn.knn_on_undersampled_dataset_feature_selection()
            if oversampling == 1:
                knn.knn_on_oversampled_dataset_feature_selection()
            if SMOTE == 1:
                knn.knn_with_SMOTENN_feature_selection()
        else:
            if entire == 1:
                knn.knn_on_entire_dataset()
            if undersampling == 1:
                knn.knn_on_undersampled_dataset()
            if oversampling == 1:
                knn.knn_on_oversampled_dataset()
            if SMOTE == 1:
                knn.knn_with_SMOTENN()

#   Naive Gaussian Bayes Classifier
    if naiveGaussian == 1:
        if feature_selection == 1:
            if entire == 1:
                naive_gaus.naive_GaussianBayesClassifier_on_entire_dataset_feature_selection()
            if undersampling == 1:
                naive_gaus.naive_GaussianBayesClassifier_on_undersampled_dataset_feature_selection()
            if oversampling == 1:
                naive_gaus.naive_GaussianBayesClassifier_on_oversampled_dataset_feature_selection()
            if SMOTE == 1:
                naive_gaus.naive_GaussianBayesClassifier_with_SMOTENN_feature_selection()
        else:
            if entire == 1:
                naive_gaus.naive_GaussianBayesClassifier_on_entire_dataset()
            if undersampling == 1:
                naive_gaus.naive_GaussianBayesClassifier_on_undersampled_dataset()
            if oversampling == 1:
                naive_gaus.naive_GaussianBayesClassifier_on_oversampled_dataset()
            if SMOTE == 1:
                naive_gaus.naive_GaussianBayesClassifier_with_SMOTENN()

#    Random Forest Classifier
    if RF == 1:
        if feature_selection == 1:
            if entire == 1:
                rf.RandomForest_on_entire_dataset_feature_selection()
            if undersampling == 1:
                rf.RandomForest_on_undersampled_dataset_feature_selection()
            if oversampling == 1:
                rf.RandomForest_on_oversampled_dataset_feature_selection()
            if SMOTE == 1:
                rf.randomForest_with_SMOTENN_feature_selection()
        else:
            if entire == 1:
                rf.RandomForest_on_entire_dataset()
            if undersampling == 1:
                rf.RandomForest_on_undersampled_dataset()
            if oversampling == 1:
                rf.RandomForest_on_oversampled_dataset()
            if SMOTE == 1:
                rf.randomForest_with_SMOTENN()

