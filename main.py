# K-fold split number
from matplotlib.font_manager import FontProperties
from matplotlib.legend_handler import HandlerBase

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
knn_neighbors = 5
bdt_max_depth = 6
rf_max_depth = 6

# Choose the rebalancing method
entire = 1
undersampling = 0
oversampling = 0
SMOTE = 0

# Choose classifier
KNN = 0
BDT = 0
naiveGaussian = 0
RF = 0

# Graph Construction
graph = 0
graph2 = 1

from matplotlib import pyplot as plt
import numpy as np
import matplotlib
import matplotlib.markers as mmarkers
import matplotlib.pyplot as plt
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

#   Graph construction
    if graph == 1:
        x = [1,2,3]
        y = [1,2,3]
        z = ['#1f77b4']
        print(z)
        fig, axs = plt.subplots()

        # marker symbol
        axs.scatter(x[1], y[1], s=80, c=z, marker=">")
        axs.set_title("marker='>'")

        plt.tight_layout()
        plt.show()


    def mscatter(x, y, ax=None, m=None, **kw):
        import matplotlib.markers as mmarkers
        if not ax: ax = plt.gca()
        sc = ax.scatter(x, y, **kw)
        if (m is not None) and (len(m) == len(x)):
            paths = []
            for marker in m:
                if isinstance(marker, mmarkers.MarkerStyle):
                    marker_obj = marker
                else:
                    marker_obj = mmarkers.MarkerStyle(marker)
                path = marker_obj.get_path().transformed(
                    marker_obj.get_transform())
                paths.append(path)
            sc.set_paths(paths)
        return sc



    if graph2 == 1:
        N = 6

        x = [0.6935, 0.7136, 0.2197, 0.1422, 0.1837, 0.7370]
        y = [0.7588, 0.7646, 0.3674, 0.4347, 0.2707, 0.8249]



        color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        markers = np.repeat(["o", "s", "D", ">", "P", "p"], N / 6)
        label = ["BDT-6", "BDT-11", "KNN-5", "KNN-10", "GaussianBayes", "RF"]

        fig, ax = plt.subplots()
        ax.legend()

        scatter = mscatter(x, y, c=color, m=markers, ax=ax)
        plt.xlabel("Recall")
        plt.ylabel("Precision")


        class MarkerHandler(HandlerBase):
            def create_artists(self, legend, tup, xdescent, ydescent,
                               width, height, fontsize, trans):
                return [plt.Line2D([width / 2], [height / 2.], ls="",
                                   marker=tup[1], color=tup[0], transform=trans)]


        ax.legend(list(zip(color, markers)), label,bbox_to_anchor=(1.05, 1.0), loc='upper left',
                  handler_map={tuple: MarkerHandler()})
        plt.tight_layout()
        plt.grid()
        plt.show()
