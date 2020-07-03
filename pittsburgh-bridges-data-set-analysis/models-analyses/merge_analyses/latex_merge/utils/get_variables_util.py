# =========================================================================== #
# sklearn IMPORT
# =========================================================================== #
from sklearn.decomposition import PCA, KernelPCA

# Import scikit-learn classes: models (Estimators).
from sklearn.naive_bayes import GaussianNB, MultinomialNB   # Non-parametric Generative Model
from sklearn.linear_model import LogisticRegression, SGDClassifier         # Parametric Linear Discriminative Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC                          # Parametric Linear Discriminative "Support Vector Classifier"
from sklearn.tree import DecisionTreeClassifier      # Non-parametric Model
from sklearn.ensemble import RandomForestClassifier  # Non-parametric Model (Meta-Estimator, that is, an Ensemble Method)

def get_dataset_location():
    dataset_path = 'C:\\Users\\Francesco\Documents\\datasets\\pittsburgh_dataset'
    dataset_name = 'bridges.data.csv'
    column_names = ['RIVER', 'LOCATION', 'ERECTED', 'PURPOSE', 'LENGTH', 'LANES', 'CLEAR-G', 'T-OR-D', 'MATERIAL', 'SPAN', 'REL-L', 'TYPE']
    TARGET_COL = 'T-OR-D'
    return dataset_path, dataset_name, column_names, TARGET_COL

def get_estimators(random_state=0):
    estimators_list = [GaussianNB(), LogisticRegression(random_state=random_state), KNeighborsClassifier(), SGDClassifier(random_state=random_state), SVC(random_state=random_state, probability=True), DecisionTreeClassifier(random_state=0), RandomForestClassifier(random_state=random_state)]
    def get_clf_name(a_clf):
        
        clf_name = str(a_clf).split('(')[0]
        if clf_name.endswith('Classifier'):
            clf_name = clf_name.split('Classifier')[0]
            pass
        if clf_name == 'KNeighbors': clf_name = 'Knn'
        elif clf_name == 'DecisionTree': clf_name = 'Tree'
        elif clf_name == 'LogisticRegression': clf_name = 'LogReg'
        
        return clf_name
    estimators_names = list(map(get_clf_name, estimators_list))
    return estimators_list, estimators_names