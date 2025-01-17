# Import libraries
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# Read legitimate and phishing CSV files and create pandas dataframes
legitimate_df = pd.read_csv("structured_data/structured_data_legitimate.csv")
phishing_df = pd.read_csv("structured_data/structured_data_phishing.csv")


# Combine legitimate and phishing dataframes, and shuffle
df = pd.concat([legitimate_df, phishing_df], axis=0)

df = df.sample(frac=1)

# Remove URL some unnecessary columns from feature vectors. Create X (test) and Y (expected answer) for the models, Supervised Learning
df = df.drop("URL", axis=1)

df = df.drop("number_of_clickable_button", axis = 1)

df = df.drop("number_of_images", axis = 1)

df = df.drop_duplicates()

X = df.drop('label', axis=1)

Y = df['label']

# Split data to train and test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=60)

# Decision Tree
dt_model = tree.DecisionTreeClassifier()

# AdaBoost
ab_model = AdaBoostClassifier()

# Gaussian Naive Bayes
nb_model = GaussianNB()

# KNeighborsClassifier
kn_model = KNeighborsClassifier()




# K-fold cross validation, and K = 5
K = 5
total = X.shape[0]
index = int(total / K)

# 1
X_1_test = X.iloc[:index]
X_1_train = X.iloc[index:]
Y_1_test = Y.iloc[:index]
Y_1_train = Y.iloc[index:]

# 2
X_2_test = X.iloc[index:index*2]
X_2_train = X.iloc[np.r_[:index, index*2:]]
Y_2_test = Y.iloc[index:index*2]
Y_2_train = Y.iloc[np.r_[:index, index*2:]]

# 3
X_3_test = X.iloc[index*2:index*3]
X_3_train = X.iloc[np.r_[:index*2, index*3:]]
Y_3_test = Y.iloc[index*2:index*3]
Y_3_train = Y.iloc[np.r_[:index*2, index*3:]]

# 4
X_4_test = X.iloc[index*3:index*4]
X_4_train = X.iloc[np.r_[:index*3, index*4:]]
Y_4_test = Y.iloc[index*3:index*4]
Y_4_train = Y.iloc[np.r_[:index*3, index*4:]]

# 5
X_5_test = X.iloc[index*4:]
X_5_train = X.iloc[:index*4]
Y_5_test = Y.iloc[index*4:]
Y_5_train = Y.iloc[:index*4]


# X and Y train and test lists
X_train_list = [X_1_train, X_2_train, X_3_train, X_4_train, X_5_train]
X_test_list = [X_1_test, X_2_test, X_3_test, X_4_test, X_5_test]

Y_train_list = [Y_1_train, Y_2_train, Y_3_train, Y_4_train, Y_5_train]
Y_test_list = [Y_1_test, Y_2_test, Y_3_test, Y_4_test, Y_5_test]


def calculate_metrics(TN, TP, FN, FP):
    model_accuracy = (TP + TN) / (TP + TN + FN + FP)
    model_precision = TP / (TP + FP)
    model_recall = TP / (TP + FN)
    model_f1_score = 2 * (model_precision * model_recall) / (model_precision + model_recall)
    return model_accuracy, model_precision, model_recall, model_f1_score


rf_accuracy_list, rf_precision_list, rf_recall_list, rf_f1_score_list = [], [], [], []
dt_accuracy_list, dt_precision_list, dt_recall_list, dt_f1_score_list  = [], [], [], []
ab_accuracy_list, ab_precision_list, ab_recall_list, ab_f1_score_list  = [], [], [], []
svm_accuracy_list, svm_precision_list, svm_recall_list, svm_f1_score_list  = [], [], [], []
nb_accuracy_list, nb_precision_list, nb_recall_list, nb_f1_score_list  = [], [], [], []
nn_accuracy_list, nn_precision_list, nn_recall_list, nn_f1_score_list  = [], [], [], []
kn_accuracy_list, kn_precision_list, kn_recall_list, kn_f1_score_list  = [], [], [], []



for i in range(0, K):
    # ----- RANDOM FOREST ----- #
    rf_model.fit(X_train_list[i], Y_train_list[i])
    rf_predictions = rf_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=rf_predictions).ravel()
    rf_accuracy, rf_precision, rf_recall, rf_f1_score = calculate_metrics(tn, tp, fn, fp)
    rf_accuracy_list.append(rf_accuracy)
    rf_precision_list.append(rf_precision)
    rf_recall_list.append(rf_recall)
    rf_f1_score_list.append(rf_f1_score)
    

    # ----- DECISION TREE ----- #
    dt_model.fit(X_train_list[i], Y_train_list[i])
    dt_predictions = dt_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=dt_predictions).ravel()
    dt_accuracy, dt_precision, dt_recall, dt_f1_score = calculate_metrics(tn, tp, fn, fp)
    dt_accuracy_list.append(dt_accuracy)
    dt_precision_list.append(dt_precision)
    dt_recall_list.append(dt_recall)
    dt_f1_score_list.append(dt_f1_score)


    # ----- ADABOOST ----- #
    ab_model.fit(X_train_list[i], Y_train_list[i])
    ab_predictions = ab_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=ab_predictions).ravel()
    ab_accuracy, ab_precision, ab_recall, ab_f1_score = calculate_metrics(tn, tp, fn, fp)
    ab_accuracy_list.append(ab_accuracy)
    ab_precision_list.append(ab_precision)
    ab_recall_list.append(ab_recall)
    ab_f1_score_list.append(ab_f1_score)

    # ----- GAUSSIAN NAIVE BAYES ----- #
    nb_model.fit(X_train_list[i], Y_train_list[i])
    nb_predictions = nb_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=nb_predictions).ravel()
    nb_accuracy, nb_precision, nb_recall, nb_f1_score = calculate_metrics(tn, tp, fn, fp)
    nb_accuracy_list.append(nb_accuracy)
    nb_precision_list.append(nb_precision)
    nb_recall_list.append(nb_recall)
    nb_f1_score_list.append(nb_f1_score)    


    # ----- K-NEIGHBOURS CLASSIFIER ----- #
    kn_model.fit(X_train_list[i], Y_train_list[i])
    kn_predictions = kn_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=kn_predictions).ravel()
    kn_accuracy, kn_precision, kn_recall, kn_f1_score = calculate_metrics(tn, tp, fn, fp)
    kn_accuracy_list.append(kn_accuracy)
    kn_precision_list.append(kn_precision)
    kn_recall_list.append(kn_recall)
    kn_f1_score_list.append(kn_f1_score)


def calculate_average(list):
    return sum(list) / len(list)

RF_accuracy = calculate_average(rf_accuracy_list)
RF_precision = calculate_average(rf_precision_list)
RF_recall = calculate_average(rf_recall_list)
RF_f1_score  = calculate_average(rf_f1_score_list)

print("Random Forest accuracy ==> ", RF_accuracy)
print("Random Forest precision ==> ", RF_precision)
print("Random Forest recall ==> ", RF_recall)
print("Random Forest f1 score ==> ", RF_f1_score)


DT_accuracy = calculate_average(dt_accuracy_list)
DT_precision = calculate_average(dt_precision_list)
DT_recall = calculate_average(dt_recall_list)
DT_f1_score = calculate_average(dt_f1_score_list)

print("Decision Tree accuracy ==> ", DT_accuracy)
print("Decision Tree precision ==> ", DT_precision)
print("Decision Tree recall ==> ", DT_recall)
print("Decision Tree f1 score ==> ", DT_f1_score)

AB_accuracy = calculate_average(ab_accuracy_list)
AB_precision = calculate_average(ab_precision_list)
AB_recall = calculate_average(ab_recall_list)
AB_f1_score = calculate_average(ab_f1_score_list)

print("AdaBoost accuracy ==> ", AB_accuracy)
print("AdaBoost precision ==> ", AB_precision)
print("AdaBoost recall ==> ", AB_recall)
print("AdaBoost f1 score ==> ", AB_f1_score)


NB_accuracy = calculate_average(nb_accuracy_list)
NB_precision = calculate_average(nb_precision_list)
NB_recall = calculate_average(nb_recall_list)
NB_f1_score = calculate_average(nb_f1_score_list)

print("Gaussian Naive Bayes accuracy ==> ", NB_accuracy)
print("Gaussian Naive Bayes precision ==> ", NB_precision)
print("Gaussian Naive Bayes recall ==> ", NB_recall)
print("Gaussian Naive Bayes f1 score ==> ", NB_f1_score)


KN_accuracy = calculate_average(kn_accuracy_list)
KN_precision = calculate_average(kn_precision_list)
KN_recall = calculate_average(kn_recall_list)
KN_f1_score = calculate_average(kn_f1_score_list)

print("K-Neighbours Classifier accuracy ==> ", KN_accuracy)
print("K-Neighbours Classifier precision ==> ", KN_precision)
print("K-Neighbours Classifier recall ==> ", KN_recall)
print("K-Neighbours Classifier f1 score ==> ", KN_f1_score)


data = {'accuracy': [NB_accuracy, DT_accuracy, RF_accuracy, AB_accuracy,  KN_accuracy],
        'precision': [NB_precision,  DT_precision, RF_precision, AB_precision, KN_precision],
        'recall': [NB_recall,  DT_recall, RF_recall, AB_recall,  KN_recall],
    }

index = ['NB', 'DT', 'RF', 'AB', 'KN']

f1_score = {
    'F1-Score': [NB_f1_score, DT_f1_score, RF_f1_score, AB_f1_score, KN_f1_score]
}

df_results = pd.DataFrame(data=data, index=index)

df_f1_score = pd.DataFrame(data=f1_score, index=index)


# visualize the dataframe
ax = df_results.plot.bar(rot=0)
ax = df_f1_score.plot.bar(rot=0)
plt.show()