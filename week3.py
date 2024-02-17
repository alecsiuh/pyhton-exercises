import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, precision_recall_fscore_support
import matplotlib as plt
from mlxtend.frequent_patterns import apriori, association_rules

# ------------- EVALUATION METRICS -------------
# exercise 1
# a.
# tp = 100
# tn = 50
# fp = 10
# fn = 5

# b.
# accuracy = (tp + tn) / (tp + fp + tn + fn)
# precision_yes = tp / (tp + fp)
# precision_no = tn / (tn + fn)
# recall_yes = tp / (tp + fn)
# recall_no = tn / (tn + fp)

# c.
# f1_yes = 2 * (precision_yes * recall_yes) / (precision_yes + recall_yes)
# f1_no = 2 * (precision_no * recall_no) / (precision_no + recall_no)

# f1_5_yes = 2.5 * (precision_yes * recall_yes) / (1.5 * precision_yes + recall_yes)
# f1_5_no = 2.5 * (precision_no * recall_no) / (1.5 * precision_no + recall_no)

def fMeasure(precision, recall, beta=1):
    return ((beta**2+1)*precision*recall)/((beta**2)*precision+recall)

# print(fMeasure(precision, recall, 1))
# print(fMeasure(precision, recall, 1.5))

# d.
# tpr = tp / (tp + fn)
# fpr = fp / ( fp + tn )

# e. make a confusion matrix
# Create the confusion matrix using sklearn.metrics
conf_matrix = np.array([[100, 5], [10, 50]])
# print(conf_matrix)

def metricsFromConfMatrix(conf_matrix, beta=1, **kwargs):
    length = len(conf_matrix[0])
    data = {}
    matrix = conf_matrix.copy()
    for i in range(0, length):
        new_conf_matrix = matrix.copy()
        new_conf_matrix[0] = matrix[i]
        new_conf_matrix[i] = matrix[0]
        matrix = new_conf_matrix.copy()

        for j in range(0, length):
            new_conf_matrix[j][0] = matrix[j][i]
            new_conf_matrix[j][i] = matrix[j][0]
        matrix = new_conf_matrix.copy()

        # get metrics
        tp = matrix[0][0]
        fn = matrix[0][1:].sum()
        fp = 0
        tn = 0
        for row in matrix[1:]:
            fp += row[0]
            tn += row[1:].sum()
        tpr = tp/(tp+fn)
        fpr = fp/(fp+tn)
        accuracy = (tp+tn)/(tp+fp+fn+tn)
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        fbeta = fMeasure(precision, recall, beta)
        labelFMeasure = "F"+str(beta)
        labelData = str(i)
        if (kwargs.get("labels")):
            labelData = kwargs.get("labels")[i]
        data[labelData] = ({"TP": tp, "FN":fn, "FP":fp, "TN":tn, "TPR": tpr, "FPR": fpr, "accuracy":accuracy, "precision":precision, "recall":recall, labelFMeasure:fbeta})
    return data

# exercise 2
# a.
# tp = 100
# tn = 5
# fp = 50
# fn = 0

# b.
# accuracy = (tp + tn) / (tp + fp + tn + fn)
# precision_yes = tp / (tp + fp)
# precision_no = tn / (tn + fn)
# recall_yes = tp / (tp + fn)
# recall_no = tn / (tn + fp)
# f1_yes = 2 * (precision_yes * recall_yes) / (precision_yes + recall_yes)
# f1_no = 2 * (precision_no * recall_no) / (precision_no + recall_no)

# c. is this a good classifier?
# no, low precision for A, low recall and F1 for B

# exercise 3
# for grass
# tp = 908
# tn = 5891
# fp = 1 # predicted as grass, but they were concrete and tree
# fn = 0

# exercise 4
# roc_curve
# slides function
# def plot_roc(y_true, y_score, title='ROC Curve', **kwargs):
#     if 'pos_label' in kwargs:
#         fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score, pos_label=kwargs.get('pos_label'))
#         auc = 1 - roc_auc_score(y_true, y_score)
#     else:
#         fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score)
#         auc = 1 - roc_auc_score(y_true, y_score)
#     # calculate optimal cut-off with Youden index method
#     optimal_idx = np.argmax(tpr - fpr)
#     optimal_threshold = thresholds[optimal_idx]
#     figsize = kwargs.get('figsize', (7, 7))
#     fig, ax = plt.subplots(1, 1, figsize=figsize)
#     ax.grid(linestyle='--')
#     # plot ROC curve
#     ax.plot(fpr, tpr, color='darkorange', label='AUC: {}'.format(auc))
#     ax.set_title(title)
#     ax.set_xlabel('False Positive Rate (FPR)')
#     ax.set_ylabel('True Positive Rate (TPR)')
#     ax.fill_between(fpr, tpr, alpha=0.3, color='darkorange', edgecolor='black')
#     # plot classifier
#     ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     # plot optimal cut-off
#     ax.scatter(fpr[optimal_idx], tpr[optimal_idx],
#                label='optimal cutoff {:.2f} op ({:.2f},{:.2f})'.format(optimal_threshold, fpr[optimal_idx],
#                                                                        tpr[optimal_idx]), color='red')
#     ax.plot([fpr[optimal_idx], fpr[optimal_idx]], [0, tpr[optimal_idx]], linestyle='--', color='red')
#     ax.plot([0, fpr[optimal_idx]], [tpr[optimal_idx], tpr[optimal_idx]], linestyle='--', color='red')
#     ax.legend(loc='lower right')
#     plt.show()

# lea funtion
def plot_roc(y_true, y_score, title='ROC curve', **kwargs):
    #roc_curve and roc_auc_score take as input two arrays:
    #   1) list of actual values (0 or 1) or labels ('m' or 'f', 'good' or 'bad', etc)
    #   2) list of odds or scores of classifier (nrs between -1 and 1 or 0 and 1)

    if 'pos_label' in kwargs:
        fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score, pos_label=kwargs.get('pos_label'))
        auc = 1-roc_auc_score(y_true, y_score)
    else:
        fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score)
        auc = roc_auc_score(y_true, y_score)

    #fpr = array of x coordinates
    #tpr = array of y coordinates
    #thresholds = array of thresholds used at the (fpr, tpr) - coordinates
    #auc = AUC

    # Calculate optimal cutoff with Youden index method
    optimal_index = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_index]

    figsize = kwargs.get('figsize', (7, 7))
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.grid(linestyle='--')

    #plot ROC curve
    ax.plot(fpr, tpr, color='darkorange', label='AUC: {}'.format(auc))
    ax.set_title(title)
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.fill_between(fpr, tpr, alpha=0.3, color='darkorange', edgecolor='black')

    #plot classifier
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    #plot optimal cut-off
    ax.scatter(fpr[optimal_index], fpr[optimal_index], label='optimal cutoff {:.2} op ({:.2},{:.2}'.format(optimal_threshold, fpr[optimal_index], tpr[optimal_index], color='red'))
    ax.plot([fpr[optimal_index], fpr[optimal_index]], [0, tpr[optimal_index]], linestyle='--', color='red')
    ax.plot([0, fpr[optimal_index]], [tpr[optimal_index], tpr[optimal_index]], linestyle='--', color='red')

    ax.legend(loc='lower right')
    plt.show()


# simpsons = pd.read_csv('simpsons_roc1.csv', delimiter=',')
#
# # Step 2: Make a model - LDA model fit on 1 input variable(V1)
# X = simpsons['y_true'].values.reshape(-1, 1)
# y = simpsons['y_score'].values # Extract the target variable as a 1D array
#
# model = LinearDiscriminantAnalysis()
# model.fit(X, y)
#
# # Step 3: confusion matrix
# predicted = model.predict(X)
# conf_matrix = confusion_matrix(y_true=y, y_pred=predicted, labels=np.unique(y))
#
# # Step 4: calculate metrics
# accuracy = accuracy_score(y_true=y, y_pred=predicted)
# precision_recall_fscore = precision_recall_fscore_support(y_true=y, y_pred=predicted, beta=1.0)
#
# # Step 5: Plot ROC curve
# y_true_for_roc = y  # 1D array containing the true labels
# y_score = model.predict_proba(X)[:, 1]  # Probability of the positive class (index 1)
# plot_roc(y_true_for_roc, y_score, pos_label='benign')

# exercise 5
# Use the plot_roc function (see slides) to draw both ROC-curves.
# From both datasets, use column '6-11yrs' as y-score and transform
# the column 'y_true' into 1 (for the values ‘6-11yrs’ and a 0 (for the
# other values).

# education1 = pd.read_csv('education_roc1.csv', delimiter=',')
# education2 = pd.read_csv('education_roc2.csv', delimiter=',')
# edu1['y_true'] = edu1['y_true'].replace('6-11yrs', 1)
# edu1['y_true'] = edu1['y_true'].replace('12+ yrs', 0)
# edu2['y_true'] = edu2['y_true'].replace('6-11yrs', 1)
# edu2['y_true'] = edu2['y_true'].replace('12+ yrs', 0)
# # plot_roc(y_true=edu1['y_true'] , y_score=edu1['6-11yrs'], title='edu1')
# # plot_roc(y_true=edu2['y_true'] , y_score=edu2['6-11yrs'], title='edu2')

# ------------- ASSOCIATION RULES -------------
# exercise 1
# a., b.
adultUCI = pd.read_csv('../datasets/AdultUCI.csv', delimiter=';')

# c. remove specific columns from the dataset
adultUCI.drop(['fnlwgt', 'education-num', 'capital-gain', 'capital-loss'], inplace=True, axis=1)

# d.
# convert the age and hours columns into classes
adultUCI.age = pd.cut(adultUCI.age, bins=[15, 25, 45, 65, 100], labels=['Young', 'Middle-aged', 'Senior', 'Old'])
# print(adultUCI.age)
adultUCI['hours-per-week'] = pd.cut(adultUCI['hours-per-week'], bins=[0, 25, 40, 60, 168], labels=['Part-time', 'Full-time', 'Over-time', 'Workaholic'])
# print(adultUCI['hours-per-week'])

# e. convert the dataframe into a transactional format
dummies = pd.get_dummies(adultUCI, prefix_sep='=')

# f. create a bar chart of items with a support of 0.1 or more
high_support = []
high_support_labels = []
# print((dummies.sum()/dummies.__len__()).index)
for support, label in zip(dummies.sum()/dummies.__len__(), (dummies.sum()/dummies.__len__()).index):
    if support >= 0.1:
        high_support.append(support)
        high_support_labels.append(label)

plt.figure(figsize=(20, 15))
fig = pd.Series(high_support).plot(kind='bar')
fig.set_xticklabels(high_support_labels)
# plt.show()
# print(high_support)

# g. what items have a very high-support
# race=White, country=US -> not a very random sample

# h. apply the apriori and association_rules algorithms with the following parameters
# support = 0.05,
# confidence = 0.6,
# minlen=2 and maxlen=3
def rule_filter(row, minlen=2, maxlen=3):
    length = len(row['antecedents']) + len(row['consequents'])
    return minlen <= length <= maxlen
itemsets = apriori(dummies, min_support=0.05, use_colnames=True, max_len=3)
# print(itemsets)
rules = association_rules(itemsets, metric='confidence', min_threshold=0.6)

count = 0
# for rule in rules.rows():
rules.apply(rule_filter, axis=1)

# print(rules) #1112


# Question 2

#a)
fruitpreferences = pd.read_csv('../datasets/Fruitpurchases.csv', delimiter=';')
#b)
fruitpreferences.drop('NameBuyer', inplace=True, axis=1)
print(fruitpreferences.columns)
itemsets = apriori(fruitpreferences, min_support=0.1, max_len=2)
rules = association_rules(itemsets, metric='confidence', min_threshold=0.3)
# print(rules)
#c)
print(rules.sort_values(by='confidence', ascending=False))  # 7 -> 9 = Pineapple -> Strawberry
#d) Pineapple
#e) Strawberry
#f) 25.4237 %
#g) Strawberries are slightly more likely to be purchased in combination with Pineapples than on their own


'''Review'''

# Evaluation Metrics
# Question 2
#a,b)
conf_matrix = np.array([[100, 0], [50, 5]])
print(conf_matrix)
print(metricsFromConfMatrix(conf_matrix))
# 'TP': 100, 'FN': 0, 'FP': 50, 'TN': 5
# 'accuracy': 0.6774193548387096
# 'precision': 0.6666666666666666
# 'recall': 1.0, 'F1': 0.8
#c)
# Slightly better than random ig
