import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE
import multilabel_oversampling as mo
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
# from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import sklearn.utils
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, confusion_matrix, \
multilabel_confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
# Use a pipeline as a high-level helper
# from transformers import pipeline, DataCollatorWithPadding
# from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
model_path = 'microsoft/deberta-v3-small'
# dataset = load_dataset('knowledgator/events_classification_biotech') 
import numpy as np

from sklearn.model_selection import train_test_split
data = pd.read_csv('final_ruling.csv')
# print(data)

#can maybe incorporate into other later but im just gonna drop the column for now
data = data.drop(columns=['Unnamed: 0', 
                          'Landlord Wants to Demolish or Convert Unit', 
                          'Landlord Rent Increase', 
                          'Other'
                          ])
# print(len(data))
# data = data[(data['Tenant-Caused Damage'] == 0) | data['Tenant-Caused Damage'] == 1] 
# print(len(data))

def isBinary(list) -> bool:
    for x in list:
        if x != 1 and x != 0:
            return False
    return True


# pipe = pipeline("fill-mask", model=model_path)
# Load model directly
# from transformers import AutoTokenizer, AutoModelForMaskedLM

# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForMaskedLM.from_pretrained(model_path)
# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# label_names = ['Tenant-Caused Damage', 'Tenant-Nonpayment', "Landlord's failure to maintain", 'Tenant caused serious problems', 'Tenant Illegal Activity', 'Landlord Family Moving in', 'Landlord Wants to Demolish or Convert Unit', 'Landlord Bad Faith Termination', 'Landlord Rent Increase', 'Other']
label_names = ['Tenant-Caused Damage', 'Tenant-Nonpayment', "Landlord Maintenance Failure", 'Tenant caused problems', 'Tenant Illegal Activity', 'Landlord Family Moving in', 'Landlord Bad Faith Termination']



# tokenized_texts = [tokenizer(text, truncation=True) for text in texts]
res = []

# ros = RandomOverSampler(random_state=42)
# X_resampled, y_resampled = ros.fit_resample(texts, labels)       


mo.seed_everything(42)
ml_oversampler = mo.MultilabelOversampler(number_of_adds=1500, number_of_tries=1500)
# new_data = ml_oversampler.fit(data, target_list=label_names)


# labels = (
#     new_data[new_data.columns[1:8]].apply(lambda row: row.dropna().tolist(), axis=1)
# )
# print(labels)
# labels = labels.to_list()
# print(len(labels))
# new_labels = list(filter(isBinary, labels))
# print(len(new_labels))
# labels = list(labels)
# texts = list(new_data['Text'].values)

# ml_oversampler.plot_all_tries()
# ml_oversampler.plot_results()

# X_train, X_test, y_train, y_test = train_test_split(
#     texts, labels, test_size=0.15, random_state=42)

train, test = train_test_split(data, test_size=0.2, random_state=42)


X_test = list(test['Text'].values)

y_test = list(test[test.columns[1:8]].apply(lambda row: row.dropna().tolist(), axis=1))


#oversample training dataset
new_train = ml_oversampler.fit(train, target_list=label_names)
# ml_oversampler.plot_all_tries()
# ml_oversampler.plot_results()

X_train = list(new_train['Text'].values)

y_train = list(new_train[new_train.columns[1:8]].apply(lambda row: row.dropna().tolist(), axis=1))

# y_train = np.array(y_train)
# y_test = np.array(y_test)

# X_train, X_test, y_train, y_test = train_test_split(
#     X_resampled, y_resampled, test_size=0.15, random_state=42)

# print(len(y_test))
# print(y_test[0])
# clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

# def sigmoid(x):
#    return 1/(1 + np.exp(-x))

# def compute_metrics(eval_pred):

#    predictions, labels = eval_pred
#    predictions = sigmoid(predictions)
#    predictions = (predictions > 0.5).astype(int).reshape(-1)
#    return clf_metrics.compute(predictions=predictions, references=labels.astype(int).reshape(-1))
# references=labels.astype(int).reshape(-1)

# MultiLabelBinarizer().fit_transform(y)
# array([[0, 0, 1, 1, 1],
#        [0, 0, 1, 0, 0],
#        [1, 1, 0, 1, 0],
#        [1, 1, 1, 1, 1],
#        [1, 1, 1, 0, 0]])

NB_pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                       ('nb_model', OneVsRestClassifier(MultinomialNB(), n_jobs=-1))])

LR_pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                        # ('SMOTE', SMOTE(random_state=42)),
                       ('lr_model', OneVsRestClassifier(LogisticRegression(random_state=42, 
                                                                           class_weight='balanced'
                                                                           ), n_jobs=-1))])

# SVM_pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
#                        ('svm_model', OneVsRestClassifier(SVC(kernel='linear',probability=True)))])

SVM_pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                        # ('SMOTE', SMOTE(random_state=42)),
                       ('svm_model', OneVsRestClassifier(LinearSVC(random_state=42,
                                                        class_weight='balanced'
                                                        ),
                                                        n_jobs=-1,
      
                                ))])


RF_pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                        # ('SMOTE', SMOTE(random_state=42)),
                       ('rf_model', RandomForestClassifier(random_state=42, 
                                                            class_weight='balanced_subsample', 
                                                            n_jobs=-1
                                                                           ))])

ML_pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                        # ('SMOTE', SMOTE(random_state=42)),
                       ('ml_model', MLPClassifier(random_state=42, 
                                                    # class_weight='balanced_subsample', 
                                                    verbose=True,
                                                    max_iter=35
                                                                           ))])

GBM_pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                        # ('SMOTE', SMOTE(random_state=42)),
                       ('gbm_model', OneVsRestClassifier(GradientBoostingClassifier(random_state=42, verbose=True), n_jobs=-1) 
                                                    # class_weight='balanced_subsample', 
                                                    # verbose=True,
                                                    # max_iter=35
                                                                           )])


def run_pipeline(pipeline, train_feats, train_lbls, test_feats, test_lbls, roc):
    pipeline.fit(train_feats, y_train)
    predictions = pipeline.predict(test_feats)
    pred_proba = pipeline.predict_proba(test_feats)
    if roc:
        print('roc_auc: ', roc_auc_score(test_lbls, pred_proba))
    print('accuracy: ', accuracy_score(test_lbls, predictions))
    # print('confusion matrices: ')
    # print(multilabel_confusion_matrix(test_lbls, predictions))
    print('classification_report: ')
    print(classification_report(test_lbls, predictions, target_names=label_names, zero_division=0))
    
def run_SVM_pipeline(pipeline, train_feats, train_lbls, test_feats, test_lbls):
    pipeline.fit(train_feats, y_train)
    predictions = pipeline.predict(test_feats)
    print('accuracy: ', accuracy_score(test_lbls, predictions))
    # print('confusion matrices: ')
    # print(multilabel_confusion_matrix(test_lbls, predictions))
    print('classification_report: ')
    print(classification_report(test_lbls, predictions, target_names=label_names, zero_division=0))



# run_pipeline(NB_pipeline, X_train, y_train, X_test, y_test)
print('LOGISTIC REGRESSION: ')
run_pipeline(LR_pipeline, X_train, y_train, X_test, y_test, True)
print('SVM: ')
run_SVM_pipeline(SVM_pipeline, X_train, y_train, X_test, y_test)
# print('Random Forest: ')
# run_pipeline(RF_pipeline, X_train, y_train, X_test, y_test, False)
# print('ML: ')
# run_pipeline(ML_pipeline, X_train, y_train, X_test, y_test, True)
# print('GBM: ')
# run_pipeline(GBM_pipeline, X_train, y_train, X_test, y_test, True)


# print(sklearn.utils.check_random_state(None))