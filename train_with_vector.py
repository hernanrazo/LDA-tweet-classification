import pickle
import numpy as np
import pandas as pd
import warnings
from sklearn import linear_model
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
feature_vectors = []
data = pd.read_csv('data/tweet_data.csv', encoding='ISO-8859-1')
bigram = pd.read_pickle('saved_pickles_models/train_bigram.pkl')
id2word = pd.read_pickle('saved_pickles_models/train_id2word.pkl')
corpus = pd.read_pickle('saved_pickles_models/train_corpus.pkl')
model = pd.read_pickle('saved_pickles_models/lda_model.model')

#get distributions from every tweet in data
print('Getting distribution...')
for i in range(len(data)):
    top_topics = model.get_document_topics(corpus[i], minimum_probability=0.0)
    topic_vector = [top_topics[i][1] for i in range(15)]
    feature_vectors.append(topic_vector)

x = np.array(feature_vectors)
y = np.array(data.relevant)

kf = KFold(5, shuffle=True, random_state=42)
cv_lr_f1, cv_lrsgd_f1, cv_svcsgd_f1 = [], [], []

print('Starting calculations...')
for train_ind, val_ind in kf.split(x, y):
    x_train, y_train = x[train_ind], y[train_ind]
    x_val, y_val = x[val_ind], y[val_ind]

    scaler = StandardScaler()
    x_train_scale = scaler.fit_transform(x_train)
    x_val_scale = scaler.transform(x_val)

    log_reg = LogisticRegression(class_weight='balanced', solver='newton-cg', fit_intercept=True).fit(x_train_scale, y_train)

    y_pred = log_reg.predict(x_val_scale)
    cv_lr_f1.append(f1_score(y_val, y_pred, average='binary'))

    sgd = linear_model.SGDClassifier(max_iter=1000, tol=1e-3, loss='log', class_weight='balanced').fit(x_train_scale, y_train)

    y_pred = sgd.predict(x_val_scale)
    cv_lrsgd_f1.append(f1_score(y_val, y_pred, average='binary'))


    sgd_huber = linear_model.SGDClassifier(max_iter=1000, tol=1e-3, alpha=20,loss='modified_huber', class_weight='balanced').fit(x_train_scale, y_train)

    y_pred = sgd_huber.predict(x_val_scale)
    cv_svcsgd_f1.append(f1_score(y_val, y_pred, average='binary'))

print(f'Logistic Regression Val f1: {np.mean(cv_lr_f1):.3f} +- {np.std(cv_lr_f1):.3f}')
print(f'Logisitic Regression SGD Val f1: {np.mean(cv_lrsgd_f1):.3f} +- {np.std(cv_lrsgd_f1):.3f}')
print(f'SVM Huber Val f1: {np.mean(cv_svcsgd_f1):.3f} +- {np.std(cv_svcsgd_f1):.3f}')

