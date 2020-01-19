import pickle
import warnings
import csv
import numpy as np
import pandas as pd
import gensim
import nltk
from nltk.corpus import stopwords
from sklearn import linear_model
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from mlxtend.evaluate import mcnemar
from mlxtend.evaluate import mcnemar_table
warnings.filterwarnings("ignore")


#set stopwords
stop_words = nltk.corpus.stopwords.words('english')
custom = ['?','(', ')', '.', '[', ']','!', '...', '-', '@', '->','https',
        ';', "`", "'", '"',',', ':', '*', '~' , '/', '//', '\\', '&', 'n', ':\\']
stop_words.extend(custom)


def csv_to_words(raw_data):
    for row in raw_data:
        yield(gensim.utils.simple_preprocess(str(row), deacc=True))


#remove stopwords
def remove_stopwords(processed_data):
    return [[words for words in gensim.utils.simple_preprocess(str(doc)) if not words in stop_words] for doc in processed_data]


#convert words into bigrams
def get_bigram(words, bi_min=15, tri_min=10):
    bigram = gensim.models.Phrases(words, min_count=bi_min)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return bigram_mod


def main():
    #open needed files
    test_data = pd.read_csv('data/test_tweet_data.csv', encoding='ISO-8859-1')
    train_data = pd.read_csv('data/tweet_data_copy.csv', encoding='ISO-8859-1')
    train_bigram = pd.read_pickle('saved_pickles_models/train_bigram.pkl')
    train_id2word = pd.read_pickle('saved_pickles_models/train_id2word.pkl')
    train_corpus = pd.read_pickle('saved_pickles_models/train_corpus.pkl')
    model = pd.read_pickle('saved_pickles_models/lda_model.model')
    feature_vectors = []
    test_vectors = []
    scaler = StandardScaler()


    #get distributions from every tweet in train_data
    print('Getting distribution...')
    for i in range(len(train_data)):
        train_top_topics = model.get_document_topics(train_corpus[i], minimum_probability=0.0)
        train_topic_vector = [train_top_topics[i][1] for i in range(15)]
        feature_vectors.append(train_topic_vector)

    x = np.array(feature_vectors)
    y = np.array(train_data.relevant)

    kf = KFold(5, shuffle=True, random_state=42)
    log_res_train_f1, log_res_sgd_train_f1, mod_huber_train_f1 = [], [], []


    print('Starting classification algorithm calculations on training data...')
    for train_ind, val_ind in kf.split(x, y):
        x_train, y_train = x[train_ind], y[train_ind]
        x_val, y_val = x[val_ind], y[val_ind]

        x_train_scale = scaler.fit_transform(x_train)
        x_val_scale = scaler.transform(x_val)

        #logistic regression
        log_reg_train = LogisticRegression(class_weight='balanced',
                                     solver='newton-cg',
                                     fit_intercept=True).fit(x_train_scale, y_train)
        log_reg_train_y_pred = log_reg_train.predict(x_val_scale)
        log_res_train_f1.append(f1_score(y_val, log_reg_train_y_pred, average='binary'))

        #loss=log
        sgd = linear_model.SGDClassifier(max_iter=1000,
                                        tol=1e-3, 
                                        loss='log',
                                        class_weight='balanced').fit(x_train_scale, y_train)
        sgd_y_pred = sgd.predict(x_val_scale)
        log_res_sgd_train_f1.append(f1_score(y_val, sgd_y_pred, average='binary'))

        #modified huber
        sgd_huber = linear_model.SGDClassifier(max_iter=1000,
                                               tol=1e-3, alpha=20,
                                               loss='modified_huber',
                                               class_weight='balanced').fit(x_train_scale, y_train)

        sgd_huber_y_pred = sgd_huber.predict(x_val_scale)
        mod_huber_train_f1.append(f1_score(y_val, sgd_huber_y_pred, average='binary'))

    print('Done with training data. Starting on testing data...\n')


    #gather all test tweets and apply the clean_data() and get_bigram() functions
    print('Cleaning testing data...')
    processed_tweets = list(csv_to_words(test_data.tweet))
    cleaned_tweets = remove_stopwords(processed_tweets)
    bigrams= get_bigram(cleaned_tweets)
    test_bigram = [bigrams[tweet] for tweet in cleaned_tweets]
    test_corpus = [train_id2word.doc2bow(tweet) for tweet in test_bigram]

    #test model on testing data
    print('Starting classification algorithm calculations on testing data...')
    for i in range(len((test_data.tweet))):
        top_topics = model.get_document_topics(test_corpus[i], minimum_probability=0.0)
        topic_vector = [top_topics[i][1] for i in range(15)]
        test_vectors.append(topic_vector)

    x_test = np.array(test_vectors)
    y_test = np.array(test_data.relevant)
    x_fit = scaler.fit_transform(x_test)

    #logistic regression
    log_reg_test = LogisticRegression(class_weight='balanced',
                                      solver='newton-cg',
                                      fit_intercept=True).fit(x_fit, y_test)
    y_pred_log_res_test = log_reg_test.predict(x_test)

    #modified huber
    sgd_huber_test = linear_model.SGDClassifier(max_iter=1000,
                                           tol=1e-3,
                                           alpha=20,
                                           loss='modified_huber',
                                           class_weight='balanced',shuffle=True).fit(x_fit, y_test)
    y_pred_huber_test = sgd_huber_test.predict(x_fit)

    #print results for both cases
    print('Calculating Summary...')
    y_target = y_test
    y_model1 = y_pred_log_res_test
    y_model2 = y_pred_huber_test

    m_table = mcnemar_table(y_target=y_test,
                            y_model1=y_model1,
                            y_model2=y_model2)

    chi2, p = mcnemar(ary=m_table, corrected=True)


    print('\n')
    print('Results from using training data distribution:')
    print(f'Logistic Regression Val f1: {np.mean(log_res_train_f1):.3f} +- {np.std(log_res_train_f1):.3f}')
    print(f'Logisitic Regression SGD Val f1: {np.mean(log_res_sgd_train_f1):.3f} +- {np.std(log_res_sgd_train_f1):.3f}')
    print(f'SVM Huber Val f1: {np.mean(mod_huber_train_f1):.3f} +- {np.std(mod_huber_train_f1):.3f}')

    print('\n')
    print('Results from using unseen test data:')
    print('Logistic regression Val f1: ' + str(f1_score(y_test, y_pred_log_res_test, average='binary')))
    print('Logistic regression SGD f1: ' + str(f1_score(y_test, y_pred_huber_test, average='binary')))

    print('\n')
    print('Summary: ')
    print('ncmamor table: ', m_table)
    print('chi-squared: ',  chi2)
    print('p-value: ', p)


    #Save feature vector and huber classifier for later use
    print('\n')
    print('Saving feature vector...')
    save_vector = open('saved_pickles_models/feature_vector.pkl', 'wb')
    pickle.dump(feature_vectors, save_vector)
    save_vector.close()

    print('\n')
    print('Saving the huber classifier...')
    save_huber = open('saved_pickles_models/huber_classifier.pkl', 'wb')
    pickle.dump(sgd_huber, save_huber)
    save_huber.close()
    print('done')

if __name__ == "__main__":
    main()
