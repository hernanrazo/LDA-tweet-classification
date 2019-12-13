L
A Tweet Classification
===

Description
---

A python script that uses latent dirichlet allocation to classify the topics in a dataset of tweets. 

Latent dirichlet allocation is a generative statistical model that reveals sets of observations by unsupervised groups that explain why some parts of the data are related.  

Code Explanation
---

The `get_data.py` file is a script that can be used to retrieve tweets from various twitter users. For this specific example, I loaded my twitter API credentials onto a seperate file called `cred.py`. Custom twitter credentials will be needed if you decide to run this code for your purposes. Data is aggregated onto a .csv file. 

The script.py file is where data is cleaned and used to run LDA. the `clean_data()` function cleans raw tweets by removing twitter mentions, URLs, and emojis. This function also sets all letter to lowercase, removes stopwords, and tokenizes and lemmatizes the data.  

To prepare the now cleaned data for lda training, the tweets are gathered into a giant list. This list is then placed into a bag of words model. After that, the dictionary is placed into a corpus that reflects the frequencies of each word. In this example, the lda model is set to 4 topics.



Sources and Helpful Links
---
https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation  
https://towardsdatascience.com/nlp-extracting-the-main-topics-from-your-dataset-using-lda-in-minutes-21486f5aa925  
https://github.com/priya-dwivedi/Deep-Learning/blob/master/topic_modeling/LDA_Newsgroup.ipynb  
