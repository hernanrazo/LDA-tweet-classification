LDA Tweet Classification
===

Description
---

A python script that uses latent dirichlet allocation to classify the topics in a dataset of tweets. 

the Dirichlet Distribution is a multivariate continuous distribution in probability theory. It is a multivariate generalization of tne Beta distribution and a popular conjugate prior to the multinomial distribution.  

Latent Dirichlet Allocation (LDA) is a generative statistical model that reveals sets of observations by unsupervised groups that explain why some parts of the data are related. LDA imagines a fixed number of topics that represent a set of words. The goal is to map all documents (tweets in ths case) to one of those topics such that the words in each document are mostly covered by that topic.  

The `train_model.py` file's code in this repo was adapted from [this repo by Marc Kelechava](https://github.com/marcmuon/nlp_yelp_review_unsupervised). [Marc's Medium article](https://towardsdatascience.com/unsupervised-nlp-topic-models-as-a-supervised-learning-input-cf8ee9e5cf28) also goes into a bit more detail. The original authors of this method where the LDA distribution is used as a feature vector for another classification algorithm are Xuan-Hieu Phan, Le-Minh Nguyen, and Susumu Horiguchi. Their paper can be found [here](http://gibbslda.sourceforge.net/fp224-phan.pdf).  

Code Explanation
---

The `get_data.py` file is a script that can be used to create a dataset by retrieve tweets from various twitter users. For this specific example, I loaded my twitter API credentials onto a seperate file called `cred.py`. Custom twitter credentials will be needed if you decide to run this code for your purposes. Data is aggregated onto a .csv file. For later classification, I added a `relevant` column that was either a 1 or 0 depending if the tweet was relevant to the topics I wanted to classify for. Irrelevant tweets usually were short replies between users.  

In my implimitation, the `script.py` file cleans the data and runs LDA. The `csv_to_words()` function grabs each row in the csv file and parses the raw data into a usable format. The `remove_stopwords()` function removes stopwords. The `get_bigrams()` function converts the cleaned data into bigrams where commonly paired words are put together and treated as one.  

To prepare the now cleaned data for LDA training, the tweets are gathered into a list of lists and converted into a dictionary of word frequencies for each word and bigram.  

In this example, tweets were extracted from the following users:  
demishassabis, goodfellow_ian, JeffDean, karpathy, ch402, iamtrask, trentmc0, gdb, NandoDF, ilyasut, AndrewYNg, MituK, msbernst, jeffbigham, GoogleAI, TensorFlow, IBMResearch, MSFTResearch, srush_nlp, amyxzh, AngelBassa, richardyoung00, sophiebushwick, seanmcarroll, EricTopol, maletsabisam, DimaKrotov, AnnaPaolaCarri, TeddySeyed, IntelAI, datasociety, facebookai, KDziugaite, LRieswijk, FryRsquared, kaifulee, DeepMind_Health, OriolVinyalsML, blaiseaguera, drfeifei, Dominic1King, ShaneLegg, weissg1234, sciam, svlevine, aanakb, tanmingxing, quocleix, hardmaru, PyTorch, an_open_mind,\_inesmontani, jsusskin, amuellerml, chelseabfinn, markus_with_k, julien_c, erichorvitz, fchollet, OpenAI, gpapamak, distillpub, stanfordnlp, BaiduResearch, NvidiaAI, StanfordAILab, StanfordHAI, Deep_AI, arxiv_org, red_abebe, rapidsai, NVIDIAAIDev, yaringal, dustinvtran, roydanroy, andrewgwils, santoroAI, DavidDuvenaud, ericjang11, allen_ai, 3blue1brown, diff_eq, MedVocab, jabrils_, TheSpaceGal, poolio, SuryaGanguli, QuantaMagazine, lavanyaai, weights_biases, _beenkim, SimoneGiertz   

Sources and Helpful Links
---
https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation  
https://towardsdatascience.com/light-on-math-machine-learning-intuitive-guide-to-latent-dirichlet-allocation-437c81220158  
https://towardsdatascience.com/nlp-extracting-the-main-topics-from-your-dataset-using-lda-in-minutes-21486f5aa925  
https://github.com/marcmuon/nlp_yelp_review_unsupervised  
[Blei, D. M., Ng, A. Y., Jordan, M. I. (2003). Latent Dirichlet Allocation. Journal of Machine Learning Research, 993-1022](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)  
[Lin, J. (2016). On The Dirichlet Distribution. Queen's University Department of Mathematics and Statistics](https://mast.queensu.ca/~communications/Papers/msc-jiayu-lin.pdf)  
