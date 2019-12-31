LDA Tweet Classification
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

In this example, tweets were extracted from the following users:  
demishassabis, goodfellow_ian, JeffDean, karpathy, ch402, iamtrask, trentmc0, gdb, NandoDF, ilyasut, AndrewYNg, MituK, msbernst, jeffbigham, GoogleAI, TensorFlow, IBMResearch, MSFTResearch, srush_nlp, amyxzh, AngelBassa, richardyoung00, sophiebushwick, seanmcarroll, EricTopol, maletsabisam, DimaKrotov, AnnaPaolaCarri, TeddySeyed, IntelAI, datasociety, facebookai, KDziugaite, LRieswijk, FryRsquared, kaifulee, DeepMind_Health, OriolVinyalsML, blaiseaguera, drfeifei, Dominic1King, ShaneLegg, weissg1234, sciam, svlevine, aanakb, tanmingxing, quocleix, hardmaru, PyTorch, an_open_mind, _inesmontani, jsusskin, amuellerml, chelseabfinn, markus_with_k, julien_c, erichorvitz, fchollet, OpenAI, gpapamak, distillpub, stanfordnlp, BaiduResearch, NvidiaAI, StanfordAILab, StanfordHAI, Deep_AI, arxiv_org, red_abebe, rapidsai, NVIDIAAIDev, yaringal, dustinvtran, roydanroy, andrewgwils, santoroAI, DavidDuvenaud, ericjang11, allen_ai, 3blue1brown, diff_eq, MedVocab, jabrils_, TheSpaceGal, poolio, SuryaGanguli, QuantaMagazine, lavanyaai, weights_biases,_beenkim, SimoneGiertz, 

Sources and Helpful Links
---
https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation  
https://towardsdatascience.com/light-on-math-machine-learning-intuitive-guide-to-latent-dirichlet-allocation-437c81220158   
https://towardsdatascience.com/nlp-extracting-the-main-topics-from-your-dataset-using-lda-in-minutes-21486f5aa925  
https://github.com/priya-dwivedi/Deep-Learning/blob/master/topic_modeling/LDA_Newsgroup.ipynb  
[Blei, D. M., Ng, A. Y., Jordan, M. I. (2003). Latent Dirichlet Allocation. Journal of Machine Learning Research, 993-1022](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)  
[Lin, J. (2016). On The Dirichlet Distribution. Queen's University Department of Mathematics and Statistics](https://mast.queensu.ca/~communications/Papers/msc-jiayu-lin.pdf)  

