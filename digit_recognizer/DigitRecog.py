#!/usr/bin/env python
# coding: utf-8

# In[407]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.stats import multivariate_normal as mvn
from scipy.stats import multinomial as mnd
from sklearn.naive_bayes import MultinomialNB
from scipy.stats import bernoulli as ber
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


get_ipython().run_line_magic('matplotlib', 'inline')


# In[380]:


uspsdata=pd.read_csv("C:/Users/Aashish/techfield/week3_4/week4hw/USPS_digit/usps_digit_recognizer.csv")


# In[381]:


#uspsdata.head(1)


# In[382]:


# changing the columns  to put data in X,y format 
labl = uspsdata.pop('label')    # save the first column in "labl" 
uspsdata['label'] = labl        # adds a column label to the dataframe with the values saved in "labl"


# In[383]:


#uspsdata.head(1)


# In[384]:


#dataframe into numpy array
data=uspsdata.to_numpy()


# In[385]:


#data.shape


# In[386]:


X = data[:,:-1]
y = data[:,-1]


# In[431]:


X.shape
#y.shape


# In[388]:


#check the distribution of digit label data
dist = uspsdata['label'].value_counts().sort_index()
dist


# In[389]:


#plot the distribution of digit label data
dist.plot(kind='bar')
plt.tick_params(axis='y', labelsize=17)
plt.tick_params(axis='x',rotation=0, labelsize=25)
plt.xlabel("Digit Labels", size=25)
plt.ylabel("Frequency", size=25)

plt.show()


# In[390]:


#normalizing X
X = X/255


# In[391]:


#reshape the flat pixels into a matrix of size 28x28 to show it
digit = X[999]
digit_reshaped = digit.reshape([28, 28]) #since it is a 28x28 size pic we resize it accordingly
plt.imshow(digit_reshaped)


# In[445]:


digit = X[169]

digit_reshaped = digit.reshape([28, 28]) #since it is a 28x28 size pic we resize it accordingly
plt.imshow(digit_reshaped, cmap='gray')
plt.show()


# In[392]:


# prepare cross validation
kfold = KFold(5, True, 1)
# enumerate splits
for train_index, test_index in kfold.split(X):
    X_train, X_test = X[train_index], X[test_index]
for train_index, test_index in kfold.split(y):
    y_train, y_test = y[train_index], y[test_index]
    


# In[393]:


#X_train.shape
#X_test.shape
#y_train.shape
#y_test.shape


# In[394]:


##########Class def Below#########################


# In[395]:


#Weclass defination for Gaussian Bayes Classification:
class GaussianBayes():
    def fit(self, X, y, epsilon = 1e-3):
        self.likelihoods = dict()
        self.priors = dict()
        
        self.K = set(y.astype(int))
        
        for k in self.K:
            X_k = X[y == k,:]
            mu_k = X_k.mean(axis=0)
            N_k, D = X_k.shape
            
            self.likelihoods[k] = {"mean": mu_k, "cov":(1/(N_k -1))*np.matmul((X_k).T, X_k - mu_k) + epsilon*np.identity(D) }  #made changes in mean and cov
            self.priors[k] = len(X_k)/len(X)
            
    def predict(self, X):
        N, D = X.shape
        
        P_hat = np.zeros((N,len(self.K)))
        
        for k, l in self.likelihoods.items():
            P_hat[:,k] = mvn.logpdf(X, l["mean"], l["cov"]) + np.log(self.priors[k])
            
        return P_hat.argmax(axis = 1)


# In[396]:


gb = GaussianBayes()


# In[397]:


gb.fit(X_train, y_train)
y_hat = gb.predict(X_test)
    


# In[398]:


def accuracy(y, y_hat):
    return np.mean(y==y_hat)


# In[399]:


print(f"accuracy:{accuracy(y_test, y_hat):0.4f}")


# In[400]:


print(classification_report(y_test, y_hat, labels=range(0,10)))


# In[410]:


print(classification_report(y_test, y_hat, labels=range(0,10)))


# In[411]:


dataclassificationreport =classification_report(y_test, y_hat, labels=range(0,10))


# In[464]:


cm = confusion_matrix(y_test, y_hat)
cm


# In[465]:


df_confusion = pd.crosstab(y_test, y_hat)
print (df_confusion)


# In[469]:


b=y_test.shape
b


# In[468]:


a=sum(np.diagonal(df_confusion))
a


# In[462]:


a/b*100


# In[429]:


#df_confusion.to_html('your_output_file_name.html')


# In[430]:


#df_confusion.to_csv('your_output_file_name.csv')


# In[ ]:





# In[ ]:





# In[200]:


#####################################################################################
#####################################################################################
#####################################################################################


# In[ ]:





# In[29]:


# DataFrame column names as a list
#colnamelist = list(uspsdata.columns)


# In[54]:


# brings the first column in the last place
#newcollist= colnamelist[1:] + colnamelist[:1]
#newcollist


# In[75]:


#data in the format (X,y) where X are features and y is class 
#digitdata = uspsdata[newcollist]
#digitdata.head(1)


# In[72]:


#seperating the data into X and y
#data=digitdata.to_numpy()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




