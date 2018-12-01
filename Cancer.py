
# coding: utf-8

# In[3]:


from sklearn.datasets import load_breast_cancer


# In[4]:


cancer = load_breast_cancer()


# In[5]:


print("cancer.keys(): \n{}".format(cancer.keys()))


# In[6]:


print(cancer.data)


# In[7]:


# print the names of the four features
print(cancer.feature_names)


# In[8]:


print(cancer.DESCR)


# In[10]:


print(cancer.target)


# In[11]:


print(cancer.data)


# In[12]:


print(cancer.target_names)


# In[13]:


print(type(cancer.data))
print(type(cancer.target))


# In[14]:


print(cancer.data.shape)


# In[15]:


print(cancer.target.shape)


# In[16]:


# store feature matrix in "X"
X = cancer.data

# store response vector in "y"
y = cancer.target


# In[20]:


print(X.shape)
print(y.shape)


# In[21]:


from sklearn.neighbors import KNeighborsClassifier


# In[22]:


knn = KNeighborsClassifier(n_neighbors=1)


# In[23]:


print(knn)


# In[24]:


knn.fit(X, y)


# In[26]:


from IPython.display import IFrame


# In[31]:


X_new = [[2.057e+01, 1.777e+01, 1.329e+02, 1.326e+03, 8.474e-02, 7.864e-02,
       8.690e-02, 7.017e-02, 1.812e-01, 5.667e-02, 5.435e-01, 7.339e-01,
       3.398e+00, 7.408e+01, 5.225e-03, 1.308e-02, 1.860e-02, 1.340e-02,
       1.389e-02, 3.532e-03, 2.499e+01, 2.341e+01, 1.588e+02, 1.956e+03,
       1.238e-01, 1.866e-01, 2.416e-01, 1.860e-01, 2.750e-01, 8.902e-02], [1.799e+01,1.228e+02, 1.038e+01, 1.001e+03, 1.184e-01, 2.776e-01,
       3.001e-01, 2.419e-01, 1.471e-01, 7.871e-02, 1.095e+00, 9.053e-01,
       8.589e+00, 1.534e+02, 6.399e-03, 4.904e-02, 5.373e-02, 1.587e-02,
       3.003e-02, 6.193e-03, 2.538e+01, 1.733e+01, 1.846e+02, 2.019e+03,
       1.622e-01, 6.656e-01, 7.119e-01 ,2.654e-01, 4.601e-01, 1.189e-01]]


# In[32]:


knn.predict(X_new)


# In[33]:


# instantiate the model (using the value K=5)
knn = KNeighborsClassifier(n_neighbors=5)

# fit the model with data
knn.fit(X, y)

# predict the response for new observations
knn.predict(X_new)


# In[34]:


# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X, y)

# predict the response for new observations
logreg.predict(X_new)


# In[36]:


from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X, y)

# predict the response values for the observations in X
logreg.predict(X)


# In[37]:


# store the predicted response values
y_pred = logreg.predict(X)

# check how many predictions were generated
len(y_pred)


# In[38]:


from sklearn import metrics
print(metrics.accuracy_score(y, y_pred))


# In[39]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
y_pred = knn.predict(X)
print(metrics.accuracy_score(y, y_pred))


# In[40]:


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)
y_pred = knn.predict(X)
print(metrics.accuracy_score(y, y_pred))


# In[41]:


# print the shapes of X and y
print(X.shape)
print(y.shape)


# In[42]:


# STEP 1: split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)


# In[43]:


# print the shapes of the new X objects
print(X_train.shape)
print(X_test.shape)


# In[44]:


# print the shapes of the new y objects
print(y_train.shape)
print(y_test.shape)


# In[45]:


# STEP 2: train the model on the training set
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[46]:


# STEP 3: make predictions on the testing set
y_pred = logreg.predict(X_test)

# compare actual response values (y_test) with predicted response values (y_pred)
print(metrics.accuracy_score(y_test, y_pred))


# In[47]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


# In[48]:


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


# In[49]:



# try K=1 through K=25 and record testing accuracy
k_range = list(range(1, 26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))


# In[50]:


# import Matplotlib (scientific plotting library)
import matplotlib.pyplot as plt

# allow plots to appear within the notebook
get_ipython().magic(u'matplotlib inline')

# plot the relationship between K and testing accuracy
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')


# In[52]:



# instantiate the model with the best known parameters
knn = KNeighborsClassifier(n_neighbors=11)

# train the model with X and y (not X_train and y_train)
knn.fit(X, y)


