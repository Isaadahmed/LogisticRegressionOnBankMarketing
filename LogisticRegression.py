
from sklearn.linear_model import LogisticRegression 

from sklearn.model_selection import train_test_split 

from sklearn import metrics 

import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 

import seaborn as sns 

bank_df = pd.read_csv("../Datasets/bank.csv") 


# list of variables which needs to be changed
col = ['default','housing','loan','deposit']

# function definition
def convert(x):
    return x.map({'yes':1,'no':0})

# calling the function
bank_df[col] = bank_df[col].apply(convert)

bank_df.head()    
categorical = bank_df.select_dtypes(include=['object'])
categorical.head()
# dummy variables of all categorical columns
dummies = pd.get_dummies(categorical,dtype=int,drop_first=True)
dummies.head()
bank_df = pd.concat([bank_df,dummies],axis=1)
bank_df.drop(columns=categorical.columns,axis=1,inplace=True)


bank_df.head()  
col = "deposit"
df1 = bank_df.loc[:, bank_df.columns != col]
print(df1.head())
X = df1# Features 

y = bank_df.deposit # Target variable 

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0) 
# instantiate the model 

logreg =  LogisticRegression(solver='liblinear') 



# fit the model with data 

logreg.fit(X_train,y_train) 



# predicting 

y_pred=logreg.predict(X_test) 

print(y_pred) 

cnf_matrix = metrics.confusion_matrix(y_test, y_pred) 

cnf_matrix

class_names=[0,1] # name  of classes 

fig, ax = plt.subplots() 

tick_marks = np.arange(len(class_names)) 

plt.xticks(tick_marks, class_names) 

plt.yticks(tick_marks, class_names) 

# create heatmap 

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g') 

ax.xaxis.set_label_position("top") 

plt.tight_layout() 

plt.title('Confusion matrix', y=1.1) 

plt.ylabel('Actual label') 

plt.xlabel('Predicted label')

print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) 

print("Precision:",metrics.precision_score(y_test, y_pred)) 

print("Recall:",metrics.recall_score(y_test, y_pred)) 

plt.show()
