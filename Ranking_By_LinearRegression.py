from sklearn.linear_model import Lasso
import pandas as pd
from sklearn import model_selection, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import sklearn.metrics
import matplotlib.pyplot as plt
import numpy as np

#filepath = 'D:/00_SFU/00_Graduate_Courses/00_CMPT741_DataMining/Project/2019-741_Data/training_data_preprocessed.csv'
#df = pd.read_csv(filepath,sep=',')

dataset = np.loadtxt("D:/00_SFU/00_Graduate_Courses/00_CMPT741_DataMining/Project/2019-741_Data/training_data_preprocessed.csv", delimiter=",")
y = dataset[:,:1]
qid = dataset[:,1]
X = dataset[:,2:]
print("\nDataset Dimensions : ",dataset.shape)

# LASSO SETUP
lasso = Lasso (alpha = 0.215,normalize=True)
lasso_coef = lasso.fit(X,y).coef_
lasso_coef_positive = lasso_coef[lasso_coef > 0]
plt.plot(range(len(lasso_coef_positive)),lasso_coef_positive)
plt.xticks(range(len(lasso_coef_positive)),range(0,58),rotation=60)
plt.ylabel('coefficients')
plt.show()

features_selected = np.where(np.array(lasso_coef) > 0)[0]
print("Features Selected [%d]:" %len(features_selected),features_selected)
X = X[:,features_selected]
print(X.shape)

#Linear Regression
# Split-out validation dataset
y = y.reshape(-1,1)
#X = df.drop(['Relevance'],axis=1)
validation_size = 0.2
seed = 7

scoring = 'r2'
X_train, X_validation, y_train, y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)

#Instaniate Linear Regression model
model = linear_model.LinearRegression()

#Cross-Validation
#kfold = model_selection.KFold(n_splits=10, random_state=seed)
#cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
#print(cv_results)
print("Actual Training Data Score\n-------------------------------------------\n")
for i in range(0,5):
    print(i+1,"\t", y_train[i])
model.fit(X_train,y_train)
y_pred = model.predict(X_train)
print("Training Data\n-------------------------------------------\nMSE - ",mean_squared_error(y_train, y_pred))
print(" R2 - ",r2_score(y_train, y_pred))
print("Predicted Training Data Score\n-------------------------------------------\n")
for i in range(0,5):
    print(i+1,"\t", y_pred[i])
#plt.plot(y_train,y_pred,marker='.',alpha=0.3,linestyle='none')
#plt.xlabel('Actual')
#plt.ylabel('Predicted')
#plt.show()

print("Actual Validation Data Score\n-------------------------------------------\n")
for i in range(0,5):
    print(i+1,"\t", y_validation[i])
y_pred = model.predict(X_validation)
print("\nValidation Data\n-------------------------------------------\nMSE - ",mean_squared_error(y_validation, y_pred))
print(" R2 - ",r2_score(y_validation, y_pred))
print("Predicted Validation Data Score\n-------------------------------------------\n")
for i in range(0,5):
    print(i+1,"\t", y_pred[i])
#plt.plot(y_validation,y_pred,marker='.',alpha=0.3,)
#plt.xlabel('Actual')
#plt.ylabel('Predicted')
#plt.show()

#print((list(zip(y_validation,y_pred)))[:])
#ratio = (y_pred/y_validation)
#print(ratio)

# Using the model built over test data
dataset = np.loadtxt("D:/00_SFU/00_Graduate_Courses/00_CMPT741_DataMining/Project/2019-741_Data/example_testing_data.csv", delimiter=",")
y = dataset[:,:1]
qid = dataset[:,1]
X = dataset[:,2:]
X = X[:,features_selected]
print(X.shape)

y = y.reshape(-1,1)
y_pred = model.predict(X)
print("Predicted Test Data Score\n-------------------------------------------\n")
for i in range(0,5):
    print(i+1,"\t", y_pred[i])

print("\n\n")
temp=[]
index=[]
old_qid = -1
count = 0
for i in range(0,len(y_pred)):
    temp.append(y_pred[i][0])
    if old_qid == -1 or old_qid != qid[i]:
        old_qid = qid[i]
        count=0
    index.append(count)
    count+=1
df = pd.DataFrame({'temp':temp,'qid':qid,'index':index})
df = df.sort_values('temp',ascending=False)
#df = df.groupby('qid').groups
grouped = (df.groupby('qid'))
for name,group in grouped:
    print (name)
    print (group)
    
print(sorted(sklearn.metrics.SCORERS.keys()))