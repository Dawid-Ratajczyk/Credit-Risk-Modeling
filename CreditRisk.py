import pandas as pd
import numpy as np
import plt
from numpy.dual import solve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

data=pd.read_csv('credit_data.csv')#ladowanie danych
X = data.drop('default', axis=1)  # dane bez default czyli Y
y = data[['default']]  # dane y default
X = pd.get_dummies(X, columns=['marital_status', 'property_type'],  drop_first=True)  # zamiana stringów na dummies czyli inty
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,random_state=123)  # podzial danych na test i train

def ModelTestDiffProb(model,x_t,y_t,name,fin_test_prob):#testowanie modelu zaleznie od prob
    print("VVV")
    print("Model name: ", name)
    probabilities=np.arange(0.95, 0, -0.05).round(2)#lista prob od 0 do 1
    precision_list=[]
    recall_list=[]
    f1_list=[]
    for prob in probabilities: #dla kazdego prob
        y_probs=model.predict_proba(x_t)[:, 1] #obliczamy prawdopodobienstwo przewidziane przez model
        y_pred=(y_probs > prob).astype(int) #porównujemy te prawdopodobienstwo z naszym progiem prob
        precision=precision_score(y_t, y_pred)#olbiczamy precyzje
        recall=recall_score(y_t, y_pred)#obliczamy recall
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1_score(y_t,y_pred))
    plt.plot(probabilities, precision_list, marker='o', label='Precyzja')
    plt.plot(probabilities, recall_list, marker='o', label='Recall')
    plt.plot(probabilities, f1_list, marker='o', label='F1')
    plt.title(name)
    plt.xlabel('Prob')
    plt.ylabel('X')
    plt.legend()
    plt.show()
    #dla finalnego prob
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs > fin_test_prob).astype(int)
    print("Score:")
    print(model.score(X_test, y_test))
    print("Macierz pomyłek:")
    print(confusion_matrix(y_test, y_pred))
    print("Raport:")
    print(classification_report(y_test, y_pred))
    scores = cross_val_score(model, X, np.ravel(y), cv=5, scoring='accuracy')
    print(f'Dokładność śr: {np.mean(scores)}')
    print(f'Odchyl stand: {np.std(scores)}')

def LogReg():
    lr=LogisticRegression(max_iter=1000,class_weight={0: 1, 1: 4})#definicja regresji logistycznej, max iter jak dlugo robimy, class weight rozna waga klas
    lr.fit(X_train,np.ravel(y_train))#dopasowanie parametrów modelu do danych
    ModelTestDiffProb(lr,X_test,y_test,"Log Reg",0.7)
    print("Intercept")
    print(lr.intercept_)
    print('Coefficients:')
    print(lr.coef_)

def DecTre():
    tree = DecisionTreeClassifier(class_weight={0: 1, 1: 4}, random_state=123,max_depth=5, min_samples_leaf=10)
    tree.fit(X_train, y_train)
    ModelTestDiffProb(tree,X_test,y_test,"Dec Tree",0.7)

def XGB():
    boost = XGBClassifier(random_state=123)
    boost.fit(X_train, y_train)
    y_pred = boost.predict(X_test)
    ModelTestDiffProb(boost,X_test,y_test,"XGBoost",0.75)

if __name__ == '__main__':
   LogReg()
   DecTre()
   XGB()
