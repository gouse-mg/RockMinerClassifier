import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV


data=pd.read_csv("Copy of sonar data.csv")

column_names = [f"Column_{i+1}" for i in range(61)]
data.columns=column_names
y=data['Column_61']
y=np.array(y)
cols=[x for x in column_names[:60]]
X=data[cols]
X=np.array(X)
y=y.reshape(-1,1)
y=np.where(y=='R' ,1,0)




skfolds=StratifiedKFold(n_splits=10)
l_r=LogisticRegression(random_state=42)
best_eval=0
best_model=None
for train_idx,test_idx in skfolds.split(X,y):
    X_train=X[train_idx]
    y_train=y[train_idx]
    X_test=X[test_idx]
    y_test=y[test_idx]
    model=clone(l_r)
    model.fit(X_train,y_train)
    y_predict=model.predict(X_test)
    eval_score=accuracy_score(y_true=y_test,y_pred=y_predict)
    # print(eval_score)
    if eval_score>best_eval:

        best_model=model
    
        best_eval=eval_score



model=best_model

input_data = (
   0.0116, 0.0179, 0.0449, 0.1096, 0.1913, 0.0924, 0.0761, 0.1092, 0.0757, 0.1006, 0.25, 0.3988, 0.3809, 0.4753, 0.6165, 0.6464, 0.8024, 0.9208, 0.9832, 0.9634, 0.8646, 0.8325, 0.8276, 0.8007, 0.6102, 0.4853, 0.4355, 0.4307, 0.4399, 0.3833, 0.3032, 0.3035, 0.3197, 0.2292, 0.2131, 0.2347, 0.3201, 0.4455, 0.3655, 0.2715, 0.1747, 0.1781, 0.2199, 0.1056, 0.0573, 0.0307, 0.0237, 0.047, 0.0102, 0.0057, 0.0031, 0.0163, 0.0099, 0.0084, 0.027, 0.0277, 0.0097, 0.0054, 0.0148, 0.0092


)


input_data = np.array(input_data).reshape(1, -1)


prediction = best_model.predict(input_data)


if prediction == 1:
    print("The object is predicted to be 'Rock \U0001F600'.")
else:
    print("The object is predicted to be 'Mine \U0001F600'.")



