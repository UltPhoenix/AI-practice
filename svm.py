from sklearn.svm import SVC
from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
sonar=pd.read_csv("sonar.csv")
Xs=np.array(sonar.values[:,:-1],'float32')
ys=[]

iris=datasets.load_iris()
Xi=iris['data']
yi=iris['target']
"""
Xi,yi为iris
Xs,ys为sonar
"""
for i in range(Xs.shape[0]):
    if sonar.values[i,-1]=='R':
        ys.append(0)
    else:
        ys.append(1)
sonarnames=['rocks','mine']
ys=np.array(ys,"float32")
svc=SVC(kernel='rbf',class_weight='balanced')
#分割成训练集和测试集
Xtrain,Xtest,ytrain,ytest=train_test_split(Xi,yi,random_state=42)
#寻找最优参数
param_grid={
    'C': [1,5,10,300],
    'gamma':[0.0001,0.0005,0.001,0.1]
}
grid=GridSearchCV(svc,param_grid)
grid.fit(Xtrain,ytrain)
#输出最优参数
print(grid.best_params_)
model=grid.best_estimator_
#训练
yfit=model.predict(Xtest)
#不用报告的注释掉
#iris报告
print(classification_report(ytest,yfit,target_names=iris.target_names))
#sonar报告
#print(classification_report(ytest,yfit,target_names=sonarnames))
#绘制混淆矩阵
def cm_plot(y,yp):#参数为实际分类和预测分类
    cm = confusion_matrix(y,yp)

    #输出为混淆矩阵

    import matplotlib.pyplot as plt
    plt.matshow(cm,cmap=plt.cm.Greens)

    # 画混淆矩阵图，配色风格使用cm.Greens

    plt.colorbar()

    # 颜色标签

    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x,y],xy=(x,y),horizontalalignment='center',verticalalignment='center')

    plt.ylabel('True label')# 坐标轴标签

    plt.xlabel('Predicted label')# 坐标轴标签

    return plt

#函数调用

cm_plot(ytest,yfit).show()
