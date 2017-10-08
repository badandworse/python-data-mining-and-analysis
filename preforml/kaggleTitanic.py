'''
问题概述:
分析哪种类型的乘客更有可能活下来
'''
#%%
import numpy as np
import pandas as pd
from pandas import DataFrame,Series

data=pd.read_csv('train.csv')
data.head()
data.info()
data.index
#观看各项属性的基本跟情况
data.describe()


#%%
#画出各数据与survived的关系
import matplotlib.pyplot as plt
import matplotlib


plt.rcParams['font.sans-serif']=['SimHei']   #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False     #用来正常显示正负号


fig=plt.figure()
fig.set(alpha=0.2) #设定图表颜色alpha颜色

#%%
#survived柱状图
plt.subplot2grid((3,5),(0,0))
data.Survived.value_counts().plot(kind='bar')
plt.title(u'获救情况(1为获救)') #标题
plt.ylabel(u'人数')



#Pclass等级柱状图
plt.subplot2grid((3,5),(0,2))
data.Pclass.value_counts().plot(kind="bar")
plt.ylabel(u"人数")
plt.title(u"乘客等级分布")


#Age and survived 图
plt.subplot2grid((3,5),(0,4))
plt.scatter(data.Survived,data.Age)
plt.ylabel(u"年龄")
plt.grid(b=True,which='major',axis='y')
plt.title(u"按年龄看获救分布(1为获救)")

#各等级的乘客年龄分布
plt.subplot2grid((3,5),(2,0),colspan=3)
data.Age[data.Pclass==1].plot(kind='kde')
data.Age[data.Pclass==2].plot(kind='kde')
data.Age[data.Pclass==3].plot(kind='kde')
plt.xlabel(u"年龄")
plt.ylabel(u"密度")
plt.title(u"各等级的乘客年龄分布")
plt.legend((u"头等舱",u"2等舱",u"3等舱"),loc='best') #sets our legend for our graph


#各登船口岸上船人数
plt.subplot2grid((3,5),(2,4))
data.Embarked.value_counts().plot(kind='bar')
plt.title(u"各登船口岸上传人数")
plt.ylabel(u"人数")
plt.show()


#属性与获救结果的关联统计
#%%
fig=plt.figure()
fig.set(alpha=0.2) #设定图标颜色alpha

#舱位级别与获救与否的关系图

Survived_0=data.Pclass[data.Survived==0].value_counts()
Survived_1=data.Pclass[data.Survived==1].value_counts()
df=pd.DataFrame({u'获救':Survived_0,u'未获救':Survived_1})
df.plot(kind='bar',stacked=True)
plt.title(u'各个乘客等级的获救情况')
plt.xlabel(u'乘客等级')
plt.ylabel(u'人数')
plt.show()
df
data.info()

#性别与获救与否的关系图
#%%
Survived_m=data.Survived[data.Sex=='male'].value_counts()
Survived_f=data.Survived[data.Sex=='female'].value_counts()
df=pd.DataFrame({u'男性':Survived_m,u'女侠':Survived_f})
df.plot(kind='bar',stacked=True)
plt.title(u"按性别看获救情况")
plt.xlabel(u"性别")
plt.ylabel(u'人数')
plt.show()

#%%
#各种舱级别情况下个性别的获救情况
# value_count()是按递减排列
fig=plt.figure(figsize=(8,6))
fig.set(alpha=0.65)
plt.title(u'根据舱等级和性别的获奖情况')

ax1=fig.add_subplot(141)
data.Survived[data.Sex=='female'][data.Pclass!=3].value_counts().plot(kind='bar',label='female highclass',color='#FA2479')
ax1.set_xticklabels([u"获救",u"未获救"],rotation=0)
ax1.legend([u"女性/高级舱"],loc='best')


ax2=fig.add_subplot(142,sharey=ax1)
data.Survived[data.Sex=='female'][data.Pclass==3].value_counts().plot(kind='bar',label='female,low class',color='pink')
ax2.set_xticklabels([u"未获救",u"获救"],rotation=0)
ax2.legend([u"女性/低级舱"],loc='best')

ax3=fig.add_subplot(143,sharey=ax1)
data.Survived[data.Sex=='male'][data.Pclass!=3].value_counts().plot(kind='bar',label='male highclass',color='lightblue')
ax3.set_xticklabels([u"未获救",u"获救"],rotation=0)
ax3.legend([u"男性/高级舱"],loc='best')

ax4=fig.add_subplot(144,sharey=ax1)
data.Survived[data.Sex=='male'][data.Pclass==3].value_counts().plot(kind='bar',label='male highclass',color='blue')
ax4.set_xticklabels([u"未获救",u"获救"],rotation=0)
ax4.legend([u"男性/低级舱"],loc='best')

plt.show()

#%%
#登录港口与获救情况之间的关系
data.Embarked.value_counts()
Survived_0=data.Embarked[data.Survived==0].value_counts()
Survived_1=data.Embarked[data.Survived==1].value_counts()
df=pd.DataFrame({u'未获救':Survived_0,u'获救':Survived_1})
df.plot(kind='bar',stacked=True)
plt.title(u"各登录港口乘客的获救情况")
plt.xlabel(u"登录港口")
plt.ylabel(u"人数")
plt.show()

#堂兄弟/妹，孩子/父母有几人，对是否获救的影响。
#%%
g=data_train.groupby(['SibSp','Survived'])
df=pd.DataFrame(g.count()['PassengerId'])
df

#%%
g = data_train.groupby(['Parch','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
df


#Cabin 与获救与否相关情况
# 有效值少，且分散
data.Cabin.value_counts()

# 故下一步选择有无Cabin信息这个粗粒度看看获救与否
fig=plt.figure()
fig.set(alpha=0.2)

#%%
Survived_cabin=data.Survived[pd.notnull(data.Cabin)].value_counts()
Survived_nocabin=data.Survived[pd.isnull(data.Cabin)].value_counts()

df=pd.DataFrame({u"有":Survived_cabin,u"无":Survived_nocabin}).transpose()
df.plot(kind='bar',stacked=True)
plt.title(u"按Cabin有无看获救情况")
plt.xlabel(u"Cabin有无")
plt.ylabel(u"人数")
plt.show()


#第七部分 简单数据预处理

#数据预处理 包含 feature engineering 过程
#特征工程十分重要
#从上面的观察看，cabin和age相关性最为突出，因此先从这两个下手
#cabin就按有无数据处理为Yes和No两种类型
#下一步为age
#缺失值处理:

#先尝试拟合补全age的缺失值
#用scikit-lean中的RandomForest拟合

#%%
data_train=pd.read_csv('train.csv')
from sklearn.ensemble import RandomForestRegressor
#利用随机森林补全缺失值
def set_missing_ages(df):
    
    # 把已有的数值型特征取出来丢尽Random Forest Regressor
    age_df=df[['Age','Fare','Parch','SibSp','Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age=age_df[age_df.Age.notnull()].as_matrix()
    unkownn_age=age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y=known_age[:,0]

    # x即特征属性值
    X=known_age[:,1:]

    # fit到RandomForestRegressor之中
    rfr=RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)
    rfr.fit(X,y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges=rfr.predict(unkownn_age[:,1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[(df.Age.isnull()),'Age']=predictedAges

    return df,rfr

def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()),'Cabin']='Yes'
    df.loc[(df.Cabin.isnull()),'Cabin']='No'
    return df

data_train,rfr=set_missing_ages(data_train)
data_train=set_Cabin_type(data_train)

data_train.info()


#逻辑回归建模时，需要输入的特征是数值型特质
#通常会对类目型的特征因子化
##使用pandas.get_dummies()将拥有不同值的变量转换为0/1数值
#%%
dummies_Cabin=pd.get_dummies(data_train['Cabin'],prefix='Cabin')
dummies_Embarked=pd.get_dummies(data_train['Embarked'],prefix='Embarked')
dummies_Sex=pd.get_dummies(data_train['Sex'],prefix='Sex')
dummies_Pclass=pd.get_dummies(data_train['Pclass'],prefix='Pclass')
df=pd.concat([data_train,dummies_Cabin,dummies_Embarked,dummies_Sex,dummies_Pclass],axis=1)

df.drop(['Pclass','Name','Sex','Ticket','Cabin','Embarked'],axis=1,inplace=True)
df.describe()

#%%
#对数值变化较大的特征，缩小到[-1,1]
#观察处理完的数据，发明Age和Fare两个属性，乘客的数值复读变化很大
#做逻辑回归与梯度下降的话，各属性之间scale差距太大，将会使收敛速度变慢，甚至不收敛
#使用scikit-learn里的preprocessing模块对Age和Fare做一个scaling
#将变化幅度较大的特征化到[-1,1]之内
import sklearn.preprocessing as preprocessing
'''
age_scale_param=scaler.fit(df['Age'])
age_scale_param
df['Age_scaled']=scaler.fit_transform(df['Age'],age_scale_param)
fare_scale_param=scaler.fit(df['Fare'])
df['Fare_scaled']=scaler.fit_transform(df['Age'],fare_scale_param)
df
'''
#新版本里面，scaler对象化的数据必须以行为单位，必须有多行


#%%
scaler=preprocessing.StandardScaler()
X=np.array(df['Age'])

Tage=X.reshape(-1,1)
age_scale_param=scaler.fit(Tage)
mm=scaler.fit_transform(Tage,age_scale_param)
T=mm.reshape(1,-1)
T[0]
df['Age_Scale']=Series(T[0])

T_fare=np.array(df['Fare'])
fare=T_fare.reshape(-1,1)
fare_scale_param=scaler.fit(T_fare.reshape(-1,1))
fare=scaler.fit_transform(fare,fare_scale_param)
fare.shape
T_fare=fare.reshape(891,)
df['Fare_Scale']=Series(T_fare)
df.info()


#%%
#逻辑回归建模
from sklearn import linear_model
# 首先把需要的feature字段提取出来
train_df=df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np=train_df.as_matrix()
# y即Survival结果
y=train_np[:,0]

# x即特征属性值
X=train_np[:,1:]

# fit到LogisticRegression之中
# 然后就得到一个模型
clf=linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-6)
clf.fit(X,y)

#%%
# 将test_data 和 train_data一样的预处理
data_test=pd.read_csv('test1.csv')
data_test.loc[(data_test.Fare.isnull()),'Fare']=0
data_test.info()
#%%
# 接着对测试级做和训练集一致的特征变换
#首先补全age和转换cabin
data_test,rfr_test=set_missing_ages(data_test)
data_test=set_Cabin_type(data_test)

#%%
dummies_Cabin=pd.get_dummies(data_test['Cabin'],prefix='Cabin')
dummies_Embarked=pd.get_dummies(data_test['Embarked'],prefix='Embarked')
dummies_Sex=pd.get_dummies(data_test['Sex'],prefix='Sex')
dummies_Pclass=pd.get_dummies(data_test['Pclass'],prefix='Pclass')

df_test=pd.concat([data_test,dummies_Cabin,dummies_Embarked,dummies_Sex,dummies_Pclass],axis=1)
df_test.drop(['Pclass','Name','Sex','Ticket','Cabin','Embarked'],axis=1,inplace=True)

Test_age=np.array(df_test['Age']).reshape(-1,1)
Test_age=scaler.fit_transform(Test_age,age_scale_param)
Test_age=Test_age.reshape(418,)
df_test['Age_Scale']=Series(Test_age)

#%%
Test_Fare=np.array(df_test['Fare']).reshape(-1,1)
Test_Fare=scaler.fit_transform(Test_Fare,fare_scale_param)
Test_Fare=Test_Fare.reshape(418,)
Test_Fare
df_test['Fare_Scale']=Series(Test_Fare)

#%%
#开始预测结果
test=df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions=clf.predict(test)
predictions

result=pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(),'Survived':predictions.astype(np.int32)})
result.to_csv("logistic_regression_result.csv",index=False)


#9逻辑回归系统优化

#将model系数和 feature关联起来
#系数为正与结果正相关，反之负相关
#%%
#9.1 逻辑系数关联分析
pd.DataFrame({"columns":list(train_df.columns)[1:],'coef':list(clf.coef_.T)})

#9.2交叉验证
#即将训练集分成两部分，一部分用于训练我们需要的模型
#另一部分数据上看我们预测算法的效果

#%%
from sklearn import cross_validation

clf=linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-6)
all_data=df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
X=all_data.as_matrix()[:,1:]
y=all_data.as_matrix()[:,0]
print(cross_validation.cross_val_score(clf,X,y,cv=5))

#%%
#分割数据，按照 训练数据:cv数据=7：3
split_train,split_cv=cross_validation.train_test_split(df,test_size=0.3,random_state=0)
train_df=split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

#生成模型
clf=linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-6)
clf.fit(train_df.as_matrix()[:,1:],train_df.as_matrix()[:,0])

#对cross validation数据进行预测
cv_df=split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions=clf.predict(cv_df.as_matrix()[:,1:])
#%%
origin_data_train=pd.read_csv('train.csv')
bad_case=origin_data_train.loc[origin_data_train['PassengerId'].isin(split_cv[predictions!=cv_df.as_matrix()[:,0]]['PassengerId'].values)]
bad_case.to_csv("bad_case.csv",index=False)

#总结：根据对badcase的观察来决定下一步怎么优化。

#9.3learning curves
#做特征工程的时候，产生的特征越来越多这些特征去训练模型，
#会对我们的训练集拟合得越来越好，同时也可能在逐步丧失泛化能力，
#从而在待预测的数据上，
#表现不佳，也就是发生过拟合问题。

#从另一个角度上说，如果模型在待预测的数据上表现不佳，
#除掉上面说的过拟合问题，也有可能是欠拟合问题，
#也就是说在训练集上，其实拟合的也不是那么好。

#用scikit-learn里面的learning_curve来分别模型的状态

#%%
from sklearn.learning_curve import learning_curve

# 用sklearn的learning_curve得到training_score和cv_score，
def plot_learing_curve(estimator,title,X,y,ylim=None,cv=None,
                        n_jobs=1,train_sizes=np.linspace(.05,1.,20),verbose=0,plot=True):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    train_size,train_scores,test_scores=learning_curve(estimator,X,y,
                                                        cv=cv,n_jobs=n_jobs,train_sizes=train_sizes,verbose=verbose)

    train_scores_mean=np.mean(train_scores,axis=1)
    train_scores_std=np.std(train_scores,axis=1)
    test_scores_mean=np.mean(test_scores,axis=1)    
    test_scores_std=np.std(test_scores,axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"训练样本书数")
        plt.ylabel(u"得分")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes,train_scores_mean-train_scores_std,
                         train_scores_mean+train_scores_std,alpha=0.1,color='b')
        plt.fill_between(train_sizes,test_scores_mean-test_scores_std,
                         test_scores_mean+test_scores_std,alpha=0.1,color='r')
        plt.plot(train_sizes,train_scores_mean,'o-',color='b',label=u"训练集上得分")
        plt.plot(train_sizes,test_scores_mean,'o-',color='r',label=u"交叉训练集上得分")

        plt.legend(loc="best")

        plt.draw()
        plt.show()
        plt.gca().invert_yaxis()
    midpoint=((train_scores_mean[-1]+train_scores_std[-1])+(test_scores_mean[-1]-test_scores_std[-1]))/2
    diff=(train_scores_mean[-1]+train_scores_std[-1])-(test_scores_mean[-1]-test_scores_std[-1])
    return midpoint,diff

plot_learning_curve(clf, u"学习曲线", X, y)


#10模型融合

#%%
from sklearn.ensemble import BaggingRegressor

train_df=df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
train_np=train_df.as_matrix()

# y即Survival结果
y=train_np[:,0]

# X即特征属性值
X=train_np[:,1:]

# fit到BaggingRegressor中
clf=linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-6)
bagging_clf=BaggingRegressor(clf,n_estimators=20,max_samples=0.8,max_features=1.0,bootstrap=True,bootstrap_features=False,n_jobs=-1)
bagging_clf.fit(X,y)

test=df_test.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
predictions=bagging_clf.predict(test)
result=pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(),'Survived':predictions.astype(np.int32)})
result.to_csv('logistic_regression_bagging_predictions2nd.csv',index=False)
