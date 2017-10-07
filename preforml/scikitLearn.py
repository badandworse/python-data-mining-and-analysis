from sklearn import datasets

iris=datasets.load_iris()
digits=datasets.load_digits()

print(digits.data)
digits.data.shape

digits.target.shape

#%%
import matplotlib.pyplot as plt

# 输出四张数字码(0,1,2,3)的8*8点阵图
# 点阵图的数据从datasets读取并存在digits中
# 我们可以通过matplotlib所提供的方法显示这些点阵图

image_and_labels=list(zip(digits.images,digits.target))
for index,(image,label) in enumerate(image_and_labels[:4]):
    plt.subplot(2,4,index+1)
    plt.axis('off')
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    plt.title('Training: %i' %label)


#选择分类器并进行设置、训练和预测
#%%
from sklearn import datasets,svm
# 读取数据
digits=datasets.load_digits()
# 建立SVM分类器
clf=svm.SVC(gamma=0.001,C=100.)
# 使用训练数据对分类器进行训练，它将会返回分类器的某些参数设置
clf.fit(digits.data[:-1],digits.target[:-1])

#%%
test = [0, 0, 10, 14, 8, 1, 0, 0,
        0, 2, 16, 14, 6, 1, 0, 0,
        0, 0, 15, 15, 8, 15, 0, 0,
        0, 0, 5, 16, 16, 10, 0, 0,
        0, 0, 12, 15, 15, 12, 0, 0,
        0, 4, 16, 6, 4, 16, 6, 0,
        0, 8, 16, 10, 8, 16, 8, 0,
        0, 1, 8, 12, 14, 12, 1, 0]

#%%
#分类器效果评估
from sklearn import metrics,datasets,svm

digits=datasets.load_digits()

clf=svm.SVC(gamma=0.001,C=100.)

#选取数据集中前500条数据作为训练集
clf.fit(digits.data[:500],digits.target[:500])

# 选取数据集中后1000条数据作为测试数据
expected=digits.target[800:]
predicted=clf.predict(digits.data[800:])
print('分类器预测结果评估:\n%s\n' %(metrics.classification_report(expected,predicted)))






