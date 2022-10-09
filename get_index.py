from sklearn import metrics
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
y_pred = np.zeros((284),dtype= np.int32)
for idx in range(184,284):
    y_pred[idx]=1
print(y_pred)
y_true = np.zeros((284),dtype= np.int32)
for idx in range(184):
    y_true[idx]=1
print(y_true)
# 设置路径
root_path = './'
total_label_path = root_path + 'txt/'  # txt存储的路径
# 逐个读取txt标注文件
i=0
for filename in os.listdir(total_label_path):
    fileNameID = os.path.splitext(filename)[0]
    # print(int(fileNameID))
    cur_label_path = total_label_path + filename
    cur_boxes = []
    # 读取当前txt文件中的内容
    with open(cur_label_path, 'r') as file:
        while True:
            line = file.readline().strip()  # .strip()用来去掉'\r,\n'
            if not line:
                break
            line_list = [ele for ele in line.split(' ')]
            cur_boxes.append(line_list)
    for box in cur_boxes:
         if str(box[0]) =='malignancy' and int(fileNameID) <918:
            y_pred[i]=1
         elif str(box[0]) =='benign' and int(fileNameID) >917:
            y_pred[i]=0
         else:
            continue
    i=i+1
print(y_pred)
acc = accuracy_score(y_true, y_pred)
print('acc is {:.2%}'.format(acc))
#metrics.precision_score(y_true, y_pred, average='micro')  # 微平均，精确率
prec = metrics.precision_score(y_true, y_pred)  # 宏平均，精确率
print('precision is {:.2%}'.format(prec))
#metrics.precision_score(y_true, y_pred, labels=[0, 1, 2, 3], average='macro')  # 指定特定分类标签的精确率
#metrics.recall_score(y_true, y_pred, average='micro')
recall = metrics.recall_score(y_true, y_pred)
print('recall is {:.2%}'.format(recall))
f1 = metrics.f1_score(y_true, y_pred)
print('f1 is {:.2%}'.format(f1))
# 混淆矩阵
C=confusion_matrix(y_true, y_pred)
# 分类报告：precision/recall/fi-score/均值/分类个数
target_names = ['class 0', 'class 1']
print(classification_report(y_true, y_pred, target_names=target_names))
plt.matshow(C, cmap=plt.cm.Reds) # 根据最下面的图按自己需求更改颜色
# plt.colorbar()

for i in range(len(C)):
    for j in range(len(C)):
        plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')

# plt.tick_params(labelsize=15) # 设置左边和上面的label类别如0,1,2,3,4的字体大小。

plt.ylabel('True label')
plt.xlabel('Predicted label')
# plt.ylabel('True label', fontdict={'family': 'Times New Roman', 'size': 20}) # 设置字体大小。
# plt.xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'size': 20})
# plt.xticks(range(0,5), labels=['a','b','c','d','e']) # 将x轴或y轴坐标，刻度 替换为文字/字符
# plt.yticks(range(0,5), labels=['a','b','c','d','e'])
plt.show()