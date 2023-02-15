import numpy as np
from scipy import stats
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix    # 生成混淆矩阵函数
import matplotlib.pyplot as plt    # 绘图库
import os

tf.compat.v1.disable_eager_execution()
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 读取数据集
df = pd.read_csv("dataset\\tt.csv")
# 数据归一化
df['open'] = minmax_scale(df['open'])
df['high'] = minmax_scale(df['high'])
df['low'] = minmax_scale(df['low'])

# 构建向量矩阵
# 总数据矩阵/标签
data = []
label = []


# 定义窗口函数
def windows(data, size):
    start = 0
    while start < data.count():
        yield int(start), int(start + size)
        start += (size / 2)


# 返回格式数据
def segment_signal(data, window_size=90):
    segments = np.empty((0, window_size, 3))
    labels = np.empty((0))
    for (start, end) in windows(data["timestamp"], window_size):
        x = data["open"][start:end]
        y = data["high"][start:end]
        z = data["low"][start:end]
        if (len(df["timestamp"][start:end]) == window_size):
            segments = np.vstack([segments, np.dstack([x, y, z])])
            labels = np.append(labels, stats.mode(data["label"][start:end])[0][0])
    return segments, labels


data, label = segment_signal(df)
# 对标签数据进行处理
for i in range(0, len(label)):
    if label[i] == -1:
        label[i] = 0

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2)
X_train = np.array(X_train).reshape(len(X_train), 90, 3)
X_test = np.array(X_test).reshape(len(X_test), 90, 3)
y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)

# 标签One-Hot
enc = OneHotEncoder()
enc.fit(y_train)
y_train = enc.transform(y_train).toarray()
y_test = enc.transform(y_test).toarray()

in_channels = 3
units = 256
epoch = 10000
batch_size = 5
batch = X_train.shape[0] / batch_size

# 创建占位符
X = tf.placeholder(tf.float32, shape=(None, 90, in_channels))
Y = tf.placeholder(tf.float32, shape=(None, 2))

# layer1
h1 = tf.layers.conv1d(X, 256, 4, 2, 'SAME', name='h1', use_bias=True, activation=tf.nn.relu)
p1 = tf.layers.max_pooling1d(h1, 2, 2, padding='VALID')
print(h1)
print(p1)

# layer2
h2 = tf.layers.conv1d(p1, 256, 4, 2, 'SAME', use_bias=True, activation=tf.nn.relu)
p2 = tf.layers.max_pooling1d(h2, 2, 2, padding='VALID')
print(h2)
print(p2)

# layer3
h3 = tf.layers.conv1d(p1, 2, 4, 2, 'SAME', use_bias=True, activation=tf.nn.relu)
p3 = tf.layers.max_pooling1d(h3, 11, 1, padding='VALID')
res = tf.reshape(p3, shape=(-1, 2))

print(h3)
print(p3)
print(res)

# loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=res, labels=Y))

# 创建正确率
ac = tf.cast(tf.equal(tf.argmax(res, 1), tf.argmax(Y, 1)), tf.float32)
acc = tf.reduce_mean(ac)

# 创建优化器
optim = tf.train.AdamOptimizer(0.0001).minimize(loss)
f=open('result\\result.txt','w')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(30000):#10000
        sess.run(optim, feed_dict={X: X_train, Y: y_train})
        if i % 100 == 0:
            los, accuracy = sess.run([loss, acc], feed_dict={X: X_train, Y: y_train})
            print(los, accuracy)
    ccc, bbb = sess.run([tf.argmax(res, 1), tf.argmax(Y, 1)], feed_dict={X: X_test, Y: y_test})
    print(len(ccc))
    for i in range(0, len(ccc)):
        #print(ccc[i])
        #print(bbb[i])
        f.write(str(ccc[i]) + " " + str(bbb[i]) + "\n")
f.close()

f=open('result\\result.txt','r')
pre=[]
t=[]
for row in f.readlines():
    row=row.strip() #去掉每行头尾空白
    row=row.split(" ")
    pre.append((row[0]))
    t.append((row[1]))
f.close()
#混淆矩阵绘制
def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype(np.float64)
    if(cm.sum(axis=0)[0]!=0):
        cm[:,0] = cm[:,0] / cm.sum(axis=0)[0]   # 归一化
    if(cm.sum(axis=0)[1]!=0):
        cm[:,1] = cm[:,1] / cm.sum(axis=0)[1]   # 归一化
    plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
cm=confusion_matrix(t,pre)      #生成混淆矩阵
y_true = np.array(list(map(int,t)))
y_scores = np.array(list(map(int,pre)))
roc=str(roc_auc_score(y_true, y_scores))    #计算ROC AUC
precision, recall, _thresholds = precision_recall_curve(y_true, y_scores)
pr =str(auc(recall, precision))     #计算PR AUC
title="ROC AUC:"+roc+"\n"+"PR AUC:"+pr
labels_name=["0.0","1.0"]
plot_confusion_matrix(cm, labels_name, title)
for x in range(len(cm)):
    for y in range(len(cm[0])):
        plt.text(y,x,cm[x][y],color='white',fontsize=10, va='center')   #text函数坐标是反着的
plt.show()
