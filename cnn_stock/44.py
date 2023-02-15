import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
import datetime
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, auc

tf.compat.v1.disable_eager_execution()
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 读取数据集
df = pd.read_csv("./dataset/tt.csv")
df.info
df.head()
df.describe()
plt.figure(figsize=(15,5));#figsize=(15,5)表示figure 的大小为宽、长（单位为inch
plt.subplot(2,1,1);#subplot（2,1,1）指的是在一个2行1列共2个子图的图中，定位第1个图来进行操作。最后的数字就是表示第几个子图，此数字的变化来定位不同的子图。
plt.plot(df.open.values,color='red',label='open')#红色曲线绘制开盘价变化情况
plt.plot(df.close.values,color='green',label='close')#绿色曲线绘制收盘价变化情况
plt.plot(df.low.values,color='blue',label='low')#蓝色曲线表示最低价变化情况
plt.plot(df.high.values,color='black',label='high')#黑色曲线绘制最高价变化情况
plt.title('stock price')
plt.xlabel('time [days]')
plt.ylabel('price')
#上面参数loc表示location哦，代表标签所放置的位置哦，下面的loc=’best’的意思就是图形你自己看着办吧，放到合适的位置就行啦，
plt.legend(loc = 'best')

plt.subplot(2,1,2)#定位第2个子图
plt.plot(df.vol.values,color='black',label='volume')
plt.title('stock volume')
plt.xlabel('time [days]')
plt.ylabel('volume')
plt.legend(loc = 'best')
plt.show()
#按照80%10%10%划分数据集。验证集，测试集
valid_set_size_percentage=10
test_set_size_percentage=10
# 数据归一化
def normalize_data(df):
    min_max_scaler=sklearn.preprocessing.minmax_scale()
    df['open']=min_max_scaler.fit_transform(df.open.values.reshape(-1,1))
    df['high']=min_max_scaler.fit_transform(df.high.values.reshape(-1,1))
    df['low']=min_max_scaler.fit_transform(df.low.values.reshape(-1,1))
    return df


# 定义输入序列并分割数据集

def load_data(stock,seq_len=20):
     data_raw=stock.to_numpy() # pd to numpy array
     data=[]
     # 创建所有可能的长度序列seq_len
     for index in range(len(data_raw)-seq_len):
         data.append(data_raw[index:index+seq_len])
     data=np.array(data)
     valid_set_size=int(np.round(valid_set_size_percentage/100 *data.shape[0]))
     test_set_size=int(np.round(test_set_size_percentage/100 * data.shape[0]))
     train_set_size=data.shape[0]-(valid_set_size+test_set_size)
     x_train=data[:train_set_size,:-1,:]
     y_train=data[:train_set_size,-1,:]
     x_valid=data[train_set_size:train_set_size+valid_set_size,:-1,:]
     y_valid=data[train_set_size:train_set_size+valid_set_size,-1,:]
     x_test=data[train_set_size+valid_set_size:,:-1,:]
     y_test=data[train_set_size+valid_set_size:,-1,:]
     return [x_train,y_train,x_valid,y_valid,x_test,y_test]
seq_len=20 # choose sequence length
x_train,y_train,x_valid,y_valid,x_test,y_test=load_data(df,20)
print('x_train.shape =',x_train.shape)#经过处理，训练数据集是5207条，所以它的数据就是0~19，维度是4（因为有四个数据，开盘价，收盘价，最高价和最低价）
print('y_train.shape =',y_train.shape)
print('x_valid.shape =',x_valid.shape)#验证数据是651条，
print('y_valid.shape =',y_valid.shape)
print('x_test.shape =',x_test.shape)#测试数据是651条
print('y_test.shape =',y_test.shape)

plt.figure(figsize=(15, 6));
plt.plot(df.open.values,color='red',label='open')
plt.plot(df.close.values,color='green',label='close')
plt.plot(df.low.values,color='blue',label='low')
plt.plot(df.high.values,color='black',label='high')
plt.title('stock')
plt.xlabel('time [days]')
plt.ylabel('normalized price/volume')
plt.legend(loc='best')
plt.show()


# 对训练数据随机化处理
index_in_epoch = 0;
perm_array = np.arange(x_train.shape[0])
np.random.shuffle(perm_array)  # shuffle将数据打乱


# 数据读取方法
def get_next_batch(batch_size):  # 从数据里边随机抽取batch_size的数量
    global index_in_epoch, x_train, perm_array
    start = index_in_epoch
    index_in_epoch += batch_size

    if index_in_epoch > x_train.shape[0]:
        np.random.shuffle(perm_array)
        start = 0
        index_in_epoch = batch_size

    end = index_in_epoch
    return x_train[perm_array[start:end]], y_train[perm_array[start:end]]  # 从x_train，y_train分别读取训练数据和标签


#定义超参
n_steps = seq_len-1#
#输入大小（与指标数量对应）
n_inputs = 4   #输入指标的数量
n_neurons =200 #循环神经网络有多少个神经元
#输出大小（与指标数量对应）
n_outputs = 4#输出指标的数量
#层数
n_layers = 2
#学习率
learning_rate =0.001
#批大小
batch_size = 50
#迭代训练次数
n_epochs = 20
#训练集大小
train_set_size = x_train.shape[0]#获取训练集的大小即训练数据的行数
#测试集大小
test_set_size = x_test.shape[0]

# 用的tensorflow定义网络的结构
tf.compat.v1.reset_default_graph()

X = tf.compat.v1.placeholder(tf.float32, [None, n_steps, n_inputs])  # 用tf.placeholder定义了输入x和输出y
y = tf.compat.v1.placeholder(tf.float32, [None, n_outputs])

# 使用GRU单元结构
layers = [tf.compat.v1.nn.rnn_cell.GRUCell(num_units=n_neurons, activation=tf.nn.leaky_relu)  # 使用GRUCell定义了两层的神经网络
          for layer in range(n_layers)]

multi_layer_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(layers)  # 将定义好的神经网络放在MultiRNNCell，构造一个多层的cell
rnn_outputs, states = tf.compat.v1.nn.dynamic_rnn(multi_layer_cell, X,
                                        dtype=tf.float32)  # 将multi_layer_cell放到dynamic_rnn里面构成一个循环神经网络

stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
stacked_outputs = tf.compat.v1.layers.dense(stacked_rnn_outputs, n_outputs)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])  # 将循环神经网络的输出reshape成我们需要的大小
outputs = outputs[:, n_steps - 1, :]  # 定义输出

# 将预测与实际结果求均方误差损失
loss = tf.reduce_mean(tf.square(outputs - y))  # 使用MSE作为损失
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)  # 采用Adam优化方法
training_op = optimizer.minimize(loss)  # 最小化loss值
# run


with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for iteration in range(int(n_epochs * train_set_size / batch_size)):
        x_batch, y_batch = get_next_batch(batch_size)  # fetch the next training batch
        sess.run(training_op, feed_dict={X: x_batch, y: y_batch})  # 依次运行training_op
        if iteration % int(
                5 * train_set_size / batch_size) == 0:  # 获取最小化的损失，每隔(5*train_set_size/batch_size)步就print一次MSE的值
            mse_train = loss.eval(feed_dict={X: x_train, y: y_train})
            mse_valid = loss.eval(feed_dict={X: x_valid, y: y_valid})
            print('%.2f epochs: MSE train/valid = %.6f/%.6f' % (
                iteration * batch_size / train_set_size, mse_train, mse_valid))

    y_train_pred = sess.run(outputs, feed_dict={X: x_train})  # 将训练集进行预测
    y_valid_pred = sess.run(outputs, feed_dict={X: x_valid})  # 将验证集进行预测
    y_test_pred = sess.run(outputs, feed_dict={X: x_test})  # 将测试集进行预测



ft = 0 # 0 = open, 1 = close, 2 = highest, 3 = lowest

#结果可视化
plt.figure(figsize=(15, 5));
plt.subplot(1,2,1);
#测试数据可视化
plt.plot(np.arange(y_train.shape[0]), y_train[:,ft], color='blue', label='train target')

plt.plot(np.arange(y_train.shape[0], y_train.shape[0]+y_valid.shape[0]), y_valid[:,ft],
         color='gray', label='valid target')

plt.plot(np.arange(y_train.shape[0]+y_valid.shape[0],
                   y_train.shape[0]+y_test.shape[0]+y_test.shape[0]),
         y_test[:,ft], color='black', label='test target')

plt.plot(np.arange(y_train_pred.shape[0]),y_train_pred[:,ft], color='red',
         label='train prediction')
#验证集数据可视化
plt.plot(np.arange(y_train_pred.shape[0], y_train_pred.shape[0]+y_valid_pred.shape[0]),
         y_valid_pred[:,ft], color='orange', label='valid prediction')
#将测试集的预测结果可视化，看测试结果与我们实际结果存在多大的误差
plt.plot(np.arange(y_train_pred.shape[0]+y_valid_pred.shape[0],
                   y_train_pred.shape[0]+y_valid_pred.shape[0]+y_test_pred.shape[0]),
         y_test_pred[:,ft], color='green', label='test prediction')

plt.title('past and future stock prices')
plt.xlabel('time [days]')
plt.ylabel('normalized price')
plt.legend(loc='best');

plt.subplot(1,2,2);

plt.plot(np.arange(y_train.shape[0], y_train.shape[0]+y_test.shape[0]),
         y_test[:,ft], color='black', label='test target')

plt.plot(np.arange(y_train_pred.shape[0], y_train_pred.shape[0]+y_test_pred.shape[0]),
         y_test_pred[:,ft], color='green', label='test prediction')

plt.title('future stock prices')
plt.xlabel('time [days]')
plt.ylabel('normalized price')
plt.legend(loc='best');
plt.show()

f=open('f/result.txt','r')
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
