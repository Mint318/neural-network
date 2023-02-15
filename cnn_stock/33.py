import numpy as np
from scipy import stats
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

tf.compat.v1.disable_eager_execution()
import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# 读取数据集
df=pd.read_csv("./dataset/tt.csv")
# 归一化
def normalize_data(df):
    min_max_scaler=sklearn.preprocessing.minmax_scale()
    df['open']=min_max_scaler.fit_transform(df.open.values.reshape(-1,1))
    df['high']=min_max_scaler.fit_transform(df.high.values.reshape(-1,1))
    df['low']=min_max_scaler.fit_transform(df.low.values.reshape(-1,1))
    return df
# 定义输入序列并分割数据集
valid_set_size_percentage=10
test_set_size_percentage=10
def load_data(stock,seq_len=20):
    data_raw=stock.values  # pd to numpy array
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

# 模型参数
n_steps=seq_len
n_inputs=3
n_neurons=200
n_outputs=3
n_layers=2
learning_rate=0.001
batch_size=50
n_epochs=100
train_set_size=x_train.shape[0]
test_set_size=x_test.shape[0]
# 定义模型结构
X=tf.placeholder(tf.float32,[None,n_steps,n_inputs])
y=tf.placeholder(tf.float32,[None,n_outputs])
# 使用GRU cell
layers=[tf.contrib.rnn.GRUCell(num_units=n_neurons,activation=tf.nn.leaky_relu)
for layer in range(n_layers)]
multi_layer_cell=tf.contrib.rnn.MultiRNNCell(layers)
rnn_outputs,states=tf.nn.dynamic_rnn(multi_layer_cell,X,dtype=tf.float32)
stacked_rnn_outputs=tf.reshape(rnn_outputs,[-1,n_neurons])
stacked_outputs=tf.layers.dense(stacked_rnn_outputs,n_outputs)
outputs=tf.reshape(stacked_outputs,[-1,n_steps,n_outputs])
# 保留序列的最后一个输出
outputs=outputs[:,n_steps-1,:]

# 定义损失及优化器
loss=tf.reduce_mean(tf.square(outputs-y)) # loss function=mean squared error
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op=optimizer.minimize(loss)

# run
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for iteration in range(int(n_epochs * train_set_size/batch_size)):
        x_batch,y_batch=get_next_batch(batch_size)  # 获取下一个批训练
        sess.run(training_op,feed_dict={X:x_batch,y:y_batch})
        if iteration % int(5 * train_set_size/batch_size) == 0:
          mse_train=loss.eval(feed_dict={X:x_train,y:y_train})
          mse_valid=loss.eval(feed_dict={X:x_valid,y:y_valid})
print('%.2f epochs:MSE train/valid=%.6f/%.6f'%(iteration * batch_size/train_set_size, mse_train,mse_valid))
y_train_pred=sess.run(outputs,feed_dict={X:x_train})
y_valid_pred=sess.run(outputs,feed_dict={X:x_valid})
y_test_pred=sess.run(outputs,feed_dict={X:x_test})

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