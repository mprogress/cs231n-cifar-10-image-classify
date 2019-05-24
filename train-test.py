import os
from PIL import Image
import numpy as np
import tensorflow as tf



train_dir = "./data/train"
test_dir = "./data/test"

model_path = "./model/"



# 从文件夹读取图片和标签到numpy数组
def read_train_data(train_dir):
    train_datas = []
    train_labels = []
    train_fpaths = []
    for fname in os.listdir(train_dir):
        train_fpath = os.path.join(train_dir, fname)
        train_fpaths.append(train_fpath)
        train_image = Image.open(train_fpath)
        train_data = np.array(train_image) / 255.0
        train_label = int(fname.split("_")[0])
        train_datas.append(train_data)
        train_labels.append(train_label)

    train_datas = np.array(train_datas)
    train_labels = np.array(train_labels)

    print("shape of datas: {}\tshape of labels: {}".format(train_datas.shape, train_labels.shape))
    return train_fpaths,train_datas, train_labels


def read_test_data(test_dir):
    test_datas = []
    test_labels = []
    test_fpaths = []
    for fname in os.listdir(test_dir):
        test_fpath = os.path.join(test_dir, fname)
        test_fpaths.append(test_fpath)
        test_image = Image.open(test_fpath)
        test_data = np.array(test_image) / 255.0
        test_label = int(fname.split("_")[0])
        test_datas.append(test_data)
        test_labels.append(test_label)

    test_datas = np.array(test_datas)
    test_labels = np.array(test_labels)

    print("shape of datas: {}\tshape of labels: {}".format(test_datas.shape, test_labels.shape))
    return test_fpaths, test_datas, test_labels

train_fpaths, train_datas, train_labels = read_train_data(train_dir)

train_num_classes = len(set(train_labels))

test_fpaths, test_datas, test_labels = read_test_data(test_dir)

train_num_classes = len(set(test_labels))



# 存放输入和标签
datas_placeholder = tf.placeholder(tf.float32, [None, 32, 32, 3])
labels_placeholder = tf.placeholder(tf.int32, [None])

# 存放DropOut参数的容器，训练时为0.25，测试时为0
dropout_placeholdr = tf.placeholder(tf.float32)

# 定卷积层, 20个卷积核, 大小为5，Relu
conv0 = tf.layers.conv2d(datas_placeholder, 20, 5, activation=tf.nn.relu)

# 池化操作
pool0 = tf.layers.max_pooling2d(conv0, [2, 2], [2, 2])

# 卷积层
conv1 = tf.layers.conv2d(pool0, 40, 4, activation=tf.nn.relu)

# 池化操作
pool1 = tf.layers.max_pooling2d(conv1, [2, 2], [2, 2])

# 将3维特征转换为1维向量
flatten = tf.layers.flatten(pool1)

# 全连接层
fc = tf.layers.dense(flatten, 400, activation=tf.nn.relu)

# DropOut层
dropout_fc = tf.layers.dropout(fc, dropout_placeholdr)

# 未激活的输出层
logits = tf.layers.dense(dropout_fc, train_num_classes)

predicted_labels = tf.arg_max(logits, 1)


# 利用交叉熵定义损失
losses = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(labels_placeholder, train_num_classes),
    logits=logits
)
# 平均损失
mean_loss = tf.reduce_mean(losses)

# 定义优化器，指定要优化的损失函数
optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(losses)

# 用于保存和载入模型
saver = tf.train.Saver()



with tf.Session() as sess:
        print("训练模式")
        # 如果是训练，初始化参数
        sess.run(tf.global_variables_initializer())
        # 定义输入和Label以填充容器，训练时dropout为0.25
        train_feed_dict = {
            datas_placeholder: train_datas,
            labels_placeholder: train_labels,
            dropout_placeholdr: 0.25
        }
        for step in range(100):
            _, mean_loss_val = sess.run([optimizer, mean_loss], feed_dict=train_feed_dict)

            if step % 10 == 0:
                print("step = {}\tmean loss = {}".format(step, mean_loss_val))
        saver.save(sess, model_path)
        print("训练结束，保存模型到{}".format(model_path))

        print("测试模式")
        # 如果是测试，载入参数
        saver.restore(sess, model_path)
        print("从{}载入模型".format(model_path))
        # label和名称的对照关系
        label_name_dict = {
            0:"飞机",
            1:"汽车",
            2:"鸟",
            3:"猫",
            4:"鹿",
            5:"狗",
            6:"青蛙",
            7:"马",
            8:"船",
            9:"卡车"
        }
        # 定义输入和Label以填充容器，测试时dropout为0
        test_feed_dict = {
            datas_placeholder: test_datas,
            labels_placeholder: test_labels,
            dropout_placeholdr: 0
        }
        predicted_labels_val = sess.run(predicted_labels, feed_dict=test_feed_dict)
        # 真实label与模型预测label
        for test_fpath, real_label, predicted_label in zip(test_fpaths, test_labels, predicted_labels_val):
            # 将label id转换为label名
            real_label_name = label_name_dict[real_label]
            predicted_label_name = label_name_dict[predicted_label]
            print("{}\t{} => {}".format(test_fpath, real_label_name, predicted_label_name))




        













