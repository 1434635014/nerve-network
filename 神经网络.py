import numpy
import scipy.special
import scipy.io as io
import time

import cv2
import PIL
import csv
import codecs
import pandas

class neuralNetwork:
    # 初始化函数
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        if (inputnodes == False):
            self.wih = hiddennodes
            self.who = outputnodes
        else:
            # 输入层节点数，隐藏层节点数。输出层节点数
            self.inodes = inputnodes
            self.hnodes = hiddennodes
            self.onodes = outputnodes
            # 学习率
            self.lr = learningrate
            # 生成权重矩阵，这里平方0.5是为了得到0-1之间的数
            # 输入层与隐藏层之间的权重矩阵
            self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
            # 隐藏层与输出层之间的权重矩阵
            self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        # 定义匿名函数（激活函数）
        self.activation_function = lambda x: scipy.special.expit(x)
    # 训练函数
    def train(self, inputs_list, targets_list):
        # 输入的节点，和目标节点
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)
        
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
    # 查询函数
    def query(self, inputs_list):
        # 输入的节点
        inputs = numpy.array(inputs_list, ndmin=2).T
        # 隐藏层乘以权重输出的节点（进行激活函数前）
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 隐藏层输出节点（进行激活函数后）
        hidden_outputs = self.activation_function(hidden_inputs)
        0
        # 最终乘以权重输出的节点（进行激活函数前）
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 最终层输出节点（进行激活函数后）
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
        
    
# 输入层节点，隐藏层节点。输出层节点
is_train = False        # 是否进行训练，否则只是对模型进行本地测试
inputnodes = 784
hiddennodes = 200       # 隐藏节点数（学习容量）
outputnodes = 10
# 世代
epochs = 10
# 学习率
learningrate = 0.1
train_fileurl = './csv/mnist_train.csv'     # ./csv/mnist_100.csv
test_fileurl = './csv/mnist_test.csv'       # ./csv/mnist_10.csv
moxing_file_name = './csv/moxing_' + str(epochs)    # 训练模型位置
size = 2000                                 # 打印间隔（数据个数）
# 本地图片
imgNum = 0                                  # 图片数字

n = []
# 创建神经网络对象
if (is_train):
    n = neuralNetwork(inputnodes, hiddennodes, outputnodes, learningrate)
else:
    moxing = io.loadmat(moxing_file_name)
    moxing_data_list_wih = moxing['wih']  # 读取mat文件
    moxing_data_list_who = moxing['who']  # 读取mat文件
    n = neuralNetwork(is_train, moxing_data_list_wih, moxing_data_list_who, [])
    
if (is_train):
    # 加载训练集
    training_data_file = open(train_fileurl, 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    num = 0             # 训练数量
    begin = time.time()
    for epochsIndex in range(epochs):
        print ('进入第' + str(epochsIndex + 1) + '个世代')
        # 循环所有训练数据
        for record in training_data_list: 
            # 切割逗号为数组
            all_values = record.split(',')
            # 输入节点
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # 目标节点（不能使用0和1，所以使用0.01 和 0.99 表示）
            targets = numpy.zeros(outputnodes) + 0.01
            targets[int(all_values[0])] = 0.99   # 这是目标值
            # 打印训练个数
            num += 1
            if num % size == 0:
                end = time.time()           # 结束计时       
                print('训练中：第' + str(num) + '个数据，已耗时：' + str(end - begin) + '秒')
            n.train(inputs, targets)
    print('训练完成')
    # 积分卡
    # 加载测试集
    print("测试集测试中...")
    test_data_file = open(test_fileurl, 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    scorecard = []
    for record in test_data_list:
        all_values = record.split(',')
        correct_label = int(all_values[0])
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        
        # 得出结果
        outputs = n.query(inputs)
        label = numpy.argmax(outputs)
        if (label == correct_label):    # 正确
            scorecard
            scorecard.append(1)
        else:                           # 错误
            scorecard
            scorecard.append(0)
    # print(scorecard)
    scorecard_array = numpy.asarray(scorecard)
    # 正确率
    rate = scorecard_array.sum() / scorecard_array.size * 100

    print ("正确率为 = %.1f" % (rate) + '%')

    # 写入模型
    io.savemat(moxing_file_name, {'wih': n.wih, 'who': n.who})
    print("模型创建成功")

def ImageToMatrix(filename):
    # 读取图片
    im = PIL.Image.open(filename)
    size = (28, 28)
    im = im.resize(size, PIL.Image.ANTIALIAS)
    im = numpy.array(im.convert("L"))
    imList = numpy.array([])
    for row in im:
        rowList = numpy.array([])
        for i in row:
            i = 255 - i
            rowList = numpy.append(rowList, 0 if i < 10 else i)
        imList = numpy.concatenate((imList, rowList))
    return imList

if (is_train == False):
    print("正在识别本地图片...")
    im = ImageToMatrix("./img/" +str(imgNum)+ ".jpg")
    img = (numpy.array(im / 255.0 * 0.99) + 0.01)
    # 得出结果
    outputs = n.query(img)
    label = numpy.argmax(outputs)
    print("识别为" + str(label) + "，" + ("正确" if imgNum == int(label) else "错误"))