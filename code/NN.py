import numpy as np
# 导入sigmod函数
import scipy.special as ss
import matplotlib.pyplot as plt


# from 线搜索 import ClassLineSreach


# function expit()

class neuralNetwork:
    def __doc__(self):
        """aaa"""

    # 初始化函数设定输入层节点隐藏层节点和输出层节点的数量
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        # self.wih = (np.random.rand(self.hnodes,self.inodes))
        # self.woh = (np.random.rand(self.onodes,self.hnodes))
        # 使用正态分布采样权重，期望是0，方差是1/下一层节点数**-.5
        self.wih = (np.random.normal(0.0, pow(self.hnodes, -.5), (self.hnodes, self.inodes)))
        self.who = (np.random.normal(0.0, pow(self.onodes, -.5), (self.onodes, self.hnodes)))
        # 定义激活函数
        self.activation_function = lambda x: ss.expit(x)

        pass

    # 自定义权重矩阵
    def set_w(self, valueih, valueho):
        self.wih = np.array(valueih) * np.ones((self.hnodes, self.inodes))
        self.who = np.array(valueho) * np.ones((self.onodes, self.hnodes))

    def reset_w(self, ih, ho):
        self.wih = ih
        self.who = ho

    # 训练 学习给定训练集样本，优化权重
    def train(self, input_list, target_list):
        # 构造目标矩阵
        targets = np.array(target_list, ndmin=2).T
        # 构建输入矩阵
        inputs = np.array(input_list, ndmin=2).T
        # 计算隐藏层输入
        hidden_inputs = self.wih @ inputs
        # 计算隐藏层输出
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算输出层输入
        final_inputs = self.who @ hidden_outputs
        # 计算输出层输出
        final_outputs = self.activation_function(final_inputs)
        # 计算输出层误差
        output_error = targets - final_outputs
        # 计算隐含层误差
        hidden_errors = self.who.T @ output_error
        # 更新隐藏层与输出层的权重
        self.who += self.lr * np.dot((output_error * final_outputs * (1.0 - final_outputs)),
                                     np.transpose(hidden_outputs))
        # 更新隐藏层与输入层的权重
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                     np.transpose(inputs))

    # 查询 给定输入从输出街店给出答案
    def query(self, input_list):
        # 构建输入矩阵
        inputs = np.array(input_list, ndmin=2).T
        # 计算隐藏层输入
        hidden_inputs = np.dot(self.wih, inputs)
        # 计算隐藏层输出
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算输出层输入
        final_inputs = np.dot(self.who, hidden_outputs)
        # 计算输出层输出
        final_outputs = self.activation_function(final_inputs)
        # 返回最终输出
        return final_outputs

    # 反向查询神经网络
    def backquery(self, target_list):
        # 计算输出层输出信号，转换为列向量
        final_outputs = np.array(target_list, ndmin=2).T
        # 计算输出层输入信号，使用SIGMOD函数的逆函数
        final_inputs = self.inverse_activation_function(final_outputs)

        # 计算隐藏层输出信号
        hidden_outputs = self.who.T @ final_inputs
        # 将信号格式化(.01 ~ .99)

    def setlearnning(self, lr):
        self.lr = lr

    def get_wh(self):
        return (self.wih, self.who)


def demo1():
    # 构造神经网络
    inputnodes = 784
    hidennodes = 100
    outputnodes = 10
    learningeate = .3
    n = neuralNetwork(inputnodes, hidennodes, outputnodes, learningeate)

    # 数据处理
    data_file = open("mnist_train_100.csv", 'r')
    data_list = data_file.readlines()
    data_file.close()
    # 使用100条数据进行训练
    for record in data_list:
        all_values = record.split(',')
        # 去掉第一个标签值
        image_array = np.asfarray(all_values[1:]).reshape((28, 28))
        # 标准化输入(.01~.99)
        scaled_input = (np.asfarray(all_values[1:]) / 255.0 * .99 + .01)
        # 构造标签
        targets = np.zeros(outputnodes) + .01
        targets[int(all_values[0])] = 0.99
        # 开始训练
        n.train(scaled_input, targets)

    # 测试网络
    # 导入测试集
    test_data_file = open("mnist_test_10.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    # 应用测试实例"3"
    test_values = test_data_list[1].split(',')
    test_scaled_input = (np.asfarray(test_values[1:]) / 255.0 * .99) + .01
    print(n.query(test_scaled_input))
    image_array_test = np.asfarray(test_values[1:]).reshape((28, 28))

    # print(scaled_input)
    plt.imshow(image_array_test, cmap='Greys', interpolation='None')





def learnningplot(learnninglist, performance, epochs=1):
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)  # 导入中文字体
    preformancelist = [performance(i, epochs) for i in learnninglist]
    plt.plot(learnninglist, preformancelist, 'go-')
    plt.grid(True)
    plt.xlabel("学习率", FontProperties=font)
    plt.ylabel("准确率", FontProperties=font)
    plt.title("学习率与准确率关系")


def demo2_performance(lr=None, epochs=1):
    # 构造神经网络
    inputnodes = 784
    hidennodes = 100
    outputnodes = 10
    learningeate = lr
    n = neuralNetwork(inputnodes, hidennodes, outputnodes, learningeate)

    # 导入手写图片训练集(60000)
    training_data_file = open("mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # 训练神经网络
    for e in range(epochs):
        # 通过“，”分割数据
        for record in training_data_list:
            all_values = record.split(",")
            # 归一化输入(防止0)
            inputs = (np.asfarray(all_values[1:]) / 255.0 * .99) + .01
            # 构造目标(target)矩阵
            targets = np.zeros(outputnodes) + .01
            targets[int(all_values[0])] = .99
            n.setlearnning(lr)
            n.train(inputs, targets)
            pass
        pass

    # 导入测试集
    test_data_file = open("mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # 测试神经网络

    # 设置计分板列表
    scorecard = []
    # 计算所有数字的得分情况
    for record in test_data_list:
        all_values = record.split(",")
        # 正确答案是第一位
        correct_labble = int(all_values[0])
        # 归一化输入
        inputs = (np.asfarray(all_values[1:]) / 255.0 * .99) + .01
        # 输出结果
        outputs = n.query(inputs)
        # 将输出的最高分作为答案
        label = np.argmax(outputs)
        # 将答案填入列表
        if (label == correct_labble):
            # 如果答案正确，加一分
            scorecard.append(1)
        else:
            # 如果答案不正确，加0分
            scorecard.append(0)
            pass
        pass
    # 计算总分，算出回归率
    scorecard_array = np.asfarray(scorecard)
    performance = scorecard_array.sum() / scorecard_array.size
    return performance
    # print("preformance=", scorecard_array.sum() / scorecard_array.size)


def epochsplot(epochslist, performance, lr):
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)  # 导入中文字体
    preformancelist = [performance(lr, i) for i in epochslist]
    plt.plot(epochslist, preformancelist, 'bo-')
    plt.grid(True)
    plt.xlabel("世代数", FontProperties=font)
    plt.ylabel("准确率", FontProperties=font)
    plt.title("世代数与准确率关系", FontProperties=font)


def demo_linearfit(lr=None, epochs=1):
    # 构造神经网络
    inputnodes = 100
    hidennodes = 100
    outputnodes = 100
    learningeate = 0.001
    n = neuralNetwork(inputnodes, hidennodes, outputnodes, learningeate)

    # 训练神经网络
    # 归一化输入输出
    x = np.linspace(-1, 1, 100)
    X = np.asfarray(x)
    inputs = (X / (max(X) - min(X)) * 0.99) + .01

    # 归一化标签
    beta = [3, 2, 1]
    y = beta[0] * X * X + beta[1] * np.random.rand(X.size) * X + beta[2]
    targets = (y / (max(y) - min(y)) * 0.99) + .01

    # targets = (y / (max(y) - min(y)) * 0.99) + .01
    # targets = np.array([0.4, 0.5, 0.9])

    # # 画出原始图像
    # plt.scatter(x, y)
    # plt.show()

    # 设置学习率
    n.setlearnning(lr)
    for e in range(epochs):
        n.train(inputs, targets)
    # 打印实际值
    print("实际值（归一化后数据）", targets)
    # 获得预测值
    print("预测参数为：", n.query(inputs))
    # 画出原始图像
    plt.scatter(X, y)
    plt.show()
    # 画预测图
    res = n.query(inputs)
    X = np.linspace(-1, 1, 100)
    plt.plot(X, res, color='r')
    plt.show()


def demo_linearfit2_func(beta, X):
    return beta[0] * X + beta[1]


def demo_linearfit2(lr, frequ=100):
    # 构造神经网络
    inputnodes = 1
    hidennodes = 10
    outputnodes = 1

    n = neuralNetwork(inputnodes, hidennodes, outputnodes, lr)
    # 自定义权重矩阵
    n.set_w(1, 0.4)
    # 输出此时的神经网络权重
    print("训练前：\nwih层的权重为{},\nwho的权重矩阵为{}".format(n.get_wh()[0], n.get_wh()[1]))
    # 训练神经网络
    """# 归一化输入输出"""
    x = np.linspace(-2, 2, frequ)
    X = np.asfarray(x)
    beta = [3, 2]
    # 不重复采样
    # y_origin = beta[0] * X + beta[1]
    y = np.random.choice(demo_linearfit2_func(beta, X), size=frequ, replace=False)
    # 加入噪点
    # y = beta[0] * X * X + beta[1] * np.random.rand(X.size) * X + beta[2]

    # 归一化标签
    targets = (y / (max(y) - min(y)) * 0.99) + .01
    # 通过反函数找到对应的横坐标
    X = (y - beta[1]) / beta[0]
    # 归一化输入
    inputs = (X / (max(X) - min(X)) * 0.99) + .01
    # 反归一输入
    anti_inputs = ((inputs - .01) / 0.99) * (max(X) - min(X))
    # 用frequ个数据进行训练
    n.setlearnning(lr)
    for i in range(frequ):
        n.train(inputs[i], targets[i])
        print("训练第{}次：\nwih层的权重为{},\nwho的权重矩阵为{}".format(i, n.get_wh()[0], n.get_wh()[1]))
    # 用1000个数据进行拟合
    x_query = np.linspace(-2, 2, 1000)
    x_query = np.asfarray(x_query)
    # query_inputs = (x_query / (max(x_query) - min(x_query)) * 0.99) + .01
    res = [float(n.query(j)) for j in x_query]
    res = np.array(res)
    # 反归一输出
    anti_outputs = ((res - .01) / 0.99) * (max(res) - min(res))
    # 画出采样散点图
    plt.scatter(anti_inputs, y)
    # 画出原始曲线
    plt.plot(x_query, demo_linearfit2_func(beta, x_query))
    # 画出预测曲线
    plt.plot(x_query, anti_outputs, color='r')
    plt.show()


def just_query(frequ, ih=None, ho=None):
    # 构造神经网络
    inputnodes = 1
    hidennodes = 5
    outputnodes = 1

    n = neuralNetwork(inputnodes, hidennodes, outputnodes, learningrate=0.06)
    n.reset_w(
        np.array(
            [[0.6],
             [0.6],
             [0.6],
             [0.6],
             [0.6]]),
        np.array([[0.4, 0.4, 0.4, 0.4, 0.4]])
    )
    # 用1000个数据进行拟合
    x_query = np.linspace(-2, 2, 1000)
    x_query = np.asfarray(x_query)
    res = [float(n.query(j)) for j in x_query]
    res = np.array(res)
    # 反归一输出
    anti_outputs = ((res - .01) / 0.99) * (max(res) - min(res))
    # 画出预测曲线
    plt.plot(x_query, anti_outputs, color='r')
    plt.show()


if __name__ == '__main__':
    # demo_linearfit(0.01, 20)
    demo_linearfit2(0.04, 10)
    just_query(10)

# if __name__ == '__main__':
#     lr=0.4
#     epochslist = range(1,10)
#     epochsplot(epochslist,demo2_performance,lr)
#
# if __name__ == '__main__':
#     learnninglist = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#     learnningplot(learnninglist, demo2_performance,1)
#
#
# if __name__ == '__mai__':
#     # 使用牛顿法以及线搜索改进的BP神经网络
#     pass
#     # 1.构造神经网络
#     inputnodes = 784
#     hiddennodes = 100
#     outputnodes = 10
#     n = neuralNetwork_newton(inputnodes, hiddennodes, outputnodes, 1000, 10 ** (-5))
#     # 2. 导入手写图片训练集(100)
#     training_data_file = open("mnist_train_100.csv", 'r')
#     training_data_list = training_data_file.readlines()
#     training_data_file.close()
#     epochs = 2
#     for e in range(epochs):
#         # 通过“，”分割数据
#         for record in training_data_list:
#             all_values = record.split(",")
#             # 归一化输入(防止0)
#             inputs = (np.asfarray(all_values[1:]) / 255.0 * .99) + .01
#             # 构造目标(target)矩阵
#             targets = np.zeros(outputnodes) + .01
#             targets[int(all_values[0])] = .99
#             n.train(inputs, targets)
#             pass
#         pass
#
#     # 导入测试集
#     test_data_file = open("mnist_test_10.csv", 'r')
#     test_data_list = test_data_file.readlines()
#     test_data_file.close()
#
#     # 测试神经网络
#
#     # 设置计分板列表
#     scorecard = []
#     # 计算所有数字的得分情况
#     for record in test_data_list:
#         all_values = record.split(",")
#         # 正确答案是第一位
#         correct_labble = int(all_values[0])
#         # 归一化输入
#         inputs = (np.asfarray(all_values[1:]) / 255.0 * .99) + .01
#         # 输出结果
#         outputs = n.query(inputs)
#         # 将输出的最高分作为答案
#         label = np.argmax(outputs)
#         # 将答案填入列表
#         if (label == correct_labble):
#             # 如果答案正确，加一分
#             scorecard.append(1)
#         else:
#             # 如果答案不正确，加0分
#             scorecard.append(0)
#             pass
#         pass
#     # 计算总分，算出回归率
#     scorecard_array = np.asfarray(scorecard)
#     print("preformance=", scorecard_array.sum() / scorecard_array.size)
