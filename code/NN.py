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
        self.who = (np.random.normal(0.0, pow(self.hnodes, -.5), (self.onodes, self.hnodes)))
        # 定义激活函数
        self.activation_function = lambda x: ss.expit(x)

        pass

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
        return (self.wih,self.who)

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


class neuralNetwork_newton(neuralNetwork):
    def __init__(self, inputnodes, hiddennodes, outputnodes, maxm, epsilon):
        super(neuralNetwork_newton, self).__init__(inputnodes, hiddennodes, outputnodes, learningrate=None)
        self.maxm = maxm
        self.epsilon = epsilon

    def train(self, input_list, target_list):
        # 构造目标矩阵
        self.targets = np.array(target_list, ndmin=2).T
        # 构建输入矩阵
        self.inputs = np.array(input_list, ndmin=2).T
        """# 计算隐藏层输入
        self.hidden_inputs = self.wih @ self.inputs
        # 计算隐藏层输出
        self.hidden_outputs = self.activation_function(self.hidden_inputs)
        # 计算输出层输入
        self.final_inputs = self.who @ self.hidden_outputs
        # 计算输出层输出
        self.final_outputs = self.activation_function(self.final_inputs)
        # 计算输出层误差
        self.output_error = self.targets - self.final_outputs
        # 计算隐含层误差
        self.hidden_errors = self.who.T @ self.output_error"""
        # 更新隐藏层与输出层的权重
        m = 0
        while np.linalg.norm(self.gfun_who(self.who)) > self.epsilon and m < self.maxm:
            dk = -np.linalg.inv(self.G_who(self.who)) @ self.gfun_who(self.who)
            Armoji = ClassLineSreach(.2, .5, 100, self.gfun_who, self.fun_who)
            self.who = Armoji.armijo(self.who, dk)
            self.mk = Armoji.getmk()
            print("梯度范数为 %s,函数最小值为 %s" % (np.linalg.norm(self.gfun_who(self.who)), self.fun_who(self.who)))
            m += 1
        # 更新输入层与隐藏层的权重
        m = 0
        while np.linalg.norm(self.gfun_wih(self.wih)) > self.epsilon and m < self.maxm:
            dk = -np.linalg.inv(self.G_wih(self.wih)) @ self.gfun_wih(self.wih)
            Armoji = ClassLineSreach(.2, .5, 100, self.gfun_wih, self.fun_wih)
            self.wih = Armoji.armijo(self.wih, dk)
            self.mk = Armoji.getmk()
            print("梯度范数为 %s,函数最小值为 %s" % (np.linalg.norm(self.gfun_wih(self.wih)), self.fun_wih(self.wih)))
            m += 1

    def gfun_who(self, who):
        # 计算隐藏层输入
        hidden_inputs = self.wih @ self.inputs
        # 计算隐藏层输出
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算输出层输入
        final_inputs = who @ hidden_outputs
        # 计算输出层输出
        final_outputs = self.activation_function(final_inputs)
        # 计算输出层误差
        output_error = self.targets - final_outputs
        return -output_error * final_outputs * (1 - final_outputs) @ np.transpose(hidden_outputs)
        pass

    def gfun_wih(self, wih):
        # 计算隐藏层输入
        hidden_inputs = wih @ self.inputs
        # 计算隐藏层输出
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算输出层输入
        final_inputs = self.who @ hidden_outputs
        # 计算输出层输出
        final_outputs = self.activation_function(final_inputs)
        # 计算输出层误差
        output_error = self.targets - final_outputs
        # 计算隐藏层误差
        hidden_errors = self.who.T @ output_error

        return -hidden_errors * hidden_outputs * (1 - hidden_outputs) @ np.transpose(self.inputs)

    def fun_who(self, who):
        # 计算隐藏层输入
        hidden_inputs = self.wih @ self.inputs
        # 计算隐藏层输出
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算输出层输入
        final_inputs = who @ hidden_outputs
        # 计算输出层输出
        final_outputs = self.activation_function(final_inputs)
        # 计算输出层误差
        output_error = self.targets - final_outputs
        return -1 / 2 * len(output_error) * np.sum([output_error[i] ** 2 for i in range(len(output_error))])

    def fun_wih(self, wih):
        # 计算隐藏层输入
        hidden_inputs = wih @ self.inputs
        # 计算隐藏层输出
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算输出层输入
        final_inputs = self.who @ hidden_outputs
        # 计算输出层输出
        final_outputs = self.activation_function(final_inputs)
        # 计算输出层误差
        output_error = self.targets - final_outputs
        # 计算隐藏层误差
        hidden_errors = self.who.T @ output_error
        return -1 / 2 * len(hidden_errors) * np.sum([hidden_errors[i] ** 2 for i in range(len(hidden_errors))])

    def G_who(self, who):
        # 计算隐藏层输入
        hidden_inputs = self.wih @ self.inputs
        # 计算隐藏层输出
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算输出层输入
        final_inputs = who @ hidden_outputs
        # 计算输出层输出
        final_outputs = self.activation_function(final_inputs)
        # 计算输出层误差
        output_error = self.targets - final_outputs
        # 计算输出层误差平方之和
        Sum_outputerror = -1 / 2 * len(output_error) * np.sum([output_error[i] ** 2 for i in range(len(output_error))])
        return final_outputs @ final_outputs.T * (1 - final_outputs) * np.dot(hidden_outputs.T, hidden_outputs) * (
                final_outputs * (Sum_outputerror - 1) + 1).T

    def G_wih(self, wih):
        # 计算隐藏层输入
        hidden_inputs = wih @ self.inputs
        # 计算隐藏层输出
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算输出层输入
        final_inputs = self.who @ hidden_outputs
        # 计算输出层输出
        final_outputs = self.activation_function(final_inputs)
        # 计算输出层误差
        output_error = self.targets - final_outputs
        # 计算隐藏层误差
        hidden_errors = self.who.T @ output_error
        # 计算隐藏层误差平方之和
        Sum_hiddenerror = -1 / 2 * len(hidden_errors) * np.sum(
            [hidden_errors[i] ** 2 for i in range(len(hidden_errors))])
        return hidden_outputs @ hidden_outputs.T * (1 - hidden_outputs) * self.inputs @ self.inputs.T * (
                hidden_outputs * (Sum_hiddenerror - 1) + 1)


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
    beta = [3,2,1]
    y = beta[0] * X * X + beta[1] * np.random.rand(X.size) * X + beta[2]
    targets = (y / (max(y) - min(y)) * 0.99) + .01

    # targets = (y / (max(y) - min(y)) * 0.99) + .01
    #targets = np.array([0.4, 0.5, 0.9])

    # # 画出原始图像
    # plt.scatter(x, y)
    # plt.show()

    # 设置学习率
    n.setlearnning(lr)
    for e in range(epochs):
        n.train(inputs, targets)
    #打印实际值
    print("实际值（归一化后数据）",targets)
    # 获得预测值
    print("预测参数为：",n.query(inputs))
    # 画出原始图像
    plt.scatter(X, y)
    plt.show()
    # 画预测图
    res = n.query(inputs)
    X = np.linspace(-1, 1, 100)
    plt.plot(X, res, color='r')
    plt.show()
def demo_linearfit2(lr,epoch=100):
    # 构造神经网络
    inputnodes = 1
    hidennodes = 5
    outputnodes = 1

    n = neuralNetwork(inputnodes, hidennodes, outputnodes, lr)
    #输出此时的神经网络权重
    print("训练前：\nwih层的权重为{},\nwho的权重矩阵为{}".format(n.get_wh()[0],n.get_wh()[1]))
    # 训练神经网络
    # 归一化输入输出
    x = np.linspace(-1, 1, epoch)
    X = np.asfarray(x)
    inputs = (X / (max(X) - min(X)) * 0.99) + .01

    # 归一化标签
    beta = [3,2,1]
    y = beta[0] * X * X + beta[1] * np.random.rand(X.size) * X + beta[2]
    targets = (y / (max(y) - min(y)) * 0.99) + .01
    #用10个数据进行训练
    n.setlearnning(lr)
    for i in range(epoch):
        n.train(inputs[i],targets[i])
    print("训练后：\nwih层的权重为{},\nwho的权重矩阵为{}".format(n.get_wh()[0], n.get_wh()[1]))
    #用1000个数据进行拟合
    x_query = np.linspace(-1,1,1000)
    x_query = np.asfarray(x_query)
    query_inputs = (x_query / (max(x_query) - min(x_query)) * 0.99) + .01
    res = [float(n.query(j)) for j in query_inputs]
    # 画出原始图像
    plt.scatter(inputs, targets)
    #画出预测曲线
    plt.plot(x_query,res,color='r')
    plt.show()

if __name__ == '__main__':
    # demo_linearfit(0.01, 20)
    demo_linearfit2(0.9,2)




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
