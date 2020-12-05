import numpy as np
# 导入sigmod函数
import scipy.special as ss
import matplotlib.pyplot as plt
from NN import neuralNetwork


class ANN(neuralNetwork):
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        super().__init__(inputnodes, hiddennodes, outputnodes, learningrate)
        # 初始化隐藏层偏置
        self.hidden_bias = (np.random.normal(0.0, np.random.randint(1, 10, 1), (self.hnodes, 1)))
        # 初始化输出层偏置
        self.output_bias = (np.random.normal(0.0, np.random.randint(1, 10, 1), (self.onodes, 1)))
        # 定义激活函数反函数
        self.inverse_activation_function = lambda x: ss.logit(x)

    def train(self, input_list, target_list):
        # 构造目标矩阵
        targets = np.array(target_list, ndmin=2).T  # 最小维数为2，列向量
        # 构建输入矩阵
        inputs = np.array(input_list, ndmin=2).T
        # 计算隐藏层的输入
        hidden_inputs = self.wih @ inputs + self.hidden_bias
        # hidden_inputs = self.wih @ inputs
        # print(hidden_inputs)
        # 计算隐藏层输出
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算输出层输入
        final_inputs = self.who @ hidden_outputs + self.output_bias
        # final_inputs = self.who @ hidden_outputs
        # 计算输出层输出
        final_outputs = self.activation_function(final_inputs)
        # 计算输出层误差
        output_error = targets - final_outputs
        # 计算隐藏层误差
        hidden_errors = self.who.T @ output_error
        # 更新隐藏层与输入层的权重
        """
        这里的梯度下降法所使用的不是”梯度向量“而是”雅克比矩阵“，
        参考：https://blog.csdn.net/liuliqun520/article/details/80019507
        不过一样有办法计算模长，既然要求的是MSE也就是军方根误差最小，而雅克比矩阵的每一行都是函数对自变量求一阶偏导的向量的转置，
        那么就相当于雅克比矩阵的范数就是每个独立向量的和的模长（不是知道这些向量是否在同一空间里能否直接加和，但根据向量的平移性，应该是可以直接加和的）  
        或者我们就用最原始lossfuction来规范，画出图像来那样子。
        
        """
        self.update_who(output_error, final_outputs, hidden_outputs, max_iter=1000)
        # 更新输入层与隐藏层的权重
        self.update_wih(hidden_errors, hidden_outputs, inputs, max_iter=1000)
        # 更新隐藏层的偏置
        self.update_bh(hidden_errors, hidden_outputs, max_iter=1000)
        # 更新输出层的偏置
        self.update_bo(output_error, final_outputs, max_iter=1000)
        # 计算军方根误差
        self.loss = float(output_error ** 2)
        # 计算隐藏层的总误差
        self.hidden_loss = sum(hidden_errors ** 2)

    def update_who(self, output_error, final_outputs, hidden_outputs, max_iter=1):
        # self.who += self.lr * np.dot((output_error * final_inputs * (1-final_outputs)),np.transpose(hidden_outputs))
        # 使用普范数
        jacobin = np.dot((output_error * final_outputs * (1 - final_outputs)),
                         np.transpose(hidden_outputs))
        count = 0
        while np.linalg.norm(jacobin, 2) > 10 ** (-3) and count <= max_iter:
            jacobin = np.dot((output_error * final_outputs * (1 - final_outputs)), np.transpose(hidden_outputs))
            self.who += self.lr * jacobin
            count += 1
            # print("who 谱范数：{}".format(np.linalg.norm(jacobin, 2)))

    def update_wih(self, hidden_errors, hidden_outputs, inputs, max_iter=1):
        jacobin = np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                         np.transpose(inputs))
        count = 0
        while np.linalg.norm(jacobin, 2) > 10 ** (-3) and count <= max_iter:
            jacobin = np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), np.transpose(inputs))
            self.wih += self.lr * jacobin
            count += 1
            # print("wih 谱范数：{}".format(np.linalg.norm(jacobin, 2)))

    def update_bo(self, output_error, final_outputs, max_iter=1):
        grid = output_error * final_outputs * (1 - final_outputs)
        count = 0
        while np.linalg.norm(grid, 2) > 10 ** (-3) and count <= max_iter:
            grid = output_error * final_outputs * (1 - final_outputs)
            self.output_bias += self.lr * grid
            count += 1
            # print("bih 范数：{}".format(np.linalg.norm(grid, 2)))

    def update_bh(self, hidden_errors, hidden_outputs, max_iter=1):
        grid = hidden_errors * hidden_outputs * (1 - hidden_outputs)
        count = 0
        while np.linalg.norm(grid, 2) > 10 ** (-3) and count <= max_iter:
            grid = hidden_errors * hidden_outputs * (1 - hidden_outputs)
            self.hidden_bias += self.lr * grid
            count += 1
            # print("bho 范数：{}".format(np.linalg.norm(grid, 2)))

    # 重写查询函数（正向传播）
    def query(self, input_list):
        # 构建输入矩阵
        inputs = np.array(input_list, ndmin=2).T
        # 计算掩藏层输入
        hidden_inputs = self.wih @ inputs + self.hidden_bias
        # 计算隐藏层输出
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算输出层输入
        final_inputs = self.who @ hidden_outputs + self.output_bias
        # 计算输出层输出
        final_outputs = self.activation_function(final_inputs)
        # 返回最终的输出
        return final_outputs

    # 添加一个输出偏置的函数
    def get_bias(self):
        return (self.hidden_bias, self.output_bias)


"""# 大坑！默认参数必须指向不可变对象！！！！"""


def sampling_function(x=np.linspace(-1, 1, 1000), beta=None):
    if beta == None:
        beta = [1, 2, 3]
    return beta[0] * x + beta[1] * x ** 2 + beta[2]


# 获取采样的值
def get_sample(num):
    x = np.linspace(-1, 1, 1000)

    sample_x = np.random.choice(x, num, replace=True)
    sample_y = sampling_function(sample_x)
    return (sample_x, sample_y)


# # 定义反函数
# def inverse_function(y, beta=None):
#     if beta == None:
#         beta = [2, 1,3]
#     x = y - beta[2]
#     return x


def linearfit(lr, epoch):
    # 构造神经网络
    inputnodes = 1
    hiddennodes = 100
    outputnodes = 1
    learningrate = lr
    n = ANN(inputnodes, hiddennodes, outputnodes, learningrate)
    # 获得训练前的权重矩阵
    print("训练前：\nih层的权重为{},\nho的权重矩阵为{}".format(n.get_wh()[0], n.get_wh()[1]))
    # 获取训练前的偏置向量
    print("训练前：\nih层的偏置为{},\nho层的偏置为{}".format(n.get_bias()[0], n.get_bias()[1]))
    """# 训练神经网络"""
    # 获得采样数据的x,y值
    x_sampledata, y_sampledata = get_sample(epoch)
    # 归一化输入？
    # x = (x_sampledata / (max(x_sapmpledata) - min(x_sapmpledata)) * 0.99) + .01
    # sigmod化标签\
    # target = (y_sampledata / (max(y_sampledata) - min(y_sampledata)) * 0.99) + .01
    target = np.array([ss.expit(i) for i in y_sampledata])

    # 不归一化
    x = x_sampledata
    # 不归一化标签
    # target = y_sampledata
    # 训练神经网络
    losses = []
    hidden_losses = []
    for i in range(epoch):
        n.train(x[i], target[i])
        # 训练第i次的权重矩阵
        # print("训练第{}次：\nih层的权重为{},\nho的权重矩阵为{}".format(i + 1, n.get_wh()[0], n.get_wh()[1]))
        # 获取训练第i次的偏置向量
        # print("训练第{}次：\nih层的偏置为{},\nho层的偏置为{}".format(i + 1, n.get_bias()[0], n.get_bias()[1]))
        losses.append(n.loss)
        # 获得隐藏层总误差
        hidden_losses.append(n.hidden_loss)

    # 构造查询输入数据
    x = np.linspace(-1, 1, 1000)
    # 归一化
    # x = (x_sapmpledata / (max(x_sapmpledata) - min(x_sapmpledata)) * 0.99) + .01
    x_query = np.asfarray(x)
    # 查询输出
    y_query = np.array([float(n.query(j)) for j in x_query])
    # 答应输出
    print("查询输出", y_query)
    # 反激活函数输出
    y_query = [ss.logit(i) for i in y_query]
    # 打印反激活函数输出
    print("反激活函数输出：", y_query)
    # 打印标签
    print("标签", target)

    fig0 = plt.figure(figsize=(4,4))
    # 画出原函数图像
    plt.plot(x, sampling_function(x), color='b')

    # 画出采样数据点
    plt.scatter(x_sampledata, y_sampledata)

    # 画出预测函数
    plt.plot(x, y_query, color='r')
    plt.show()

    # 画出输出层误差曲线
    fig1 = plt.figure(figsize=(4, 4))
    plt.plot([i for i in range(len(losses))], losses)
    plt.title("output_losses")
    plt.show()

    # 画出隐藏层误差曲线
    fig2 = plt.figure(figsize=(4, 4))
    plt.plot([i for i in range(len(hidden_losses))], hidden_losses)
    plt.title("hidden_losses")
    plt.show()

    # 单独画出预测函数
    fig3 = plt.figure(figsize=(4, 4))
    plt.plot(x, y_query, color='r')
    plt.title('only_fit')
    plt.show()


if __name__ == '__main__':
    linearfit(0.008, 100)
