import numpy as np
# 导入sigmod函数
import scipy.special as ss
import matplotlib.pyplot as plt
from NN import neuralNetwork


class ANN(neuralNetwork):
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        super().__init__(inputnodes, hiddennodes, outputnodes, learningrate)
        # 初始化输入层-隐藏层偏置
        self.input_hidden_bias = (np.random.normal(0.0, 1.0, (self.hnodes, 1)))
        # 初始化隐藏层-输出层偏置
        self.hidden_output_bias = (np.random.normal(0.0, 1.0, (self.onodes, 1)))

    def train(self, input_list, target_list):
        # 构造目标矩阵
        targets = np.array(target_list, ndmin=2).T  # 最小维数为2，列向量
        # 构建输入矩阵
        inputs = np.array(input_list, ndmin=2).T
        # 计算隐藏层的输入
        hidden_inputs = self.wih @ inputs + self.input_hidden_bias
        # print(hidden_inputs)
        # 计算隐藏层输出
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算输出层输入
        final_inputs = self.who @ hidden_outputs + self.hidden_output_bias
        # 计算输出层输出
        final_outputs = self.activation_function(final_inputs)
        # 计算输出层误差
        output_error = targets - final_outputs
        # 计算隐藏层误差
        hidden_errors = self.who.T @ output_error
        # 更新隐藏层与输入层的权重
        self.update_who(output_error, final_outputs, hidden_outputs)
        # 更新输入层与隐藏层的权重
        self.update_wih(hidden_errors, hidden_outputs, inputs)
        # 更新隐藏层与输入层的偏置
        self.update_bih(output_error, final_outputs)
        # 更新输入层与隐藏层的偏置
        self.update_bho(hidden_errors, hidden_outputs)

    def update_who(self, output_error, final_outputs, hidden_outputs):
        # self.who += self.lr * np.dot((output_error * final_inputs * (1-final_outputs)),np.transpose(hidden_outputs))
        # 使用普范数
        jacobin = np.dot((output_error * final_outputs * (1 - final_outputs)),
                         np.transpose(hidden_outputs))
        count = 0
        while np.linalg.norm(jacobin, 2) <= 10 ** (-5) and count <= 1000:
            jacobin = np.dot((output_error * final_outputs * (1 - final_outputs)), np.transpose(hidden_outputs))
            self.who += self.lr * jacobin
            count += 1
            print("who 谱范数：{}".format(np.linalg.norm(jacobin, 2)))

    def update_wih(self, hidden_errors, hidden_outputs, inputs):
        jacobin = np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                         np.transpose(inputs))
        count = 0
        while np.linalg.norm(jacobin, 2) <= 10 ** (-5) and count <= 1000:
            jacobin = np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), np.transpose(inputs))
            self.wih += self.lr * jacobin
            count += 1
            print("wih 谱范数：{}".format(np.linalg.norm(jacobin, 2)))

    def update_bih(self, output_error, final_outputs):
        grid = output_error * final_outputs * (1 - final_outputs)
        count = 0
        while np.linalg.norm(grid, 2) <= 10 ** (-5) and count <= 1000:
            grid = output_error * final_outputs * (1 - final_outputs)
            self.input_hidden_bias += self.lr * grid
            count += 1
            print("bih 范数：{}".format(np.linalg.norm(grid, 2)))

    def update_bho(self, hidden_errors, hidden_outputs):
        grid = hidden_errors * hidden_outputs * (1 - hidden_outputs)
        count = 0
        while np.linalg.norm(grid, 2) <= 10 ** (-5) and count <= 1000:
            grid = hidden_errors * hidden_outputs * (1 - hidden_outputs)
            self.hidden_output_bias += self.lr * grid
            count += 1
            print("bho 范数：{}".format(np.linalg.norm(grid, 2)))

    # 重写查询函数（正向传播）
    def query(self, input_list):
        # 构建输入矩阵
        inputs = np.array(input_list, ndmin=2).T
        # 计算掩藏层输入
        hidden_inputs = self.wih @ inputs + self.input_hidden_bias
        # 计算隐藏层输出
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算输出层输入
        final_inputs = self.who @ hidden_outputs + self.hidden_output_bias
        # 计算输出层输出
        final_outputs = self.activation_function(final_inputs)
        # 返回最终的输出
        return final_outputs

    # 添加一个输出偏置的函数
    def get_bias(self):
        return (self.input_hidden_bias, self.hidden_output_bias)


"""# 大坑！默认参数必须指向不可变对象！！！！"""


def sampling_function(x=np.linspace(-1, 1, 1000), beta=None):
    if beta == None:
        beta = [2, 1]
    return beta[0] + beta[1] * x


# 获取采样的值
def get_sample(num):
    y = sampling_function()
    sample_y = np.random.choice(y, num, replace=True)
    return sample_y


# 定义反函数
def inverse_function(y, beta=None):
    if beta == None:
        beta = [2, 1]
    x = (y - beta[0]) / beta[1]
    return x


def linearfit(lr, epoch):
    # 构造神经网络
    inputnodes = 1
    hiddennodes = 10
    outputnodes = 1
    learningrate = lr
    n = ANN(inputnodes, hiddennodes, outputnodes, learningrate)
    # 获得训练前的权重矩阵
    print("训练前：\nih层的权重为{},\nho的权重矩阵为{}".format(n.get_wh()[0], n.get_wh()[1]))
    # 获取训练前的偏置向量
    print("训练前：\nih层的偏置为{},\nho层的偏置为{}".format(n.get_bias()[0], n.get_bias()[1]))
    """# 训练神经网络"""
    # 获得10个采样数据
    y_sampledata = get_sample(10)
    # 获得采样数据的反函数值
    x_sapmpledata = inverse_function(y_sampledata)
    # 归一化输入？
    # 归一化标签
    # 训练神经网络
    # 构造查询输入数据
    # 查询输出
    # 反归一化输出
    # 画出原函数图像
    # 画出采样数据点
    # 画出预测函数


if __name__ == '__main__':
    pass
