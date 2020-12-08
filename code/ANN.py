import numpy as np
# 导入sigmod函数
import scipy.special as ss
import matplotlib.pyplot as plt
from NN import neuralNetwork
from data_create import DataCreater

"""预测器，将全部是训练数据的均方根误差计算出来，再反向传播，迭代n次"""


class ANN(neuralNetwork):
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        super().__init__(inputnodes, hiddennodes, outputnodes, learningrate)
        # 初始化隐藏层偏置,偏执设为0试一下
        self.hidden_bias = np.zeros_like(self.hnodes, 1)
        # 初始化输出层偏置
        self.output_bias = np.zeros_like(self.onodes, 1)
        # 定义激活函数反函数
        self.inverse_activation_function = lambda x: ss.logit(x)
        # 定义误差列表
        self.losses = []

    def train_forward(self, input_list, target_list):
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
        final_outputs = final_inputs
        # 计算输出层误差
        output_error = targets - final_outputs
        # 将误差保存
        self.losses.append(output_error)
        # 将最后一次的隐藏层输出保存
        self.last_hidden_outputs = hidden_outputs
        # 保留最后一次输入
        self.last_inputs = inputs

    # 反向传播过程
    def train_backword(self, i):
        # 计算均方根误差
        self.sum_losses = sum(np.array(self.losses) ** 2) / (2 * len(self.losses))
        # if j % 100 == 0:
        #     sum_losses = sum(np.array(self.losses) ** 2) / (2 * len(self.losses))
        #     self.get_sumlosses()
        #     print("第{}次均方误差为: {}".format(j,sum_losses))

        # 计算局方误差的导数
        self.losses = sum(self.losses) / len(self.losses)

        # 更新who
        self.who += i * (self.lr+np.random.random()) * self.losses * np.transpose(self.last_hidden_outputs)
        # 更新ob
        self.output_bias += i * self.lr * self.losses
        # 将误差分配到隐藏层上
        hidden_errors = i * self.who.T @ self.sum_losses
        # 更新wih
        jacobin = np.dot((hidden_errors * self.last_hidden_outputs * (1 - self.last_hidden_outputs)),
                         np.transpose(self.last_inputs))
        self.wih += i * self.lr * jacobin
        # 更新hb
        grid = hidden_errors * self.last_hidden_outputs * (1 - self.last_hidden_outputs)
        self.hidden_bias += i * self.lr * grid
        # 导数归零
        self.losses = []

    # 训练函数
    def train(self, input_list, target_list):
        pass

    def update_who(self, output_error, hidden_outputs, max_iter=1):
        # self.who += self.lr * np.dot((output_error * final_inputs * (1-final_outputs)),np.transpose(hidden_outputs))
        # 使用普范数
        jacobin = output_error * np.transpose(hidden_outputs)
        count = 0
        while np.linalg.norm(jacobin, 2) > 10 ** (-3) and count <= max_iter:
            jacobin = output_error * np.transpose(hidden_outputs)
            self.who += -self.lr * jacobin
            count += 1
            # print("who 谱范数：{}".format(np.linalg.norm(jacobin, 2)))

    def update_wih(self, hidden_errors, hidden_outputs, inputs, max_iter=1):
        jacobin = np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                         np.transpose(inputs))
        count = 0
        while np.linalg.norm(jacobin, 2) > 10 ** (-3) and count <= max_iter:
            jacobin = np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), np.transpose(inputs))
            self.wih += -self.lr * jacobin
            count += 1
            # print("wih 谱范数：{}".format(np.linalg.norm(jacobin, 2)))

    def update_bo(self, output_error, max_iter=1):
        grid = output_error
        count = 0
        while np.linalg.norm(grid, 2) > 10 ** (-3) and count <= max_iter:
            grid = output_error
            self.output_bias += -self.lr * grid
            count += 1
            # print("bih 范数：{}".format(np.linalg.norm(grid, 2)))

    def update_bh(self, hidden_errors, hidden_outputs, max_iter=1):
        grid = hidden_errors * hidden_outputs * (1 - hidden_outputs)
        count = 0
        while np.linalg.norm(grid, 2) > 10 ** (-3) and count <= max_iter:
            grid = hidden_errors * hidden_outputs * (1 - hidden_outputs)
            self.hidden_bias += -self.lr * grid
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
        # final_outputs = self.activation_function(final_inputs)
        # 返回最终的输出
        final_outputs = final_inputs
        return final_outputs

    # 添加一个输出偏置的函数
    def get_bias(self):
        return (self.hidden_bias, self.output_bias)
    # 修改学习率
    def set_lr(self,lr):
        self.lr = lr

    def get_sumlosses(self):
        return self.sum_losses


"""# 大坑！默认参数必须指向不可变对象！！！！"""


def sampling_function(x=np.linspace(0, 60, 50), beta=None):
    if beta == None:
        beta = [1, 2, 3]
    return beta[0] * x + beta[1] * x ** 2 + beta[2]


# 获取采样的值
def get_sample(num):
    x = np.linspace(0.1, 1, 1000)

    sample_x = np.random.choice(x, num, replace=True)  # F代表有放回，T代表无放回
    sample_y = sampling_function(sample_x)
    return (sample_x, sample_y)


# # 定义反函数
# def inverse_function(y, beta=None):
#     if beta == None:
#         beta = [2, 1,3]
#     x = y - beta[2]
#     return x
def 失败品():
    inputnodes = 1
    hiddennodes = 100
    outputnodes = 1
    learningrate = 0.008
    epoch = 15
    np.random.seed(3)
    n = ANN(inputnodes, hiddennodes, outputnodes, learningrate)
    # 获得训练前的权重矩阵
    # print("训练前：\nih层的权重为{},\nho的权重矩阵为{}".format(n.get_wh()[0], n.get_wh()[1]))
    # 获取训练前的偏置向量
    # print("训练前：\nih层的偏置为{},\nho层的偏置为{}".format(n.get_bias()[0], n.get_bias()[1]))
    """# 训练神经网络"""
    # 获得不重复采样数据的x,y值
    x_sampledata, y_sampledata = get_sample(epoch)
    print("采样的x：{}\ny:{}".format(x_sampledata[:10], y_sampledata[:10]))
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
    # 打印最后一次训练的误差
    print("打印最后一次训练的误差:", n.loss)
    # 打印最后一次隐藏层总和误差
    print("打印最后一次隐藏层总和误差:", n.hidden_loss)

    # 构造查询输入数据
    x = np.linspace(-1, 1, 1000)
    # 归一化
    # x = (x_sapmpledata / (max(x_sapmpledata) - min(x_sapmpledata)) * 0.99) + .01
    x_query = np.asfarray(x)
    # 查询输出
    y_query = np.array([float(n.query(j)) for j in x_query])
    # 答应输出
    print("查询输出", y_query[:10])
    # 反激活函数输出
    # y_query = [ss.logit(i) for i in y_query]
    # 打印反激活函数输出
    print("反激活函数输出：", y_query[:10])
    # 打印标签
    print("标签", target[:10])

    fig0 = plt.figure(figsize=(4, 4))
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

    # 单独输出采样数据的预测值
    fig4 = plt.figure(figsize=(4, 4))
    # 预测值-sigmod
    fit_value = np.array([ss.expit(n.query(j)) for j in x_sampledata])
    plt.scatter(x_sampledata, fit_value, color='g')
    # 实际值-sigmod
    plt.scatter(x_sampledata, target, color='r')
    # 打印训练产生的误差
    error_train = np.array([ss.expit(i) for i in y_sampledata]) - np.array([ss.expit(i) for i in fit_value])
    print("打印训练产生的误差:", error_train ** 2)
    # # 打印实际的误差
    # error_true =


def linearfit(lr, epoch):
    # 构造神经网络
    inputnodes = 1
    hiddennodes = 100
    outputnodes = 1
    learningrate = lr
    n = ANN(inputnodes, hiddennodes, outputnodes, learningrate)
    # 获得训练前的权重矩阵
    # print("训练前：\nih层的权重为{},\nho的权重矩阵为{}".format(n.get_wh()[0], n.get_wh()[1]))
    # 获取训练前的偏置向量
    # print("训练前：\nih层的偏置为{},\nho层的偏置为{}".format(n.get_bias()[0], n.get_bias()[1]))
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

    fig0 = plt.figure(figsize=(4, 4))
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
    # 输入与输出不做变换，输出层无激活函数
    """初始化神经网络"""
    inputnodes = 1
    hiddennodes = 5
    outputnodes = 1
    learningrate = 0.05
    np.random.seed(1)
    n = ANN(inputnodes, hiddennodes, outputnodes, learningrate)

    """获取训练数据"""
    num = 15# 15个10个神经元，0.01拟合的不错
    data = DataCreater(num)  # 获取50个
    # 将x归一化
    data.normliza_x()
    x, y = data.get_data()

    """
    训练
    
    训练n个数据，获取到总误差（均方误差），进行一次反向传播，此为完成一次迭代.
    第二次再将n个数据带入，再计算一次误差，再反向传播，迭代m次.
    """
    m = 20000
    losses_lis = []
    for j in range(1,m+1):
        for i in range(num):
            n.train_forward(x[i], y[i])
        # 衰减学习率
        dacy = 0.5
        learningrate = learningrate/(1+dacy * j)
        # n.set_lr(learningrate)
        if j % 100 == 0:  # 每100步检查一次误差是否稳定
            print("第{}均方误差为：{}".format(j,n.get_sumlosses()))
        #     losses_lis.append(n.get_sumlosses())
        #     if (len(losses_lis) > 2) and (losses_lis[-1] - losses_lis[-2] < 0.00001):
        #         learningrate = learningrate * 1.5
        #         n.set_lr(learningrate)
        #     if (len(losses_lis) > 2) and (losses_lis[-1] - losses_lis[-2] >= 0.00001):
        #         learningrate = learningrate * .09
        #         n.set_lr(learningrate)
        n.train_backword(i=1)



    """查询"""
    # 生成1000个归一化后的数据
    query = DataCreater(num)
    # query.normliza_x()
    x_query, _ = query.get_data()
    y_query = [float(n.query(j)) for j in x_query]
    fig = plt.figure(figsize=(4, 4))
    # 画出预测图像
    plt.plot(x_query, y_query, color='r')
    plt.show()
    # 画出原始图像
    query.plot_data()
