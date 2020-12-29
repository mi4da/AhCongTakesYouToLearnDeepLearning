# d导入包
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as ss
from data_create import DataCreater


class FitANN:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        """定义初始化"""
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        """初始化权重，取均值为1，方差为下一层节点数的根"""
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        """初始化偏置，全为0"""
        self.hidden_bias = np.zeros((self.hnodes, 1))
        # self.final_bias = np.zeros((self.onodes, 1))
        """学习率"""
        self.lr = learningrate
        """激活函数"""
        self.activation_function = lambda x: ss.expit(x)
        """误差列表"""
        self.losses = []

    def __doc__(self):
        """
        函数形式：5 * np.sin(4*self.x) + 3
        训练数据：100
        学习率：0.05
        衰减率：0.001
        隐藏神经元数量：20
        最终误差：0.1721155
        拟合效果：第一个峰无法拟合，其余良好
        """
        pass

    def train(self, input, target, i=1):
        """正向传播过程"""
        # 输入层输入
        # inputs = np.array(input,ndmin=2).T # 1 * 1标量
        # targets = np.array(targetlist,ndmin=2).T # 1* 1 标量
        # 隐藏层输入
        hidden_inputs = self.wih @ input  # (10*1) = (10*8) @ (8*1) 括号代表矩阵/向量的形状。
        # 隐藏层输出
        hidden_outputs = self.activation_function(hidden_inputs + self.hidden_bias)  # （10*1） = （10*1）+（10*1）
        # 输出层输入
        final_inputs = self.who @ hidden_outputs  # (1,1) = (1*10) @ (10*1),（1，1）代表标量
        # 输出层输出
        final_outputs = final_inputs  # (1,1)
        # 获得误差
        final_error = target - final_outputs  # (1,1)
        # 添加误差
        self.losses.append(final_error)

        """反向传播"""
        # 更新who
        who_grid = final_error * hidden_outputs.T  # (1*10)
        self.who += i * self.lr * who_grid  # (1*10)
        # 更新wih (10*8) = (1,1) * (10*1) * (10*1) * (10*1) @ (1*8),这里的矩阵乘法不一样哦
        wih_grid = final_error * self.who.T * hidden_outputs * (1 - hidden_outputs) @ input.T
        self.wih += i * self.lr * wih_grid # (10*8)
        # 跟新hb(10*1) = (1,1) * (10*1) * (10*1) * (10*1)
        hb_grid = final_error * self.who.T * hidden_outputs * (1 - hidden_outputs)
        self.hidden_bias += i * self.lr * hb_grid

    def query(self, input):
        """正向传播过程"""
        # 输入层输入
        # =
        # 隐藏层输入
        hidden_inputs = self.wih @ input
        # 隐藏层输出
        hidden_outputs = self.activation_function(hidden_inputs + self.hidden_bias)
        # 输出层输入
        final_inputs = self.who @ hidden_outputs
        # 输出层输出
        final_outputs = final_inputs
        return final_outputs

    def get_sumlosses(self):
        sum_losses = sum(np.array(self.losses) ** 2) / (2 * len(self.losses))
        return sum_losses

    def set_lr(self, lr):
        self.lr = lr
    def get_lr(self):
        return self.lr

if __name__ == '__main__':
    """主逻辑"""
    np.random.seed(1)
    """初始化数据，x归一化，y正常"""
    # num = 100 # 训练数据数量,这里取全部
    data = DataCreater()  # 如果空值。意味着获取全部的数据
    # 获取文件数据，返回训练集与测试集长度
    num ,num_test= data.get_csv()
    """初始化神经网络"""
    inputnodes = 8  # 一共有八个特征
    hiddennodes = 30
    outputnodes = 1
    learningrate = 0.011156
    n = FitANN(inputnodes, hiddennodes, outputnodes, learningrate)
    """训练"""
    # 设置衰减学习率
    epoch = 10000  # 训练次数
    """动画"""
    plt.ion()
    loslis = []
    for j in range(epoch):
        for i in range(num):
            n.train(data.get_train_x(i),data.get_train_y(i))

        if j % 1 == 0:  # 每1次查询一次损失函数
            plt.cla()
            if len(n.losses) != 0:

                sumlosses = float(n.get_sumlosses())
                plt.text(0, sumlosses, "LOSS=%.4f" % sumlosses, fontdict={'size': 20, 'color': 'red'})
                plt.text(0, sumlosses+15, "lr=%.4f" % n.get_lr(), fontdict={'size': 15, 'color': 'blue'})
                plt.text(0, sumlosses + 30, "hidden_nodes=%d" % hiddennodes, fontdict={'size': 15, 'color': 'green'})
                loslis.append(sumlosses)
                plt.plot(range(len(loslis)), loslis, 'r')
                # 当loss维持在某个固定的水平时增大学习率
                if sumlosses <= 30:
                    if j % 100 == 0:
                        n.set_lr(learningrate + 0.001)
                else:
                    if j % 100 == 0:
                        n.set_lr(learningrate - 0.01)
                # print("第{}次均方误差为{}".format(j, n.get_sumlosses()))
            plt.show()
            plt.pause(0.01)



        # 误差归零
        n.losses = []
    plt.ioff()
    """查询"""
    llosses = []
    for j in range(num_test):
        y_query = float(n.query(data.get_test_x(j)))
        y_real = data.get_test_y(j)
        llosses.append(1/2 * (y_query - y_real) ** 2)
    print("测试集误差平方和：",sum(llosses))


    # """画图"""
    # fig = plt.figure(figsize=(4, 4))
    # # 画出预测图像
    # plt.plot(x_query, y_query, color='r')
    # plt.show()
    # # # 画出原始图像
    # data.plot_data()
