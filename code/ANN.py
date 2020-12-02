import numpy as np
# 导入sigmod函数
import scipy.special as ss
import matplotlib.pyplot as plt
from NN import neuralNetwork


class ANN(neuralNetwork):
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        super().__init__(inputnodes, hiddennodes, outputnodes, learningrate)
        # 初始化输入层-隐藏层偏置
        self.input_hidden_bias = (np.random.normal(0.0,1.0,(self.hnodes,1)))
        # 初始化隐藏层-输出层偏置
        self.hidden_output_bias = (np.random.normal(0.0,1.0,(self.onodes,1)))

    def train(self, input_list, target_list):
        # 构造目标矩阵
        targets = np.array(target_list,ndmin=2).T # 最小维数为2，列向量
        # 构建输入矩阵
        inputs = np.array(input_list,ndmin=2).T
        # 计算隐藏层的输入
        hidden_inputs = self.wih @ inputs + self.input_hidden_bias
        #print(hidden_inputs)
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
        self.update_who(output_error,final_outputs,final_inputs,hidden_outputs)


        # 更新输入层与隐藏层的权重
        self.update_wih(hidden_errors,hidden_outputs,hidden_inputs,inputs)
        # 更新隐藏层与输入层的偏置
        # 更新输入层与隐藏层的偏置

    def update_who(self, output_error, final_outputs, final_inputs, hidden_outputs):
        # self.who += self.lr * np.dot((output_error * final_inputs * (1-final_outputs)),np.transpose(hidden_outputs))
        grid = self.lr * np.dot((output_error * final_inputs * (1-final_outputs)),np.transpose(hidden_outputs))
        

    def update_wih(self, hidden_errors, hidden_outputs, hidden_inputs, inputs):
        pass


if __name__ == '__main__':
    inputnodes = 1
    hiddennodes = 10
    outputnodes = 1
    learningrate = 0.1
    demo = ANN(inputnodes, hiddennodes, outputnodes, learningrate)
    demo.train(-1,1)
