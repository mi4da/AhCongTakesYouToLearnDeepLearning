import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class DataCreater:
    def __init__(self, num=None):
        if num != None:
            self.num = num
        else:
            self.num = 800
        self.x = np.arange(0, 5, 5 / self.num)
        self.y = (self.x - 2.5) ** 2

    def normliza_x(self):
        self.x = self.x / self.x.max() + .01  # 防止梯度消失

    def normlize_xy(self):
        self.x = self.x / self.x.max()
        self.y = self.y / self.y.max()

    def get_data(self):
        return (self.x, self.y)

    def make_random(self):
        self.y = self.y + np.random.randint(1, 5, len(self.x))

    def plot_data(self):
        plt.scatter(self.x, self.y, color='b')
        plt.plot(self.x, self.y, color='b')
        plt.show()

    def get_csv(self):
        data = pd.read_csv("./concrete.csv")
        df = data.copy()
        # 随机抽取一些数据
        df = df.sample(frac=1.0, random_state=1)  # 全部打乱
        cut_idx = int(round(0.75 * df.shape[0]))  # 75%作为训练数据
        df_train, df_test = df.iloc[:cut_idx], df.iloc[cut_idx:]
        # 分割数组
        self.df_train_x = df_train[['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg',
                                    'fineagg', 'age']].values
        self.df_train_y = df_train[['strength']].values

        self.df_test_x = df_test[['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg',
                                  'fineagg', 'age']].values
        self.df_test_y = df_test[['strength']].values

        return (len(df_train),len(df_test))  # 返回训练数据长度和测试数据长度

    # 将数据全部在这里处理完
    def get_test_x(self, i):
        self.x = self.df_test_x
        # 归一化
        self.normliza_x()
        return self.x[i].reshape(8, 1)

    def get_train_x(self, i):
        self.x = self.df_train_x
        # 归一化
        self.normliza_x()
        return self.x[i].reshape(8, 1)

    def get_test_y(self, i):
        self.y = self.df_test_y
        return self.y[i]

    def get_train_y(self, i):
        self.y = self.df_train_y
        return self.y[i]


if __name__ == '__main__':
    demo = DataCreater(10)
    demo.plot_data()
    demo.normliza_x()
    print(demo.get_data())
