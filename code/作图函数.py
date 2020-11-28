import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)#导入中文字体

x=[1,2,3,4]
y=[12,3,5,7]
plt.plot(x,y,'go-')
plt.grid(True)
plt.xlabel("显示",FontProperties=font)