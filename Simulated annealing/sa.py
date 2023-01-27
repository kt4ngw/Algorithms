import numpy as np
import math
import matplotlib.pyplot as plt
class Sa():
    def __init__(self, maxT=100, minT=0.5, k=100, ):
        self.maxT = maxT # 最高温度
        self.minT = minT # 最低温度
        self.solution = np.random.uniform(-1, 2.5) # 生成初始解
        self.k = k # 每个温度变化中迭代次数

    def createNewSolution(self, solution):
        newSolution = solution + np.random.uniform(-0.0015, 0.0015) * self.maxT
        return newSolution

def func(x):
    return x * np.cos(5 * np.pi * x) + 3.5

def main():
    sa = Sa()
    t = 0
    while(sa.maxT > sa.minT):
        for i in range(sa.k):
            y = func(sa.solution)
            newSolution = sa.createNewSolution(sa.solution)
            if(newSolution  <= -1 and newSolution  >= 2.5):
                newY = func(newSolution)
                if newY > y:
                    sa.solution = newSolution
                else:
                    pt = math.exp(-(abs(newSolution - y)) / sa.maxT)
                    r = np.random.uniform(0, 1)
                    if r < pt:
                        sa.solution = newSolution
        t = t + 1
        sa.maxT = 100 * 0.7 ** t

    # print(sa.solution)
    x = np.arange(-1, 2.5, 0.1)
    y = func(x)
    plt.plot(x, y, color="green")
    X = sa.solution
    Y = func(sa.solution)
    plt.scatter(X, Y, color='red')
    plt.show()



if __name__ == '__main__':
    main()

