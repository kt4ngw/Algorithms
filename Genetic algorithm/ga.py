import numpy as np
import matplotlib.pyplot as plt
class Ga():
    def __init__(self, populationSize=50, crossRate=0.5, mutateRate=0.01, generations=500, choreLength=19):

        self.populationSize = populationSize # 种群大小
        self.crossRate = crossRate # 交叉概率
        self.mutateRate = mutateRate # 变异概率
        self.generations = generations # 进化代数
        self.choreLength = choreLength # 染色体长度
        self.population = np.random.randint(0, 2, size=(populationSize, choreLength,)) # 初始化种群


    def decoding(self, population):
        # 把二进制数组转化成在定义域范围内的数
        newArr = np.zeros(shape=(self.populationSize, self.choreLength))
        for i in range(self.choreLength):
            newArr[:,i] = 2 ** (self.choreLength - 1 - i) * population[:,i]
        newArr = -1 + (2.5 - -1) * np.sum(newArr, axis=1) / ((2 ** self.choreLength) - 1)
        return newArr

    def selection(self, population):
        # 选择种群
        newArr = self.decoding(population)
        fitness = self.fitness(newArr)
        fitness = fitness / fitness.sum()
        idx = np.random.choice(np.array(list(range(population.shape[0]))), size=population.shape[0],
                                  p=fitness)
        return population[idx]

    def fitness(self, newArr):
        # 适应度函数 一般为优化函数
        return newArr * np.cos(5 * np.pi * newArr) + 3.5 # 在此处修改优化函数

    def crossover(self, population, crossRate):
        # 交叉
        for i in range(0, population.shape[0], 2):
            xa = population[i, :]
            xb = population[i + 1, :]
            for j in range(population.shape[1]):
                if np.random.rand() <= crossRate:
                    xa[j], xb[j] = xb[j], xa[j]
            population[i, :] = xa
            population[i + 1, :] = xb
        return population

    def mutation(self, population, mutateRate):
        # 变异
        for i in range(population.shape[0]):
            for j in range(population.shape[1]):
                if np.random.rand() <= mutateRate:
                    population[i, j] = (population[i, j] + 1) % 2
        return population


def main():
    ga = Ga()
    for i in range(ga.generations):
        print('the {} generation:'.format(i + 1))
        newPopulation = ga.selection(population=ga.population)
        newPopulation = ga.crossover(population=newPopulation, crossRate=ga.crossRate)
        newPopulation = ga.mutation(population=newPopulation, mutateRate=ga.mutateRate)
        ga.population = newPopulation
    print(ga.decoding(ga.population))
    x = np.arange(-1, 2.6, 0.1)
    y = ga.fitness(x)
    plt.plot(x, y, color="green")
    X = ga.decoding(ga.population)
    Y = ga.fitness(X)
    plt.scatter(X, Y, color='red')
    plt.show()

if __name__ == '__main__':
    main()













