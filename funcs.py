import numpy as np

def sphere(array):
    return np.sum(np.square(array))

def tablet(array):
    return 1000000*np.square(array[0]) + np.sum(np.square(array[1:]))

def cigar(array):
    return np.square(array[0]) + 1000000*np.sum(np.square(array[1:]))

def rosenbrock(array):#m=0, sigma=2
    ans = 0
    for i in range(len(array)-1):
        ans += 100*(array[i+1]-array[i]**2)**2+(array[i] - 1) ** 2
    return ans

def ellipiside(array):
    n = len(array)
    a = np.array([1000.0**((i-1)/(n-1)) for i in range(1, n+1)])
    return np.sum((a * array) ** 2)






def ackley(array):#m=15.5, sigma=14.5
    n = len(array)
    term1 = -20 * np.exp(-0.2 * np.sqrt((1/n) * np.sum(array**2)))
    term2 = -np.exp((1/n) * np.sum(np.cos(2 * np.pi * array)))

    return term1 + term2 + 20 + np.exp(1)

def rastrigin(array):#m=3, sigma=2
    n = len(array)
    return 10 * n + np.sum(array**2 - 10 * np.cos(2 * np.pi * array))

def six_hunp_camel(array):
    if len(array)>2:
        raise ValueError
    x1 = array[0]
    x2 = array[1]
    return (4 - 2.1 * (x1**2) + (x1**4) / 3) * (x1**2)+ x1 * x2 + (-4 + 4 * x2**2) * (x2**2)
