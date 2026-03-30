import numpy as np # pyright: ignore[reportMissingImports]
import random as rand

def sum_dist(lst: list):
    ret_list = []
    curr_sum = 0
    for x in lst:
        curr_sum += x
        ret_list.append(curr_sum)
    return ret_list


def rand_from_dist(f, x_min = 0, x_max = 1, res: int = 100):
    r = rand.random()
    width = (x_max-x_min)
    n = int(width * res)
    dx = 1/res
    x_list = list(np.linspace(x_min, x_max, n))
    sums = sum_dist([f(x_list[i])*dx for i in range(len(x_list))]) # numerical integration basically
    for i in range(len(sums)):
        if r <= sums[i] / sums[-1]:
            return x_list[i]
    print("OUT OF RANGE: ", r, sums[i]/width)
    raise Exception("Base random doesn't fall on normalized distribution map. Is the input function defined over your whole domain?")

def rfd_array(f, n: int = 1000, x_range: list = [-1, 1]):
    sample_pts = []
    for i in range(n):
        point = rand_from_dist(f, x_range[0], x_range[1])
        sample_pts.append(point)
    return sample_pts
