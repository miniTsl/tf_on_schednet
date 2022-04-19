# import argparse
# parser = argparse.ArgumentParser(description='PyTorch RL trainer')
# parser.add_argument('--abc', default=100, type=int,help='1st paramter abc')
# args = parser.parse_args()
# if args.abc == 100:
#     args.ABC = 99   # 可以直接自动添加参数
# print(args)

from cmath import inf
import numpy as np
# np.concatenate()

# a = ((np.array([1]), np.array([1]), np.array(range(10))),(np.array([0]), np.array([0]), np.array(range(-10))))
# print(a)

a = np.array([[1,2],[3,4]])
print(a.reshape(1,-1))
print(a, '\n', a.flatten())

a = {'is_completed': np.array([11,2])}
a['apple'] = 123
print(a)

def print_tuple(data):
    data_ = []
    for x in data:
        temp = [x[0]]
        temp+= x[1].flatten().tolist()  # 用➕合并两个list
        data_.append(temp)
    return data_
a = ((1.2, np.zeros((1,1,59))),(3.4, np.zeros((1,1,59))))
print(print_tuple(a))

a = np.array([1,1,1,1])
print(np.sum(a))

a = [np.array([1,2,3]), np.array([2,3,3])]
print(np.concatenate(a), type(np.concatenate(a)))
print(inf>1e100)