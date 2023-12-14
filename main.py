import numpy as np
import torch
from copy import deepcopy

import setting_parameters
import models
#import main_algorithm
import map_algorithm



def main():
    group_dim = models.dim + models.para_size + models.rand_para_size
    
    total_group = []
    tmp_stack = []
    point = 0

    tmp_arr = []
    for i in range(0, models.group_size[0]):
        tmp_arr.append(models.gen_init_val())
    tmp_stack.append(tmp_arr)

    for i in range(models.dim, group_dim):
        tmp_arr = []
        if models.group_size[i] == 1:
            tmp_arr.append(models.min_para[i])
        else:
            delta_val = (models.max_para[i] - models.min_para[i]) / (models.group_size[i]-1)
            for j in range(0, models.group_size[i] - 1):
                tmp_arr.append(models.min_para[i] + j * delta_val)
            tmp_arr.append(models.max_para[i])

        tmp_stack.append(tmp_arr)

    total_group = deepcopy(tmp_stack[0])
    point = 1
    while 1:
        if len(total_group[0]) >= models.dim + point:
            point += 1
        if point >= len(tmp_stack):
            break
        tmp_vector = total_group.pop(0)
        for i in range(0, len(tmp_stack[point])):
            total_group.append(tmp_vector + [tmp_stack[point][i]])

    final_group = [[] for n in range(group_dim)]
    for i in range(0, len(total_group)):
        for j in range(0, group_dim):
            final_group[j].append(total_group[i][j])
    #print(final_group)

    if (models.this_is_map == 0):
        print("continuous system computation not finish!!")
        pass
    else:
        map_algorithm.map_algorithm(final_group);
    return 0;


if __name__ == '__main__':
    main()