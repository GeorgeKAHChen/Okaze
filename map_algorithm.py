import torch 
import time
#from layer import map_calc
#from layer import rand_map_calc

import setting_parameters
import models
import lya_spec


is_map_algorithm_init = 0
iter_results = []

def map_algorithm_init(dim, tensor_size):
    global iter_results, is_map_algorithm_init
    if is_map_algorithm_init == 1:
        return
    else:
        iter_results = torch.Tensor([[0 for p in range(tensor_size)] for q in range(models.dim)]).double().to(setting_parameters.device)
        is_map_algorithm_init = 1


def map_algorithm(final_group):
    # Parameter for tensor computation 
    tensor_size = len(final_group[0])

    # System parameter
    curr_t      = 0
    curr_x      = torch.Tensor(final_group[0: models.dim]).double().to(setting_parameters.device)
    para        = torch.Tensor(final_group[models.dim:models.dim+models.para_size]).double().to(setting_parameters.device)
    rand_para   = torch.Tensor(final_group[models.dim+models.para_size: len(final_group)]).double().to(setting_parameters.device)
    
    # rand system parameter
    min_rand    =   - rand_para.double()
    max_rand    =     rand_para.double()
    dis_rand    = 2 * rand_para.double()

    # Orbit parameter
    t_ob_mark   = -1

    # Lya spec parameter
    t_le_mark   = -1
    spectrum    = torch.Tensor([[0 for p in range(tensor_size)] for q in range(models.dim)]).double().to(setting_parameters.device)
    eye         = torch.Tensor([[0 for p in range(tensor_size)] for q in range(models.dim * models.dim)]).double().to(setting_parameters.device)
    curr_le_t   = 0

    # PS parameter
    t_ps_mark   = -1
    ps_print    = -1
    ps_return   = torch.Tensor().double().to(setting_parameters.device)
    
    # For output
    runtime = time.time()

    # memory initialization
    map_algorithm_init(models.dim, tensor_size)


    if (setting_parameters.calc_ob == 1):
        t_ob_mark = setting_parameters.step_max * setting_parameters.t_ob;
    if (setting_parameters.calc_le == 1):
        t_le_mark = setting_parameters.step_max * setting_parameters.t_le;
    if (setting_parameters.calc_ps == 1):
        t_ps_mark = setting_parameters.step_max * setting_parameters.t_ps;

    # Computatiom method
    use_maruyama = 0
    print_kase = 0

    while 1:
        if (curr_t > setting_parameters.step_max):
            break

        # system iteration
        if (use_maruyama == 0):
            curr_x              = models.f(iter_results, curr_x, curr_t, para, tensor_size, setting_parameters.device)
        else:
            map_euler_x         = models.f(iter_results, curr_x, curr_t, para, tensor_size, setting_parameters.device)
            map_random_value    = torch.rand(min_rand.size()).double().to(setting_parameters.device)
            map_random_value    = (map_random_value * dis_rand + min_rand).double().to(setting_parameters.device)
            curr_x              = models.rand_f(map_euler_x, curr_x, curr_t, map_random_value, rand_para, 1, tensor_size, setting_parameters.device)

        # orbit output
        if (setting_parameters.calc_ob == 1 and curr_t >= t_ob_mark):
            pass
            if (models.rand_para_size == 0 or models.rand_dim == 0):
                use_maruyama = 0
            else:
                use_maruyama = 1;

        # Lya spec
        if (setting_parameters.calc_le == 1 and curr_t > t_le_mark):
            eye, spectrum = lya_spec.lya_spec(models.dim, curr_x, 1, models.Jf, eye, spectrum, curr_le_t, para, tensor_size)
            if (models.rand_para_size == 0 or models.rand_dim == 0):
                use_maruyama = 0
            else:
                use_maruyama = 1;

        # Poincare section output
        if (setting_parameters.calc_ps == 1 and curr_t > t_ps_mark):
            pass    # Ps
            if (models.rand_para_size == 0 or models.rand_dim == 0):
                use_maruyama = 0
            else:
                use_maruyama = 1;
        
        # time iteration
        if setting_parameters.calc_le == 1 and curr_t > t_le_mark:
            curr_le_t += 1
        curr_t += 1
        print_kase += 1

        if (print_kase % 10000 == 0):
            print(str(curr_t) + " " + str(setting_parameters.step_max) + ": " + str(time.time() - runtime), end = "\n")
            runtime = time.time()

    if (setting_parameters.calc_le == 1):
        pass    #le output
    # Saved end Value if system don't save orbit
