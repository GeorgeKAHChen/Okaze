import torch
import time
import setting_parameters

mat_result = []
new_spec = []
results = []
is_lya_spec_init = 0

def lya_spec_init(dim, tensor_size):
    global mat_result, new_spec, results, is_lya_spec_init
    if is_lya_spec_init == 1:
        return 
    else:
        mat_result = torch.Tensor([[0 for n in range(tensor_size)]for n in range(dim * dim)]).double().to(setting_parameters.device)
        new_spec = torch.Tensor([[0 for n in range(tensor_size)]for n in range(dim)]).double().to(setting_parameters.device)
        results = torch.Tensor([[0 for n in range(tensor_size)]for n in range(dim * dim)]).double().to(setting_parameters.device)
        is_lya_spec_init = 1
        return

def mat_multi(dim, mat_x, mat_y, tensor_size):
    global mat_result
    for i in range(0, dim):
        for j in range(0, dim):
            mat_result[i*dim + j] = torch.Tensor([0 for n in range(tensor_size)]).double().to(setting_parameters.device)
            for k in range(0, dim):
                mat_result[i*dim + j] += mat_x[i * dim + k] * mat_y[j + k*dim]
    return mat_result


def gram_schmidt(dim, mat_result, eye, tensor_size):
    global new_spec

    eye = torch.clone(mat_result)
    
    for kase in range(0, dim):
        for i in range(0, kase):
            
            inner_beta = torch.Tensor([0 for n in range(tensor_size)]).double().to(setting_parameters.device)
            inner_ab = torch.Tensor([0 for n in range(tensor_size)]).double().to(setting_parameters.device)

            for j in range(0, dim):
                inner_beta += eye[i+j*dim] * eye[i+j*dim]
                inner_ab += eye[i+j*dim] * mat_result[kase+j*dim]
            
            for j in range(0, dim):
                eye[kase + j*dim] -= (inner_ab/inner_beta) * eye[i+j*dim]

    for i in range(0, dim):
        new_spec[i] = torch.Tensor([0 for n in range(tensor_size)]).double().to(setting_parameters.device)
        for j in range(0, dim):
            new_spec[i] += eye[i + dim*j] * eye[i + dim*j]
        new_spec[i] = torch.sqrt(new_spec[i]);
        for j in range(0, dim):
            eye[i + dim*j] /= new_spec[i];
    
    return eye, new_spec



def lya_spec(dim, curr_x, delta_t, Jf, eye, spectrum, t_after, para, tensor_size):
    # Memory initialization
    lya_spec_init(dim, tensor_size)

    # Calculate the Jacobian Matrix
    mat_Jaco = Jf(results, curr_x, delta_t, para, tensor_size, setting_parameters.device);
    
    # Matrix multiply
    mat_result = mat_multi(dim, mat_Jaco, eye, tensor_size);
    
    # Gram_Schmidt and normalization
    eye, new_spec = gram_schmidt(dim, mat_result, eye, tensor_size);

    # Change the Spectrum
    for i in range(0, dim):
        spectrum[i] = (spectrum[i] * t_after + torch.log(new_spec[i])) / (t_after + delta_t);

    return eye, spectrum

