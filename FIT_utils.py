import numpy as np
import os
import torch
import torch.nn as nn

import time

class FIT:
    def __init__(self, model, device, data_loader, layer_filter=None):
        
        names, param_nums, params = self.layer_accumulator(model, layer_filter)
        param_sizes = [p.size() for p in params]
        self.hook_layers(model, layer_filter)
        batch = next(iter(data_loader))
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, start_positions, end_positions = batch
        _ = model(input_ids, segment_ids, input_mask, start_positions, end_positions)
        act_sizes = []
        act_nums = []
        for name, module in model.named_modules():
            if module.act_quant:
                act_sizes.append(module.act_in[0].size())
                act_nums.append(np.prod(np.array(module.act_in[0].size())[1:]))
                
        self.names = names
        self.param_nums = param_nums
        self.params = params
        self.param_sizes = param_sizes
        self.act_sizes = act_sizes
        self.act_nums = act_nums
        
        self.device = device
             
    def layer_accumulator(self, model, layer_filter=None):
        def layer_filt(nm):
            if layer_filter is not None:
                return layer_filter not in name
            else:
                return True
        layers = []
        names = []
        param_nums = []
        params = []
        for name, module in model.named_modules():
            if (isinstance(module, nn.Linear) or (isinstance(module, nn.Conv2d)) or (isinstance(module, nn.ConvTranspose2d))) and (layer_filt(name)):
                for n, p in list(module.named_parameters()):
                    if n.endswith('weight'):
                        names.append(name)
                        p.collect = True
                        layers.append(module)
                        param_nums.append(p.numel())
                        params.append(p)
                    else:
                        p.collect = False
                continue
            for p in list(module.parameters()):
                if p.requires_grad:
                    p.collect = False
        print(len(layers))
        print(np.sum(param_nums))
        for i, (n, p) in enumerate(zip(names, param_nums)):
            print(i, n, p)

        return names, np.array(param_nums), params
    
    def hook_layers(self, model, layer_filter=None):

        def hook_inp(m, inp, out):
            m.act_in = inp

        def layer_filt(nm):
            if layer_filter is not None:
                return layer_filter not in name
            else:
                return True

        for name, module in model.named_modules():
            if (isinstance(module, nn.Linear) or (isinstance(module, nn.Conv2d))) and (layer_filt(name)):

                module.register_forward_hook(hook_inp)
                module.act_quant = True
            else:
                module.act_quant = False
                
    def EF(self, model, 
           data_loader, 
           criterion, 
           tol=1e-3, 
           min_iterations=100, 
           max_iterations=100):
        
        model.eval()
        F_act_acc = []
        F_param_acc = []
        param_estimator_accumulation = []
        act_estimator_accumulation = []

        F_flag = False

        total_batches = 0.

        TFv_act = [torch.zeros(ps).to(self.device) for ps in self.act_sizes[1:]]  # accumulate result
        TFv_param = [torch.zeros(ps).to(self.device) for ps in self.param_sizes]  # accumulate result

        ranges_param_acc = []
        ranges_act_acc = []
        
        while(total_batches < max_iterations and not F_flag):
            
            for i, data in enumerate(data_loader, 1):
                model.zero_grad()
                
                batch = tuple(t.to(self.device) for t in data)
                batch_size= len(batch)
                input_ids, input_mask, segment_ids, start_positions, end_positions = batch
                loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions)
                
                ranges_act = []
                actsH = []
                for name, module in model.named_modules():
                    if module.act_quant:
                        actsH.append(module.act_in[0])
                        ranges_act.append((torch.max(module.act_in[0]) - torch.min(module.act_in[0])).detach().cpu().numpy())

                ranges_param = []
                paramsH = []
                for paramH in model.parameters():
                    if not paramH.collect:
                        continue
                    paramsH.append(paramH)
                    ranges_param.append((torch.max(paramH.data) - torch.min(paramH.data)).detach().cpu().numpy())
                    
                G = torch.autograd.grad(loss, [*paramsH, *actsH[1:]])
                
                G2 = []
                for g in G:
                    G2.append(batch_size*g*g)
                    
                indiv_param = np.array([torch.sum(x).detach().cpu().numpy() for x in G2[:len(TFv_param)]])
                indiv_act = np.array([torch.sum(x).detach().cpu().numpy() for x in G2[len(TFv_param):]])
                param_estimator_accumulation.append(indiv_param)
                act_estimator_accumulation.append(indiv_act)
                    
                TFv_param = [TFv_ + G2_ + 0. for TFv_, G2_ in zip(TFv_param, G2[:len(TFv_param)])]
                ranges_param_acc.append(ranges_param)
                TFv_act = [TFv_ + G2_ + 0. for TFv_, G2_ in zip(TFv_act, G2[len(TFv_param):])]
                ranges_act_acc.append(ranges_act)
                
                total_batches += 1
                
                TFv_act_normed = [TFv_ / float(total_batches) for TFv_ in TFv_act]
                vFv_act = [torch.sum(x) for x in TFv_act_normed]
                vFv_act_c = np.array([i.detach().cpu().numpy() for i in vFv_act])

                TFv_param_normed = [TFv_ / float(total_batches) for TFv_ in TFv_param]
                vFv_param = [torch.sum(x) for x in TFv_param_normed]
                vFv_param_c = np.array([i.detach().cpu().numpy() for i in vFv_param])
                
                F_act_acc.append(vFv_act_c)
                F_param_acc.append(vFv_param_c)
                
                if total_batches >= 2:
 
                    param_var = np.var((param_estimator_accumulation - vFv_param_c)/vFv_param_c)/total_batches
                    act_var= np.var((act_estimator_accumulation - vFv_act_c)/vFv_act_c)/total_batches
                    
                    print(f'Iteration {total_batches}, Estimator variance: W:{param_var} / A:{act_var}')
                
                    if act_var < tol and param_var < tol and total_batches > min_iterations:
                        F_flag = True
                
                if F_flag or total_batches >= max_iterations:
                    break
        
        self.EFw = vFv_param_c
        self.EFa = vFv_act_c
        self.FAw = F_param_acc
        self.FAa = F_act_acc
        self.Rw = ranges_param_acc
        self.Ra = ranges_act_acc
        
        return vFv_param_c, vFv_act_c, F_param_acc, F_act_acc, ranges_param_acc, ranges_act_acc
    
    # compute FIT:
    def noise_model(self, ranges, config):
        return (ranges/(2**config - 1))**2

    def FIT(self, wconfig, aconfig):
        pert_acts = self.noise_model(np.mean(self.Ra, axis=0)[1:], aconfig)
        pert_params = self.noise_model(np.mean(self.Rw, axis=0), wconfig)

        f_acts_T = pert_acts*self.EFa
        f_params_T = pert_params*self.EFw
        pert_T = np.sum(f_acts_T) + np.sum(f_params_T)
        return pert_T