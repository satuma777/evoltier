# -*- coding: utf-8 -*-
from six.moves import range
import csv
import os


class Updater(object):
    '''
    Abstraction of main loop.
    '''
    
    def __init__(self, optimizer, obj_func, pop_size=1, threshold=None, max_iter=10000, out='result', logging=False):
        self.opt = optimizer
        self.obj_func = obj_func
        self.pop_size = pop_size
        self.threshold = threshold
        self.max_iter = max_iter
        self.min = optimizer.w_func.min
        self.out = out
        self.logging = logging

        if self.logging:
            if not os.path.isdir(out):
                os.makedirs(out)
            
            with open(out+'/log.csv', 'w') as log_file:
                self.header = ['Generation', 'BestEval'] + self.opt.generate_header() + self.opt.target.generate_header()
                csv_writer = csv.DictWriter(log_file, fieldnames=self.header)
                csv_writer.writeheader()
        
        if self.threshold is None and self.min:
            self.threshold = 1e-6
        elif self.threshold is None:
            self.threshold = 1e+6
    
    def run(self):
        best_solution = None
        success = False
        if self.min:
            best_eval = float('inf')
            for i in range(1, self.max_iter + 1):
                sample = self.opt.target.sampling(pop_size=self.pop_size)
                evals = self.obj_func(sample)
                self.opt.update(evals=evals, sample=sample)
                
                if evals.min() < best_eval:
                    best_eval = evals.min()
                    best_solution = sample[evals.argmin()]
                
                self.print_log(i, best_eval)
                
                if self.logging:
                    self.write_csv_log(i, best_eval)
                
                if best_eval < self.threshold:
                    success = True
                    break
        else:
            best_eval = float('-inf')
            for i in range(1, self.max_iter + 1):
                sample = self.opt.target.sampling(pop_size=self.pop_size)
                evals = self.obj_func(sample)
                self.opt.update(evals=evals, sample=sample)
        
                if evals.max() > best_eval:
                    best_eval = evals.max()
                    best_solution = sample[evals.argmax()]

                self.print_log(i, best_eval)
                
                if self.logging:
                    self.write_csv_log(i, best_eval)
                
                if best_eval > self.threshold:
                    success = True
                    break
        
        return best_solution, best_eval, success

    def print_log(self, i, best_eval):
        updater_info = 'Generation: {}\t BestEval: {}\t '.format(i, best_eval)
        distribution_info = self.opt.target.get_info()
        
        print(updater_info + distribution_info)
    
    def write_csv_log(self, i, best_eval):
        info = {'Generation': i, 'BestEval': best_eval}
        opt_info = self.opt.get_info_dict()
        info.update(opt_info)
        
        distribution_info = self.opt.target.get_info_dict()
        info.update(distribution_info)
        
        with open(self.out+'/log.csv', 'a') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=self.header)
            csv_writer.writerows([info])
