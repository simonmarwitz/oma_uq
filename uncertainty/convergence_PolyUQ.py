import numpy as np
from uncertainty.polymorphic_uncertainty import RandomVariable,MassFunction,PolyUQ
import sys
import psutil
import time

def example_a():
    example_num = 0

    q1 = RandomVariable(name='q1', dist='norm', params=[15, 4], primary=True)
    q2 = RandomVariable(name='q2', dist='norm', params=[8, 2], primary=True)
    
    dim_ex='cartesian'
    
    vars_ale = [q1,q2]
    vars_epi = []
    return example_num, dim_ex, vars_ale, vars_epi,
    
def example_b():
    example_num = 1
    
    inc_q1a1 = (RandomVariable(name='q1', dist='norm', params=[15,4], primary=False), # set same name to share input samples
               -0.05,
               0.05)
    inc_q1a0 = (inc_q1a1[0],
                RandomVariable(name='q1a0l', dist='norm', params=[-0.4,0.1], primary=False), 
                RandomVariable(name='dq1a0r', dist='norm', params=[0.5,0.06], primary=False))
    q1 = MassFunction(name='q1', focals=[inc_q1a1, inc_q1a0], masses=[0.5,0.5], primary=True, incremental=True)
    
    inc_q2a1 = (RandomVariable(name='q2', dist='norm', params=[8,2], primary=False), # set same name to share input samples
               -0.05,
               0.05)
    inc_q2a0 = (inc_q2a1[0],
                RandomVariable(name='dq2a0l', dist='norm', params=[-0.6,0.08], primary=False), 
                RandomVariable(name='dq2a0r', dist='norm', params=[0.4,0.11], primary=False))
    q2 = MassFunction(name='q2', focals=[inc_q2a1, inc_q2a0], masses=[0.5,0.5], primary=True, incremental=True)
    
    dim_ex='cartesian'
    
    vars_ale = [*inc_q1a0,*inc_q2a0]
    vars_epi = [q1, q2]
    return example_num, dim_ex, vars_ale, vars_epi,
    

def example_c():
    example_num = 2

    mu1 = MassFunction(name='mu1', focals=[(14.79,14.79),(13.96,15.61)], masses=[0.5,0.5], primary=False)
    sig1 = MassFunction(name='sig1', focals=[(4.17,4.17),(3.66,4.85)], masses = [0.5, 0.5], primary=False)
    q1 = RandomVariable(name='q1', dist='norm', params=[mu1, sig1])
    
    mu2 = MassFunction(name='mu2', focals=[(8.1,8.1),(7.67,8.52)], masses=[0.5,0.5], primary=False)
    sig2 = MassFunction(name='sig2', focals=[(2.14,2.14),(1.88,2.48)], masses = [0.5, 0.5], primary=False)
    q2 = RandomVariable(name='q2', dist='norm', params=[mu2, sig2])
    
    dim_ex='cartesian'
    
    vars_epi = [mu1,sig1, mu2, sig2]
    vars_ale = [q1, q2]
    return example_num, dim_ex, vars_ale, vars_epi,
    
def example_d():
    example_num = 3

    q1a1mu    = MassFunction(name='q1a1mu',    focals=[(14.62,      ), (13.8, 15.45)], masses=[0.5,0.5],  primary=False)
    q1a1sig   = MassFunction(name='q1a1sig',   focals=[(4.16 ,      ), (3.65, 4.83)], masses=[0.5,0.5],   primary=False)
    q1a1      = RandomVariable(name='q1',    dist='norm', params=[q1a1mu, q1a1sig], primary=False)
    
    dq1a0lmu  = MassFunction(name='dq1a0lmu',  focals=[(-0.41,      ), (-0.46, -0.38)], masses=[0.5,0.5],  primary=False)
    dq1a0lsig = MassFunction(name='dq1a0lsig', focals=[( 0.10,      ), ( 0.09,  0.12)], masses=[0.5,0.5],  primary=False)
    dq1a0l    = RandomVariable(name='dq1a0l',  dist='norm', params=[dq1a0lmu, dq1a0lsig], primary=False)
    
    dq1a0rmu  = MassFunction(name='dq1a0rmu',  focals=[( 0.50,      ), ( 0.49,  0.52)], masses=[0.5,0.5],  primary=False)
    dq1a0rsig = MassFunction(name='dq1a0sig',  focals=[( 0.06,      ), ( 0.05,  0.08)], masses=[0.5,0.5],  primary=False)
    dq1a0r    = RandomVariable(name='dq1a0r',  dist='norm', params=[dq1a0rmu, dq1a0rsig], primary=False)
    
    q1        = MassFunction(name='q1', focals=[(q1a1, 0, 0), (q1a1, dq1a0l, dq1a0r)], masses=[0.5,0.5], primary=True, incremental=True)
    
    q2a1mu    = MassFunction(name='q2a1mu',   focals=[(8.10,     ), (7.67, 8.52)], masses=[0.5,0.5],  primary=False)
    q2a1sig   = MassFunction(name='q2a1sig',  focals=[(2.12,     ), (1.86, 2.46)], masses=[0.5,0.5],   primary=False)
    q2a1      = RandomVariable(name='q2',   dist='norm', params=[q2a1mu, q2a1sig], primary=False)
    
    dq2a0lmu  = MassFunction(name='dq2a0lmu',  focals=[(-0.59,      ), (-0.61, -0.57)], masses=[0.5,0.5],  primary=False)
    dq2a0lsig = MassFunction(name='dq2a0lsig', focals=[( 0.08,      ), ( 0.07,  0.09)], masses=[0.5,0.5],  primary=False)
    dq2a0l    = RandomVariable(name='dq2a0l',  dist='norm', params=[dq2a0lmu, dq2a0lsig], primary=False)
    
    dq2a0rmu  = MassFunction(name='dq2a0rmu',  focals=[(0.40,     ), (0.38, 0.42)], masses=[0.5,0.5],  primary=False)
    dq2a0rsig = MassFunction(name='dq2a0sig',  focals=[(0.10,     ), (0.09, 0.12)], masses=[0.5,0.5],  primary=False)
    dq2a0r    = RandomVariable(name='dq2a0r',  dist='norm', params=[dq2a0rmu, dq2a0rsig], primary=False)
    
    q2        = MassFunction(name='q2', focals=[(q2a1,     ), (q2a1, dq2a0l, dq2a0r)], masses=[0.5,0.5], primary=True, incremental=True)
    
    dim_ex='hadamard'
    
    vars_epi = [q1a1mu, q1a1sig, dq1a0lmu, dq1a0lsig, dq1a0rmu, dq1a0rsig, q1,
                q2a1mu, q2a1sig, dq2a0lmu, dq2a0lsig, dq2a0rmu, dq2a0rsig, q2]
    vars_ale = [q1a1, dq1a0l, dq1a0r,
                q2a1, dq2a0l, dq2a0r,]
    return example_num, dim_ex, vars_ale, vars_epi,

def example_e():
    example_num = 4

    mu1 = MassFunction(name='mu1', focals=[(14.79,14.79),(13.96,15.61)], masses=[0.5,0.5], primary=False)
    sig1 = MassFunction(name='sig1', focals=[(4.17,4.17),(3.66,4.85)], masses = [0.5, 0.5], primary=False)
    q1 = RandomVariable(name='q1', dist='norm', params=[mu1, sig1])
    
    inc_q2a1 = (RandomVariable(name='q2', dist='norm', params=[8,2], primary=False), # set same name to share input samples
               -0.05,
               0.05)
    inc_q2a0 = (inc_q2a1[0],
                RandomVariable(name='dq2a0l', dist='norm', params=[-0.6,0.08], primary=False), 
                RandomVariable(name='dq2a0r', dist='norm', params=[0.4,0.11], primary=False))
    q2 = MassFunction(name='q2', focals=[inc_q2a1, inc_q2a0], masses=[0.5,0.5], primary=True, incremental=True)
    
    dim_ex='cartesian'
    vars_epi = [mu1,sig1, q2]
    vars_ale = [q1, *inc_q2a0]
    
    return example_num, dim_ex, vars_ale, vars_epi,

def main():
    
    
    def deterministic_mapping2(q1,q2):
        return (189/500*q1+3*q2)*16/3
    arg_vars = {'q1':'q1', 'q2':'q2'}
    
    N_mcs = int(sys.argv[2])
    example_num = int(sys.argv[1])
    
    now = time.time()
    _, dim_ex, vars_ale, vars_epi, = [example_a, example_b, example_c, example_d, example_e][example_num]()    
    
    poly_uq = PolyUQ(vars_ale, vars_epi, dim_ex=dim_ex)
    poly_uq.sample_qmc(N_mcs, N_mcs, check_sample_sizes=False)
    
    poly_uq.propagate(deterministic_mapping2, arg_vars)
    fcount = poly_uq.fcount
    poly_uq.estimate_imp(False)
    
    def stat_fun(a, weight, i_stat):
        exceed = a>=260
        return np.sum(weight[exceed])

    focals_Pf, hyc_mass = poly_uq.optimize_inc(stat_fun, 1)
    
    mem=psutil.Process().memory_info().rss / (1024 * 1024)
    
    runtime = now - time.time()
    
    
    np.savez(f'{N_mcs}_{example_num}', **{'focals_Pf':focals_Pf, 'fcount':fcount, 'mem':mem, 'runtime':runtime})
    
    return example_num, N_mcs, focals_Pf, fcount, mem, runtime
    
if __name__ == '__main__':
    main()
    
    
    
    
    