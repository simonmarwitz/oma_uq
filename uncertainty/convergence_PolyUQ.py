import numpy as np
import scipy.stats
from uncertainty.polymorphic_uncertainty import RandomVariable, MassFunction, PolyUQ
from helpers import get_pcd
import sys
import psutil
import time
import os

import glob
import matplotlib
import matplotlib.pyplot as plt

def example_a():
    example_num = 0

    q1 = RandomVariable(name='q1', dist='norm', params=[15, 4], primary=True)
    q2 = RandomVariable(name='q2', dist='norm', params=[8, 2], primary=True)
    
    dim_ex='hadamard'
    
    vars_ale = [q1,q2]
    vars_epi = []
    return example_num, dim_ex, vars_ale, vars_epi,
    
def example_b():
    example_num = 1
    
    inc_q1a1 = (RandomVariable(name='q1', dist='norm', params=[15,4], primary=False), # set same name to share input samples
               )
    inc_q1a0 = (inc_q1a1[0],
                RandomVariable(name='q1a0l', dist='norm', params=[-0.4,0.1], primary=False), 
                RandomVariable(name='dq1a0r', dist='norm', params=[0.5,0.06], primary=False))
    q1 = MassFunction(name='q1', focals=[inc_q1a1, inc_q1a0], masses=[0.5,0.5], primary=True, incremental=True)
    
    inc_q2a1 = (RandomVariable(name='q2', dist='norm', params=[8,2], primary=False), # set same name to share input samples
               )
    inc_q2a0 = (inc_q2a1[0],
                RandomVariable(name='dq2a0l', dist='norm', params=[-0.6,0.08], primary=False), 
                RandomVariable(name='dq2a0r', dist='norm', params=[0.4,0.11], primary=False))
    q2 = MassFunction(name='q2', focals=[inc_q2a1, inc_q2a0], masses=[0.5,0.5], primary=True, incremental=True)
    
    dim_ex='hadamard'
    
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
    
    dim_ex='hadamard'
    
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
    
    inc_q2a1 = (RandomVariable(name='q2', dist='norm', params=[8,2], primary=False),) # set same name to share input samples
    inc_q2a0 = (inc_q2a1[0],
                RandomVariable(name='dq2a0l', dist='norm', params=[-0.6,0.08], primary=False), 
                RandomVariable(name='dq2a0r', dist='norm', params=[0.4,0.11], primary=False))
    q2 = MassFunction(name='q2', focals=[inc_q2a1, inc_q2a0], masses=[0.5,0.5], primary=True, incremental=True)
    
    dim_ex='cartesian'
    vars_epi = [mu1,sig1, q2]
    vars_ale = [q1, *inc_q2a0]
    
    return example_num, dim_ex, vars_ale, vars_epi,

def main():
    
    global fcount
    fcount = 0
    
    def deterministic_mapping2(q1,q2):
        global fcount
        fcount += 1
        return (189/500*q1+3*q2)*16/3
    arg_vars = {'q1':'q1', 'q2':'q2'}
    
    N_mcs = int(10**(float(sys.argv[2])/8))
    # 2 - 7 in steps of 0.125 -> 100...1333521 -> 16...50
    example_num = int(sys.argv[1])
    print('N_mcs', N_mcs)
    # if example_num==4:
        # N_mcs = np.floor(np.sqrt(N_mcs)).astype(int) # that makes N_mcs == fcount

    now = time.time()
    _, dim_ex, vars_ale, vars_epi, = [example_a, example_b, example_c, example_d, example_e][example_num]()
    # 1 3 4
    poly_uq = PolyUQ(vars_ale, vars_epi, dim_ex=dim_ex)
    if True:
        poly_uq.sample_qmc(N_mcs, N_mcs, check_sample_sizes=False)
        
        poly_uq.propagate(deterministic_mapping2, arg_vars)
        fcount = poly_uq.fcount
        poly_uq.estimate_imp(True, opt_meth = 'Nelder-Mead')
        
        def stat_fun(a, weight, i_stat):
            exceed = a>=260
            return np.sum(weight[exceed])
    
        focals_Pf, hyc_mass = poly_uq.optimize_inc(stat_fun, 1)
    else:
        fcount = 0
        focals_Pf, hyc_mass = poly_uq.naive_uq(N_mcs, deterministic_mapping2, arg_vars)
        example_num += 5
    
    mem = psutil.Process().memory_info().rss / (1024 * 1024)
    
    runtime = time.time() - now
    
    print({'focals_Pf':focals_Pf*1e4, 'fcount':fcount, 'mem':mem, 'runtime':runtime})
    
    np.savez(f'/vegas/scratch/womo1998/modal_uq/convergence_PolyUQ/{example_num}_{N_mcs}_{os.environ["SLURM_ARRAY_JOB_ID"]}', **{'focals_Pf':focals_Pf, 'fcount':fcount, 'mem':mem, 'runtime':runtime})
    
    return example_num, N_mcs, focals_Pf, fcount, mem, runtime

def analyze(example_num, hyc_num=None):
    files = glob.glob(f'/vegas/scratch/womo1998/modal_uq/convergence_PolyUQ/{example_num}_*')
    if not files: return
    
    N_mcss = []
    focals = []
    fcounts = []
    mems = []
    runtimes = []
    for file in files:
        N_mcss.append(int(file.split('_')[-2].strip('.npz')))
        arr = np.load(file, allow_pickle=True)
        focals.append(arr['focals_Pf'])
        fcounts.append(arr['fcount'].item())
        mems.append(arr['mem'].item())
        runtimes.append(arr['runtime'].item())
        
    # plt.plot(N_mcss, runtimes, ls='none', marker='.')
    # plt.show()
    fcounts=np.array(fcounts)
    N_mcss=np.array(N_mcss)
    fcounts=N_mcss
    
    
    
    focals = np.stack(focals)
    n_hyc = focals.shape[2]
    if example_num==0:
        it_hyc = [0]
        ref = [[1.003e-3],]
        color='#377eb8'
    elif example_num==1:
        it_hyc = [0,1]
        ref = [[10.27e-4],
               [3.21e-4,21.94e-4]]
        color='#4daf4a'
    elif example_num==2:
        it_hyc = [0,1]
        ref = [[20.10e-4],
               [2.33e-4,120.16e-4]]
        color='#ff7f00'
    elif example_num==3:
        it_hyc = [0,1,2,3]
        
        ref = [[19.12e-4],
               [1.66e-4,111.55e-4],
               [7.6e-4,37.07e-4],
               [0.54e-4,183.12e-4],
               ]
        color='#e41a1c'
    elif example_num==4:
        it_hyc = [0,1,2,3,4,5,6,7]
        
        ref = [[None], # complete, precise
               [None,None], # partly incomplete, precise
               [None,None], # partly incomplete, precise
               [None,None], # incomplete, precise
               [None,None], # complete, imprecise
               [None,None], # partly incomplete, imprecise
               [None,None], # partly incomplete, imprecise
               [None,None], #incomplete, imprecise
               ]
        color='#984ea3'
    # print(focals.shape)
    
    '''

    
    compute relative error for each sample
    compute confidence interval for each fcount
    plot fill_between
    
    '''
    
    for i,i_hyc in enumerate(it_hyc):
        if hyc_num is not None:
            if hyc_num!=i_hyc: continue
        for high_low in range(len(ref[i])):
            unique = np.unique(fcounts)
            unique = unique[unique>=1e3]
            conf= np.zeros((len(unique),2))
            if ref[i][high_low] is None:
            # if True:
                ref[i][high_low] = np.mean(focals[fcounts==np.max(unique),0,i_hyc,high_low])
            print(f'Ex {example_num} Hyc {i_hyc} u/l {high_low}')
            print(np.mean(focals[fcounts==np.max(unique),0,i_hyc,high_low]), ref[i][high_low])
            for j in range(len(unique)):
                
                inds = fcounts==unique[j]
                # print(f'Ex {example_num} Hyc {i_hyc} u/l {high_low} Nmcs {unique[j]}; samples: {np.sum(inds)}')
                data = focals[inds,0,i_hyc,high_low]
                data -= ref[i][high_low]
                data /= ref[i][high_low]
                data *= 100
                data = np.abs(data)
                
                conf[j,:] = scipy.stats.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=scipy.stats.sem(data)) 
            
            plt.fill_between(unique, conf[:,0], conf[:,1], alpha=0.5, color=color, ls=['solid','dashed'][high_low])
            
        
        plt.xscale('log')
    # plt.axhline(1, c='grey', lw=.5)
    # plt.show()

def analyze2():
    '''
    group into (example, hyc) 
    '''
    with matplotlib.rc_context(rc=get_pcd()):
        fig, axes = plt.subplots(2,2,sharex=True, sharey='col')
        axes = axes.ravel()
        
        for n in [0,1,2,3,]:
        # for n in [3,]:
            plt.sca(axes[n])
            if n == 0:
                datasets = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]]  # pure variability
                title='variability (var)'
            elif n == 1:
                datasets = [ [3, 2],[1, 1], [4, 4]]  # variability+imprecision
                title='var. + imprecision (imp.)'
            elif n == 2:
                datasets = [[2, 1], [3, 1], [4, 3]]  # incompleteness+variability
                title='incompleteness (inc.) + var.'
            elif n == 3:
                datasets = [[3, 3], [4, 7]]  # inc+var+imp
                title='inc. + var. + imp.'
                
            for example_num,hyc_num in datasets:
                analyze(example_num, hyc_num)
                pass
            # axes[n].set_ylim((-99,99))
            plt.title(title, y=1.0, pad=-10, size=10)
            
            if axes[n].is_last_row():
                # plt.xlabel("No. of Monte Carlo Samples [-]")
                pass
            if axes[n].is_first_col():
                # plt.ylabel("Avg. rel. error [\%]")
                plt.ylim((0,15))
                pass
            if axes[n].is_last_col():
                axes[n].yaxis.tick_right()
                plt.ylim((0,400))
        fig.text(0.5, 0.02, "No. of Monte Carlo Samples [-]", va='center', ha='center')
        fig.text(0.02, 0.5, "Avg. rel. error [\%]", va='center', ha='center', rotation='vertical', )
        fig.text(0.98, 0.5, "Avg. rel. error [\%]", va='center', ha='center', rotation='vertical', )
            
            
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#377eb8', label='pure aleatory'),
                           Patch(facecolor='#4daf4a', label='primary epistemic'),
                           Patch(facecolor='#ff7f00', label='primary aleatory'),
                           Patch(facecolor='#e41a1c', label='primary epistemic'),
                           Patch(facecolor='#984ea3', label='mixed ale.-epi.')]
        fig.legend(handles=legend_elements, loc=(0.625,0.24)).set_draggable(True)
        fig.subplots_adjust(top=0.97,bottom=0.110, left=0.075, right=0.9, hspace=0.07, wspace=0.035)
        plt.xlim((1e3,1333521))
        # fig.savefig(f'/vegas/users/staff/womo1998/Projects/2019_OMA_UQ/tex/figures/math_basics/convergence_poly_uq.pdf')
        # fig.savefig(f'/vegas/users/staff/womo1998/Projects/2019_OMA_UQ/tex/figures/math_basics/convergence_poly_uq.png', dpi=300)

        plt.show()

if __name__ == '__main__':
    sys.argv.append(3)
    sys.argv.append(30)
    main()
    # for i in range(8):
        # plt.figure()
        # analyze(4,i)
        #
    # plt.show()
    # analyze2()