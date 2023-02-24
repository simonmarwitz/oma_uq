
import numpy as np
import pandas as pd
import scipy.stats
import scipy.stats.qmc
import scipy.optimize
import scipy.interpolate
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product, chain
from uncertainty.data_manager import HiddenPrints, simplePbar, DataManager
import uuid
import os
import copy
import time
import sys
import warnings
from uncertainty import data_manager

warnings.filterwarnings("ignore", message="Initial guess is not within the specified bounds")
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

'''
TODO:
- Implement Importance Sampling rather than just uniform
- Implement sensitivities in estimate_imp to allow for vacuous dimension extension on non-sensitive variables

'''

class UncertainVariable(object):
    '''
    Implementation of various types of uncertain variables
    to be used with uniform quasi-random sampling over the support
    that is why they can not be directly sampled
    '''
    def __init__(self, name, children, primary=True):
        self.value = None
        self.name = name
        self.primary = primary
        
        for child in children:
            if isinstance(child, UncertainVariable):
                is_poly = True
                break
        else:
            is_poly = False
        
        self.is_poly = is_poly
        
    def support(self, ):
        '''
        getting the support is usually needed in pre-processing, before propagation
        no frezzing is necessary, simply evaluating the minimal and maximum values possible
        '''
        raise NotImplementedError("When subclassing UncertainVariable you should implement a support() method.")
            
    def freeze(self, value):
        # fix variable, make it a "CertainVariable"
        assert isinstance(value, (int,float))
        assert not np.isnan(value)
        self.value = value
    
    def unfreeze(self, ):
        self.freeze(None)
    
    @property
    def _children(self):
        raise NotImplementedError("When subclassing UncertainVariable you should implement a _children property.")
    
    @property
    def frozen(self, ):
        return self.value is not None
    
    def __repr__(self,):
        return self.name
    
class RandomVariable(UncertainVariable):
    
    def __init__(self, dist, name, params, primary=True):
        super().__init__(name, params, primary)
        
        self.dist = dist
        self.dist_fun = getattr(scipy.stats, self.dist)
        self.params = params
    
    @property
    def _children(self):
        if self.is_poly:
            return self.params
        else:
            return ()
    
    def support(self, percentiles=(0.001,0.999)):
        eval_params = []
        for param in self.params:
            if isinstance(param, UncertainVariable):
                param = param.support(percentiles=percentiles)
            if isinstance(param, (float, int)):
                param = [param]
            eval_params.append(param)
        supp = [np.infty, -np.infty]
        for params in product(*eval_params): # nested for loop over all eval_params -> cartesian product = full factorial
            rv = self.dist_fun(*params)
            if isinstance(self.dist_fun, scipy.stats.rv_continuous):
                this_supp = rv.ppf(percentiles)
            elif isinstance(self.dist_fun, scipy.stats.rv_discrete):
                this_supp = rv.support()
            else:
                raise TypeError(f"Random variable {self} is neither discrete nor continuous but {type(rv)}.")
            
            supp[0] = min(supp[0], this_supp[0])
            supp[1] = max(supp[1], this_supp[1])
        if np.isnan(supp).any():
            raise RuntimeError(f"Variable {self} has invalid support. Check your definitions")
        return supp
    
    def rvs(self, size=1, **kwargs):
        # we need to have all uncertain parameters fixed
        eval_params = []
        for param in self.params:
            if isinstance(param, UncertainVariable):
                if not param.frozen:
                    raise RuntimeError(f"Variable {param.name} is not frozen.")
                eval_params.append(param.value)
            else:
                eval_params.append(param)
        # rv = self.dist_fun(*eval_params)
        
        return self.dist_fun.rvs(*eval_params, size=size, **kwargs)
    
    def prob_dens(self, values=None):
        # we need to have all uncertain parameters fixed
        eval_params = []
        for param in self.params:
            if isinstance(param, UncertainVariable):
                if not param.frozen:
                    raise RuntimeError(f"Variable {param.name} is not frozen.")
                eval_params.append(param.value)
            else:
                eval_params.append(param)

        # rv = self.dist_fun(*eval_params)
        # pdf = rv.pdf(*eval_params,values)
        dist_fun = self.dist_fun
        if isinstance(dist_fun, scipy.stats.rv_continuous):
            pdf = self.dist_fun.pdf(values, *eval_params)
            
        elif isinstance(dist_fun, scipy.stats.rv_discrete):
            pdf = self.dist_fun.pmf(values, *eval_params)
        else:
            raise RuntimeError('Distribution function is neither continuous nor discrete. Check your definitions.')
        isnan = np.isnan(pdf)
        if np.any(isnan):
            logger.warning('Probability density contains NaN. Check your (hyper-)parameters.')
            pdf[isnan] = 0
        return pdf
#         return pdf / pdf.sum()
    
    def __repr__(self,):
        return f'<RV: {self.name}>'
    
class MassFunction(UncertainVariable):
    # currently only continuous-valued belief functions are supported
    # may be extended to set-valued belief functions 
    # Another Class fur fuzzy sets may be created, exposing the same set of methods
    '''
       - properties such as frame, core, bel, pl, q, focal, singletons, 
   - transformations: pignistic, normalize, inverse pignistic (defining a mass function from samples)
   
   incremental focal sets should be implemented, this is important for polymorphy only
       implement internal storage of incremental focals sets
           return non-incremental focal sets
           
    '''
    
    def __init__(self, name, focals, masses, frame=None, primary=True, incremental=False):
        super().__init__(name, chain.from_iterable(focals), primary)
        
        # focals may be a 3-tuple, where the first is the incremental reference and will be dropped in numeric_focal,support etc
        # focals may be a 1-tuple to define a crisp focal set, containing only singletons
        # otherwise it should be a 2-tuple
        
        if not isinstance(focals, np.ndarray): # note, it may contain objects
            focals_l = focals
            focals = np.empty((len(focals),3), dtype=object)
            for focal_arr,focal in zip(focals,focals_l):
                if len(focal) == 1:
                    focal_arr[:] = np.nan
                    focal_arr[1] = focal[0]
                elif len(focal) == 2:
                    focal_arr[1:] = focal
                    focal_arr[0] = np.nan
                    if not (isinstance(focal[0], UncertainVariable) or isinstance(focal[1], UncertainVariable)):
                        assert focal[1] >= focal[0]
                else:
                    focal_arr[:] = focal
                    incremental = True
            
        if not isinstance(masses,  np.ndarray):
            masses = np.array(masses)

        assert focals.shape[0] == masses.shape[0]
        
        total_mass = np.sum(masses)
        if total_mass > 1:
            raise ValueError(f'The sum of all mass values ({total_mass}) must not be greater than one.')
        if  total_mass < 1:
            if frame is None:
                raise ValueError("If unassigned mass remains, the frame must be specified")
            focals = np.vstack([focals, frame])
            masses = np.vstack([masses, [1 - total_mass]])
        
        self.incremental = incremental
        self._focals = focals
        self.masses = masses
        self.frame = frame
    
    @property
    def n_focals(self,):
        return self._focals.shape[0]
    
    @property
    def numeric_focals(self,):
        return self.numeric_focal(None)
        
    def numeric_focal(self, i_hyc=None):
        # evaluates all focal sets to numeric values
        # raises error if nested UncertaunVariables are not frozen
        
        incremental = self.incremental
        focals = self._focals
        if i_hyc is not None:
            focals = focals[i_hyc:i_hyc + 1]
        n_focals = len(focals)
        numeric_focals = np.empty((n_focals, 2))
        for i, (incvar, lbound, ubound) in enumerate(focals): # iterates over rows
            
#             print(i, (incvar, lbound, ubound))
            if isinstance(lbound, UncertainVariable):
                if not lbound.frozen:
                    raise RuntimeError(f"Variable {lbound.name} is not frozen.")
                lbound = lbound.value
                            
            if isinstance(ubound, UncertainVariable):
                if not ubound.frozen:
                    raise RuntimeError(f"Variable {ubound.name} is not frozen.")
                ubound = ubound.value
            elif np.isnan(ubound): # crisp set / interval / singleton
                if incremental: ubound = 0
                else: ubound = lbound
                
            if isinstance(incvar, UncertainVariable):
                if not incvar.frozen:
                    raise RuntimeError(f"Variable {incvar.name} is not frozen.")
                incvar = incvar.value
    
#             print(i, numeric_focals.shape, lbound, ubound, incvar)
            if incremental and np.isnan(incvar):
                numeric_focals[i, :] = (lbound, lbound + ubound)
            elif incremental:
                numeric_focals[i, :] = (incvar + lbound, incvar + ubound)
            else:    
                numeric_focals[i, :] = (lbound, ubound)
                
        return numeric_focals
    
    @property
    def _children(self):
        if self.is_poly:
            return self._focals.ravel()
        else:
            return ()
        
    def support(self, *args, **kwargs):
        incremental = self.incremental
        focals = self._focals
        supp = [np.infty, -np.infty]
        for incvar, lbound, ubound in focals: # iterates over rows
            
            if isinstance(lbound, UncertainVariable):
                lsupp = lbound.support(*args, **kwargs) 
            else:
                lsupp = (lbound, lbound)
                
            if isinstance(ubound, UncertainVariable):
                ubound = ubound.support(*args, **kwargs)[1]
            elif np.isnan(ubound):  # crisp set / interval / singleton
                ubound = lsupp[1]
                
            if isinstance(incvar, UncertainVariable):
                incsupp = incvar.support(*args, **kwargs)
            else:
                incsupp = (incvar, incvar)
    
            if incremental and isinstance(incvar, UncertainVariable):
                lbound = incsupp[0] + lsupp[0]
                ubound += incsupp[1]
            elif incremental:
                lbound = lsupp[0]
                ubound += lsupp[1]
            else:    
                lbound = lsupp[0]
            
            supp[0] = min(supp[0], lbound)
            supp[1] = max(supp[1], ubound)

        if np.isnan(supp).any():
            raise RuntimeError(f"Variable {self} has invalid support. Check your definitions")
        return supp
    
    def rvs(self, size=1, *args, **kwargs):
        # get the support
        supp = self.support(*args, **kwargs)
        
        return scipy.stats.uniform.rvs(supp[0], supp[1] - supp[0], size=size, **kwargs)
    
    def mass(self, value=None):
        # only single values may be provided        
        # value may be in more than one focal set, bel and pl are better used instead
        # may return multiple or no mass values
        numeric_focals = self.numeric_focals
        
        masses = self.masses
        inds = np.logical_and(numeric_focals[:,0] >=value,
                              numeric_focals[:,1] <=value)
        return masses[inds]
    
    def __repr__(self,):
        return f'<MF: {self.name}>'



    
def approximate_out_intervals(output, hyc_dat_inds):
    n_hyc = hyc_dat_inds.shape[0]
    # for each input hypercube get the output focal set of the output
    intervals = np.full((n_hyc, 2), np.nan)
    for i_hyc in range(n_hyc):
        selector = hyc_dat_inds[i_hyc, :]
        if selector.any():
            this_output = output[selector]
            intervals[i_hyc, :] = np.min(this_output), np.max(this_output)
        else:
            pass
            #print('Encountered empty hypercube -> no focal set assigned.')
    return intervals 

# def optimize_out_intervals(mapping, arg_vars, hyc_foc_inds, vars_epi, vars_ale=None, fun_kwargs=None):
#
#     print('This function is not implemented correctly, currently a placeholder, a wrapper must be written to account for arg_vars')
#     '''
#     This function may either be used to quantify incompleteness or imprecision:
#         incompleteness: 
#             mapping is stat_fun, needs stochastic samples and weights, weights are obtained by freezing aleatory variables in each optimization step (need a wrapper around stat_fun)
#             arg_vars is a dict mapping argument names to stochastic samples and weights
#             hyc_foc_inds are inc_hyc_foc_inds
#             vars_epi are vars_inc
#             vars_ale are not needed?
#             fun_kwargs are additional function keyword arguments, e.g. histogram bins, etc.
#         imprecision:
#             mapping is model function
#             arg_vars is a dict mapping argument names to variable names
#             hyc_foc_inds are imp_hyc_foc_inds
#             vars_epi are vars_imp
#             vars_ale are not needed?
#             fun_kwargs are additional function keyword arguments, e.g. model parameters
#     bounds and initial conditions for each hypercube are obtained from vars_epi for each variable in arg_vars
#
#
#     '''
#
# #     this_out = mapping(**{arg:samples[var].iloc[ind_ale] for arg, var in arg_vars.items() if var in names_ale}, 
# #                    **{arg:samples[var].iloc[ind_epi] for arg, var in arg_vars.items() if var in names_epi},)
#     numeric_focals = [var.numeric_focals for var in vars_epi]
#     n_hyc = len(hyc_foc_inds)
#     intervals = np.full((n_hyc ,2), np.nan)
#
#     for i_hyc in range(n_hyc):
#
#         bounds = [...]
#         init = [np.mean(bound) for bound in bounds]
#
#         resl = scipy.optimize.minimize(lambda x, args: mapping(*x, args), init, fun_args, bounds = bounds)
#         resu = scipy.optimize.minimize(lambda x, args: -mapping(*x, args), init, fun_args, bounds = bounds)
#
#         intervals[i_hyc, :] = [resl.fun, -resu.fun]
#
#     return intervals

def compute_belief(focals, masses, cumulative=False, bins=None):
    if bins is None:
        bins = np.sort(focals.ravel())
    elif not isinstance(bins, (np.ndarray, list, tuple)):
        # assuming bins = nbins -> int: number of bins
        bins = np.linspace(focals.min(), focals.max(), int(bins))

    bel = np.zeros(bins.size)
    pl = np.zeros(bins.size)
    q = np.zeros(bins.size)

    # Bins are sets A
    # Focals are sets B or C
    for i in range(bins.size - 1):
        # set A
        lbin, ubin = bins[i:i+2]
        if cumulative:
            lbin = focals.min()

        # get all sets B that are a subset of A
        # that means the lower boundary of B must be greater or equal than the lower boundary of A
        # and similarly for the upper boundary
        belinds = np.logical_and(focals[:,0] >= lbin,
                                 focals[:,1] <= ubin)
        bel[i] = np.sum(masses[belinds])
        # get all sets B that intersect with A
        # that means the lower boundary of B must not be higher (= must be strictly lower) than the upper boundary of A (B entirely outside of A to the right)
        # and the upper boundary of B must not be lower (= must be strictly higher) than the lower boundary of A (B  entirely outside of A to the left)
        plinds = np.logical_and(focals[:,0] < ubin, 
                                focals[:,1] > lbin, )
        pl[i] = np.sum(masses[plinds])

        # get all sets B that are a superset of A   (=A is a subset of B)
        # that means the lower boundary of A must be greater or equal to the lower boundary of B
        # and similarly for the upper boundary
        qinds = np.logical_and(focals[:,0] <= lbin,
                               focals[:,1] >= ubin)   
        q[i] = np.sum(masses[qinds])

    bel[-1]=bel[-2]
    pl[-1]=pl[-2]
    q[-1]=q[-2]
    
    return bins, bel, pl, q

def aggregate_mass(focals_stats, hyc_mass, nbin_fact=1, cum_mass=False):
     
    n_stat, n_hyc, _ = focals_stats.shape 
    n_bins_bel = np.ceil(np.sqrt(n_hyc) * nbin_fact).astype(int)
    bins_bel = np.linspace(np.nanmin(focals_stats), np.nanmax(focals_stats), n_bins_bel)
    bel_stats = np.empty((n_stat, n_bins_bel))
    pl_stats = np.empty((n_stat, n_bins_bel))
    q_stats = np.empty((n_stat, n_bins_bel))    
    for i_bin in range(n_stat):
    
        # compute belief, plausibility & commonality -belief functions for each aleatory bin
        _, bel,pl,q = compute_belief(focals_stats[i_bin, :, :], hyc_mass, cumulative=cum_mass, bins=bins_bel)
        bel_stats[i_bin, :] = bel
        pl_stats[i_bin, :] = pl
        q_stats[i_bin, :] = q

    return bel_stats, pl_stats, q_stats, bins_bel

def plot_focals(focals, mass, ax, highlight=None):
    cm=0
    for i,((l,r),m) in enumerate(zip(focals,mass)): 
        if i == highlight:
            ax.bar(l,m*0.9,(r-l),bottom = cm+0.05*m, align='edge', color='lightcoral', edgecolor='black')
        else:
            ax.bar(l,m*0.9,(r-l),bottom = cm+0.05*m, align='edge', color='lightgrey', edgecolor='black')
        cm += m
#     ax.set_xticks([0,0.5,1,1.5,2])
#     ax.set_xticklabels(['','','','','',])


    return ax

def plot_grid(df, output='y'):
    global hue_norm
    hue_norm = matplotlib.colors.Normalize(3/2*np.min(df[output])-1/2*np.max(df[output]),np.max(df[output]))
    grid = sns.pairplot(df, hue=output, diag_kind='scatter', corner=True, markers='.', plot_kws={'s':25, 'hue_norm':hue_norm}, palette='cubehelix', )
    grid.fig.set_size_inches(5.92,5.92)
    grid.fig.set_dpi(150)
    grid.fig.subplots_adjust(top=0.97,bottom=0.08, left=0.12, right=0.9)
    grid.fig.align_labels()
    for i, x_var in enumerate(grid.x_vars):
        
        grid.axes[i,i].set_axis_off()
        ax = grid.axes[i,i].twinx()
        sns.scatterplot(data=df, x=x_var,y=output, hue=output, markers='.',s=5, palette=grid._orig_palette, hue_norm=hue_norm, ax=ax, legend=False)
        grid.axes[i,i] = ax
    global rectangles
    rectangles=[]
    global scatters
    scatters=[]
    return grid

def plot_hyc_grid(df_hyc, grid, output='y', maxx=None, minx=None, out_up=None, out_low=None):
    # n_vars_imp = inputs_hyc.shape[1]
    # df_hyc = pd.DataFrame(inputs_hyc, columns=[f'$x_{i}$' for i in range(n_vars_imp)])
    # df_hyc['$y$']=outputs_hyc
    
    global scatters
    global hue_norm
    create_legend = not scatters
    for obj in scatters:
        obj.remove()
    scatters=[]
    for i, x_var in enumerate(grid.x_vars):
        for j, y_var in enumerate(grid.y_vars):
            if i==j: continue
            ax = grid.axes[j,i]
            if ax is None: continue
            sns.scatterplot(data=df_hyc, x=x_var,y=y_var, hue=output, 
                            markers='.',s=3, palette=grid._orig_palette, 
                            hue_norm=hue_norm, ax=ax, legend=False)
            scatters.append(ax.collections[-1])
            if maxx is not None:
                p=ax.scatter(maxx[i],maxx[j], marker='x',color='k')
                scatters.append(p)
            if minx is not None:
                p=ax.scatter(minx[i],minx[j], marker='x',color='k')
                scatters.append(p)

        ax = grid.axes[i,i]

        sns.scatterplot(data=df_hyc, x=x_var,y=output, hue=output, 
                        markers='.',s=3, palette=grid._orig_palette, 
                        hue_norm=hue_norm, ax=ax, legend=False)
        scatters.append(ax.collections[-1])
        if maxx is not None and out_up is not None:
            p=ax.scatter(maxx[i],out_up, marker='x',color='k')
            scatters.append(p)
        if minx is not None and out_low is not None:
            p=ax.scatter(minx[i],out_low, marker='x',color='k', label='sampled opt.')
            scatters.append(p)
        else:
            p = None
        
    if create_legend:
        if p is not None:
            grid.fig.legends[0].legendHandles.append(p)
        grid.fig.legend(handles=grid.fig.legends[0].legendHandles, title=output)
        grid.fig.legends[0].remove()

def plot_opt_res_grid(grid, vars_opt,
                      x_low, x_up,
                      focals,
                      out_min, out_max,
                      out_low, out_up):
    
    opt_var_inds=np.cumsum(vars_opt) - 1
    global rectangles
    
    create_legend = not rectangles
    
    for obj in rectangles:
        obj.remove()
    rectangles = []
    for i,var1 in enumerate(grid.x_vars):
        # opt_var_ind1 = opt_var_inds[i]
        foc_1=focals[i]
        # (x_low[i], x_up[i]) should be foc1 
        if not vars_opt[i]: 
            xl1 = foc_1[0]
            xu1 = foc_1[1]
        else:
            xl1 = x_low[0, i]
            xu1 = x_up[0, i]
        for j,var2 in enumerate(grid.y_vars):
            
            if var1==var2: continue
            ax = grid.axes[j,i]
            if ax is None: continue
            
            # opt_var_ind2 = opt_var_inds[j]
            foc_2=focals[j]
            
            # (x_low[j], x_up[j]) should be foc2 
            if not vars_opt[j]: 
                xl2 = foc_2[0]
                xu2 = foc_2[1]
            else:
                xl2 = x_low[0, j]
                xu2 = x_up[0, j]
                
            # plot nearest neighbor of optimal point
            p=ax.scatter(xl1,xl2, marker='o',edgecolors='k',color='none')
            rectangles.append(p)
            
            p=ax.scatter(xu1,xu2, marker='o',edgecolors='k',color='none')
            rectangles.append(p)

            # plot design hypercube
            p=ax.add_patch(matplotlib.patches.Rectangle((foc_1[0], foc_2[0]), foc_1[1] - foc_1[0], foc_2[1] - foc_2[0], fill=False, ls='dashed', color='k', alpha=0.5))
            rectangles.append(p)

        #plot optimal hypercube
        ax = grid.axes[i, i]

        # plot design hypercube
        p=ax.add_patch(matplotlib.patches.Rectangle((foc_1[0], out_min), foc_1[1] - foc_1[0], out_max - out_min, fill=False, ls='dashed', color='k', alpha=0.5))
        rectangles.append(p)

        # plot nearest neighbor of optimal point
        p=ax.scatter(xl1,out_low, marker='o',edgecolors='k',color='none')
        rectangles.append(p)
        p=ax.scatter(xu1,out_up, marker='o',edgecolors='k', label='appr. opt.',color='none')
        rectangles.append(p)
    if create_legend:
        grid.fig.legends[0].legendHandles.append(p)
        grid.fig.legend(handles=grid.fig.legends[0].legendHandles, title='y')
        grid.fig.legends[0].remove()
 
class PolyUQ(object):
    
    def __init__(self, vars_ale, vars_epi, dim_ex='cartesian'):
        '''
        primary / secondary (hyper-) variables
        
        to take into account variables that share samples they can be named equally
        e.g. to create a a precise variable in an imprecision context
        
        Parameters:
        -----------
            vars_epi: list of epistemic variables (objects of UncertainVariable class) 
                   (uncertainty quantification by interval optimization [approximated by QMCS])
            vars_ale: list of aleatory variables (objects of UncertainVariable class) 
                   (uncertainty quantification by quasi Monte-Carlo methods)
        '''
        for var in vars_ale:
            assert isinstance(var, UncertainVariable)
        for var in vars_epi:
            assert isinstance(var, UncertainVariable)
        
        assert dim_ex in ['cartesian', 'hadamard', 'vacuous']
        
        # freeze the variable sets in tuples, to avoid later alterations which might break all kinds of things
        self.vars_ale = tuple(vars_ale)
        self.vars_epi = tuple(vars_epi)
        self.vars_imp = tuple([var for var in vars_epi if var.primary])
        self.vars_inc = tuple([var for var in vars_epi if not var.primary])
        self._hyc_hyp_vars = None
        
        self.dim_ex = dim_ex
                  
        self.N_mcs_ale = None
        self.N_mcs_epi = None
        self.percentiles = None
        self.seed = None
        self.var_supp = None
        self.inp_samp_prim = None
        self.inp_suppl_ale = None
        self.inp_suppl_epi = None
                          
        self.fcount = None
        self.loop_ale = None
        self.loop_epi = None
        self.out_name = None
        self.out_samp = None
        self.out_valid = [-np.infty, np.infty]
                          
        self.imp_foc = None
        self.val_samp_prim = None
        self.intp_errors = None
        self.intp_exceed = None
        self.intp_undershot = None
                          
        self.focals_stats = None
        self.focals_mass = None
        
    def sample_qmc(self, N_mcs_ale=1000000, N_mcs_epi=100, percentiles=(0.0001, 0.9999), 
                   sample_hypercubes=False, seed=None, check_sample_sizes=True, **kwargs):
        '''
        A function to generate quasi Monte Carlo samples for mixed aleatory-epistemic uncertainty quantification
        
        A Halton sequence of all variables is generated over the support within 
        user-defined percentiles of each variable. The computation of support 
        takes into account polymorphic uncertainties. For details see doc for 
        UncertainVariable and their subclasses. 
        
        Input sequences are stored separately as class variables according to the 
        type of input variables:
            * primary, both aleatory and epistemic, pd.DataFrame (N_vars_prim, max(N_mcs_*))
            * secondary epistemic (hypervariables), pd.DataFrame (N_vars_epi_noprim, max(N_mcs_*))
            * secondary aleatory (hypervariable), pd.DataFrame (N_vars_ale_noprim, max(N_mcs_*))
        Only primary input samples are needed for uncertainty propagation. 
        Secondary / hyper-variables can be treated in post-processing as the 
        mapping function does not depend directly on them. The respective input
        sequences may be used in post-processing where sampling methods are 
        employed to ensure consistent and "independent" low-discrepancy sequences.
        
        Equally named variables may share their samples, e.g. for allowing precise
        inputs in an imprecision context. However, that may lead to unexpected 
        results, when sampling. The support of an imprecise variable is the 
        support of the incremental base variable +- the support of the incremental
        delta variables. Sampling the imprecise variable over this support and 
        then using these samples later to freeze the incremental base variable 
        with the same name effectively extents the support beyond the  specified 
        percentiles. However, sampling over the support of the base variable (child_var) 
        effectively reduces the support of the imprecise variable. Another alternative
        would be to take the above into account when computing eistemic hypercube
        extents, which would add a lot of error-prone logic. -> Just live with
        this minor imperfection.
        
        Probabilities of stochastic samples may be integrated numerically
        from the RandomVariables' probability densities using the given input samples.
        
        UNTESTED:
        To enrich a sample set, this function can simply be called again, with identical inputs. Ensure 
        to provide the same seed, as stored as class variable from the previous function call.

        Parameters:
        -----------
            N_mcs_* : int
                number of quasi Monte Carlo Samples to use
            percentiles: 2-tuple
                the percentiles to use for computing support boundaries of input variables
            seed: int
                a random seed to initialize the random number generator to enable
                repeatable sequence generation and / or enrichment of input samples
            check_sample_size: bool
                Whether to check the sample sizes for approximation of each epistemic
                hypercube (might be time consuming for a high number of stochastic samples)
        '''
        vars_epi = self.vars_epi
        vars_ale = self.vars_ale 
        
        vars_ale_prim = [var for var in vars_ale if var.primary]
        vars_epi_prim = list(self.vars_imp)
        
        all_vars_prim = vars_ale_prim + vars_epi_prim # needed to fix positions of variables for indexing/assigning/accessing corresponding sequences
        
        # determine, if primary variables are all the same type, or if mixed type inputs are present
        loop_ale = np.any([var.primary for var in vars_ale])
        loop_epi = np.any([var.primary for var in vars_epi])
        
        def traverse_children(var):
            # recursively add all child variables to the all_vars dictionary
            for child_var  in var._children:
                if not isinstance(child_var, UncertainVariable): continue
                if not child_var.name in all_vars:
                    all_vars[child_var.name] = child_var
                traverse_children(child_var)
            return
        
        all_vars = {}
        # taking into account variables that share samples / are named equally
        for var in all_vars_prim:
            all_vars[var.name] = var
            traverse_children(var)
        
        all_vars = list(all_vars.values()) 
        n_vars = len(all_vars)
        logger.info(f'Establishing support domain for all variables" {[var.name for var in all_vars]}')
        # get (truncated) support (define upper and lower bounds, e.g. 99.99 % quantiles, 0.01% quantiles)     
        # define equivalent uniform distributions
        vars_unif =[]
        var_supp = pd.DataFrame(np.empty((2, n_vars)), columns=[var.name for var in all_vars])
        for var in all_vars:
            supp = var.support(percentiles)
            assert np.all(np.abs(supp)!=np.infty)
            var_supp[var.name] = supp
            # print(var.name, supp)
            if isinstance(var, RandomVariable):    
                if isinstance(var.dist_fun, scipy.stats.rv_discrete):
                    vars_unif.append(scipy.stats.randint(supp[0], supp[1] + 1))
                else:
                    vars_unif.append(scipy.stats.uniform(supp[0], supp[1] - supp[0]))
            else:
                vars_unif.append(scipy.stats.uniform(supp[0], supp[1] - supp[0]))
                
        if kwargs.get('supp_only', False):
            self.var_supp = var_supp
            return
        
        # sampling parameters
        N_mcs = max(N_mcs_ale,N_mcs_epi)
        
        # check sample sizes
        n_vars_imp = len(vars_epi_prim)
        if N_mcs_epi < 2.5**n_vars_imp:
            logger.warning(f'Epistemic sample size {N_mcs_epi} may not suffice '
                           f'interval optimization (imprecision) over {n_vars_imp}'
                           f' variables. As a rule of thumb to conceptually cover '
                           f'all hypercube corners and midpoints 3^n = {3**n_vars_imp} samples are required.')
        if N_mcs_ale < 10000:
            logger.warning(f'Aleatory sample size {N_mcs_ale} may not suffice '
                           'depending on the statistic to be applied. Large confidence intervals may occur.')
        
        if seed is None:
            seed = np.random.randint(np.iinfo(np.int32).max)
        
        logger.info(f"Generating low-discrepancy sequences for all variables (scramble seed: {seed})... ")
        # sample N_mcs samples from a Halton Sequence and transform to uniform bounds    
        seed_seq = np.random.SeedSequence(seed).spawn(1)[0]
        seed_seq = np.random.default_rng(seed_seq)
        engine = scipy.stats.qmc.Halton(n_vars, seed=seed_seq)
        samples = engine.random(N_mcs)
        
        
        # check discrepancy for each pair of inputs by leave-one-out validation
        # print(scipy.stats.qmc.discrepancy(samples))
        discrepancies = np.zeros((n_vars,n_vars))
        for i in range(n_vars):
            for j in range(i+1,n_vars):
                discrepancies[i,j] = scipy.stats.qmc.discrepancy(samples[:,[i,j]])
        
        triuinds=np.triu_indices(n_vars, 1, n_vars)
        for i in range(n_vars):
            for j in range(i+1,n_vars):
                thisd = discrepancies[i,j]
                discrepancies[i,j] = 0
                mu = np.mean(discrepancies[triuinds])
                sig  = np.sqrt(np.var(discrepancies[triuinds]))
                if np.abs(mu-thisd)>3*sig:
                    logger.warning(f"The sequence's discrepancy for variables {all_vars[i].name} and {all_vars[j].name} is {(thisd - mu)/sig:1.2g} std above average.")
                discrepancies[i,j] = thisd
        
        samples = pd.DataFrame(samples, columns=[var.name for var in all_vars])
        
        # transform to uniform (could also be done by qmc.scale(low,high)
        for i,var in enumerate(all_vars):
            # print(var, vars_unif[i].ppf([0,1]))
            this_samples = samples[var.name]
            if isinstance(var, MassFunction):
                # scale (0,1) to [0,1]
                this_samples -= this_samples.min()
                this_samples /= this_samples.max()
            # scale to physical boundaries
            samples[var.name] = vars_unif[i].ppf(this_samples)
                
        
        inp_samp_prim = samples[[var.name for var in all_vars_prim]]
        inp_suppl_ale = samples.iloc[:N_mcs_ale][[var.name for var in vars_ale if not var.primary]]
        inp_suppl_epi = samples.iloc[:N_mcs_epi][[var.name for var in vars_epi if not var.primary]]
        
        # check the number of samples per focal set
        if check_sample_sizes:
            self.check_sample_sizes(vars_epi_prim, samples, N_mcs_ale, N_mcs_epi)
        
        self.N_mcs_ale = N_mcs_ale
        self.N_mcs_epi = N_mcs_epi
        
        self.percentiles = percentiles
        self.seed = seed
        
        self.loop_ale = loop_ale
        self.loop_epi = loop_epi
        
        self.var_supp = var_supp
        self.inp_samp_prim = inp_samp_prim
        self.inp_suppl_ale = inp_suppl_ale
        self.inp_suppl_epi = inp_suppl_epi
        
        return 
    
    def sample_stochastic(self, N_mcs):
        '''
        all_vars_prim
        loop_ale = True
        loop_epi = False
        
        initialize samples pd.DataFrame? (n_vars_prim, N_mcs)
        
        compute all_vars (without duplicates)
        initialze rng
        
        for n_ale in range(N_mcs):
            sample and freeze var_inc (uniform) pass 'random_state'=rng
            sample and freeze var_ale (pass percentiles) pass 'random_state'=rng
            sample var_imp (uniform) pass 'random_state'=rng
        
        split prim, suppl_ale, suppl_epi
        
        save variables and store class
            
        '''
        
    def to_data_manager(self, title, **kwargs):
        '''
        combining PolyUQ and DataManager
        options: 
            1) create a new class and copy the relevant parts from both
            2) create a new class, subclassing both
            3) create to/from methods in PolyUQ (possibly stripping of post-processing methods from DataManager into a different class), that would however eliminate the possibility for mapping-based interval optimization
        
        go with option 3:
        in PolyUQ we need to create input samples as an xarray, save and load with DataManager
        
        keep in mind staged propagation to allow reusing the first pure aleatory/epistemic part of expensive model runs in a mixed-aleatory-epistemic grid
        in a mixed grid, each aleatory sample is combined with all epistemic samples and vice versa
        independent parts of the mapping function can thus be re-used in the combination part
        these parts are most likely only the first few steps of the mapping function
        but some logic is needed to ensure these parts are already computed, before any other samples try to use them
        so basically, we have to pre-compute them in a separate run
        assign aleId and epiId and save the results under aleId_epiId_stageNr
        create separate mapping functions for pre-computation and actual run
        
        where would we generate the grid? 
            in PolyUQ: that would not require any new logic from DataManager, 
                return three DataManager databases: two for pre-computation (ale/epi) and one for final propagation
            in DataManager: We need the logic in DataManager, i.e. is expansion necessary
        
        in PolyUQ also create a method to load output sample arrays created in DataManager
        '''
        
        logger.info("Exporting samples to DataManager for distributed propagation...")
        
        # hypervariables (polymorphic outer variables) are only relevant in post-processing
        
        vars_epi = self.vars_epi
        vars_ale = self.vars_ale
        N_mcs_ale = self.N_mcs_ale
        N_mcs_epi = self.N_mcs_epi
        
        inp_samp_prim = self.inp_samp_prim
        
        # determine, if primary variables are all the same type, or if mixed type inputs are present
        loop_ale = self.loop_ale
        if not loop_ale:
            N_mcs_ale = 1
        loop_epi = self.loop_epi
        if not loop_epi:
            N_mcs_epi = 1
        
        inds_ale, inds_epi = np.mgrid[0:N_mcs_ale, 0:N_mcs_epi]
        inds_ale, inds_epi = inds_ale.ravel(), inds_epi.ravel()
        
        ids_ale = np.array([str(uuid.uuid4()).split('-')[-1] for _ in range(N_mcs_ale)])
        ids_epi = np.array([str(uuid.uuid4()).split('-')[-1] for _ in range(N_mcs_epi)])
        
        names_ale = [var.name for var in vars_ale if var.primary]
        names_epi = [var.name for var in vars_epi if var.primary]
        names_grid = names_ale + names_epi
        
        arrays_ale = [inp_samp_prim[name].iloc[:N_mcs_ale] for name in names_ale]
        arrays_epi = [inp_samp_prim[name].iloc[:N_mcs_epi] for name in names_epi]
        
        arrays_grid = [inp_samp_prim[name].iloc[inds_ale] for name in names_ale]
        arrays_grid += [inp_samp_prim[name].iloc[inds_epi] for name in names_epi]
        ids_grid = np.array(list(map('_'.join, zip(ids_ale[inds_ale], ids_epi[inds_epi]))))
        
        arrays_ale.append(ids_ale)
        arrays_epi.append(ids_epi)
        arrays_grid.append(ids_grid)
        for names in (names_ale, names_epi, names_grid):
            names.append('ids')
        # return
        manager_ale = DataManager(title + '_ale', entropy=self.seed, **kwargs)
        manager_ale.provide_sample_inputs(arrays_ale, names_ale)
        
        manager_epi = DataManager(title + '_epi', entropy=self.seed, **kwargs)
        manager_epi.provide_sample_inputs(arrays_epi, names_epi)
        
        manager_grid = DataManager(title, entropy=self.seed, **kwargs)
        manager_grid.provide_sample_inputs(arrays_grid, names_grid)
        
        self.fcount = N_mcs_ale * N_mcs_epi
        
        return manager_grid, manager_ale, manager_epi
    
    def from_data_manager(self, manager=None, ret_name=None, ret_ind=None,out_ds=None):
        '''
        May be called repeatedly when new samples have been propagated to update
        out_samp and subsequently update imp, inc and so on
        
        instead of manager, out_ds may be passed, to avoid reopening and closing
        db if multiple ret_names/ ret_inds are to be extracted in a loop
        
        ret_ind: dict 
            {'dim':ndindex, ...}
        '''
        
        logger.info(f"Importing propagated samples from DataManager using the output variable {ret_name}")
        loop_ale = self.loop_ale
        loop_epi = self.loop_epi
            
        assert isinstance(ret_ind, dict)
        N_mcs_ale = self.N_mcs_ale
        N_mcs_epi = self.N_mcs_epi
        if not loop_ale:
            N_mcs_ale = 1
        if not loop_epi:
            N_mcs_epi = 1
        
        if manager is not None:
            with manager.get_database(database='out', rw=False) as out_ds:
                '''
                need to re-construct grid array from out_ds[ret_name]
                how to handle intermediate processing? to be done in manager.process_samples()
                
                how to handle multi-valued outputs for polymorphic uncertainty quantification processing? 
                e.g. modeshapes: either do the IvO individually or introduce some processed meassure e.g. MAC, strain energy, to work on
                
                do we need to keep indices around, just in case? don't think so:
                interval optimization has to be performed for each output quantity individually
                
                get ret_name from out_ds
                construct the grid from inds_ale (row indices) and inds_epi (column indices)
                return out_samp
                '''
                assert out_ds.entropy == self.seed
                out_flat_all = out_ds[ret_name]
        elif out_ds is not None:
            out_flat_all = out_ds[ret_name]
        else:
            raise RuntimeError("Either manager or out_ds must be provided.")
            
        if ret_ind is not None:
            out_flat = out_flat_all[ret_ind]
        else:
            out_flat = out_flat_all
        assert out_flat.ndim == 1
        
        out_grid = np.empty((N_mcs_ale, N_mcs_epi))
        # .flat returns a  C-style order flat iterator over the array
        out_grid.flat = out_flat
        
        if np.any(np.isnan(out_flat)):
            indices=np.where(np.any(np.isnan(out_grid),axis=1))
            logger.warning(f'Output contains NaNs beginning at the {indices[0][0]}th aleatory sample, expect subsequent routines to behave unexpectedly.')
            # print(indices, len(indices),np.any(np.isnan(out_grid),axis=1).shape, out_grid.shape)
        if np.any(np.isinf(out_flat)):
            indices=np.where(np.any(np.isinf(out_grid),axis=1))
            logger.warning(f'Output contains +/- infty beginning at the {indices[0][0]}th aleatory sample, expect subsequent routines to behave unexpectedly.')
        if not out_flat.dtype.kind in set('buif'):
            logger.warning(f'Output dtype ({out_flat.dtype}) may cause trouble.')
        
        if self.out_name is not None:
            assert self.out_name == f'{ret_name}-{".".join(str(e) for e in ret_ind.values())}'
        self.out_name = f'{ret_name}-{".".join(str(e) for e in ret_ind.values())}'
        self.out_samp = out_grid
        self.out_valid = [out_flat_all.min(skipna=True).item(), out_flat_all.max(skipna=True).item()]
            
            
    def propagate(self, mapping, arg_vars):
        '''
        An input sample lattice using aleatory input samples in rows and 
        epistemic input samples in columns is used for uncertainty propagation.
        The output sample lattice can be used for stochastic/statistical analysis, 
        approximate interval optimization, sensitivity analyses or surrogate modeling. 
        
        Parameters:
        -----------
            mapping: callable
                that takes keyword arguments as specified in arg_vars
            arg_vars: dictionary 
                mapping the function arguments (keys) to the variable names (values)
         
        Returns:
        --------
            out: np.ndarray (N_mcs_ale, N_mcs_epi)
                sequence of output samples
        '''
        # hypervariables (polymorphic outer variables) are only relevant in post-processing
        
        vars_epi = self.vars_epi
        vars_ale = self.vars_ale 
        N_mcs_ale = self.N_mcs_ale
        N_mcs_epi = self.N_mcs_epi
        
        inp_samp_prim = self.inp_samp_prim
        
        # determine, if primary variables are all the same type, or if mixed type inputs are present
        loop_ale = self.loop_ale
        loop_epi = self.loop_epi
    
        # propagation
        logger.info("Propagating mapping function...")
        all_vars = vars_ale + vars_epi
        for var_name in arg_vars.values():
            for var in all_vars:
                if var.primary and var.name == var_name: break
            else:
                raise RuntimeError(f'Variable {var} is not marked as primary but used as a function argument')
        for var in all_vars:
            if not var.primary: continue
            for var_name in arg_vars.values():
                if var.name == var_name: break
            else:
                raise RuntimeError(f'Variable {var} is marked as primary but not used as a function argument')
        
        names_ale = [var.name for var in vars_ale if var.primary]
        names_epi = [var.name for var in vars_epi if var.primary]
        
        arg_vars_ale = {arg: var for arg, var in arg_vars.items() if var in names_ale}
        arg_vars_epi = {arg: var for arg, var in arg_vars.items() if var in names_epi}
        
        inp_samp_ale = inp_samp_prim[list(arg_vars_ale.values())].values
        inp_samp_epi = inp_samp_prim[list(arg_vars_epi.values())].values
        
        if loop_ale and loop_epi:
            out_samp = np.zeros((N_mcs_ale, N_mcs_epi))
        elif loop_ale:
            out_samp = np.zeros((N_mcs_ale, 1))
        elif loop_epi:
            out_samp = np.zeros((1, N_mcs_epi))
        
        fcount = 0
        # for ind_ale, ind_epi in np.nditer([inds_alef, inds_epif]):
            # this_out = mapping(**{arg:inp_samp_prim.iloc[ind_ale][var] for arg, var in arg_vars.items() if var in names_ale}, 
            #                    **{arg:inp_samp_prim.iloc[ind_epi][var] for arg, var in arg_vars.items() if var in names_epi},)
        for ind_ale in range(out_samp.shape[0]):
            for ind_epi in range(out_samp.shape[1]):
            
                this_out = mapping(**{arg:inp_samp_ale[ind_ale, ind_var] for ind_var, arg in enumerate(arg_vars_ale.keys())}, 
                                   **{arg:inp_samp_epi[ind_epi, ind_var] for ind_var, arg in enumerate(arg_vars_epi.keys())},)
                fcount += 1
                
                out_samp[ind_ale, ind_epi] = this_out
                
        if loop_ale and loop_epi:
            logger.info(f'Mapping function was called {fcount} times in a mixed aleatory epistemic loop.')
        elif loop_ale:
            logger.info(f'Mapping function was called {fcount} times in a pure aleatory loop.')
        elif loop_epi:
            logger.info(f'Mapping function was called {fcount} times in a pure epistemic loop.')
            
        self.out_name = 'out'
        self.out_samp = out_samp
        self.fcount = fcount
        self.out_valid = [np.nanmin(out_samp), np.nanmax(out_samp)]
        
        return out_samp#, p_weights
    
    def probabilities_imp(self, i_imp=None):
        
        # many secondary aleatory variables may be provided and sampled
        # however they may not necessarily be used 
        #     e.g. two different focal sets to account for imprecision may be provided for an input variable
        #     both focal sets are bound by a different pair of aleatory variables
        #     computing the joint pdf of all aleatory variables is wrong since for each focal set only a pair of two is used
        # probability weights of primary aleatory variables
        #     if non-polymorphic -> can be readily computed
        #     if polymorphic -> depend on epistemic samples
        # for imprecise variables, probabilities are not changing, with epistemic values 
        # -> p_weights are independent, not changing within epistemic loop
        
        # pass i_imp to compute weights only for imprecise hypercube number i_imp
        
        logger.info("Computing aleatory probability weights...")
        # compute probabilities for approximately equally spaced (due to low-discrepancy sampling) bins
        # TODO: theoretically integration has to be performed
        
        # Variables
        vars_ale = self.vars_ale
        hyc_hyp_vars = self.hyc_hyp_vars # no-imp: probably empty
        
        # Epistemic Hypercubes
        imp_hyc_foc_inds = self.imp_hyc_foc_inds # no-imp: is a list containing a single empty tuple
        
        if i_imp is None:
            n_imp_hyc = len(imp_hyc_foc_inds) # no-imp: would be 1
            it_imp = range(n_imp_hyc)
        else:
            n_imp_hyc = 1
            it_imp = [i_imp]
        
        # Samples
        N_mcs_ale = self.N_mcs_ale
        
        inp_samp_prim = self.inp_samp_prim.iloc[:N_mcs_ale]
        inp_suppl_ale = self.inp_suppl_ale.iloc[:N_mcs_ale]
        
        # compute probabilities for imprecise samples
        p_weights = np.ones((N_mcs_ale, n_imp_hyc))
        all_hyp_vars = set(chain(*hyc_hyp_vars))
        # assemble probability weights from primary aleatory variables 
        for var in vars_ale:
            if var in all_hyp_vars and var.primary:
                logger.info(f'Variable {var.name} is primary and a hyper variable. Ignoring primary state for probability weights.')
                continue
            if var.primary:
                # assign weight to to all hypercubes
                p_weights *= np.repeat(var.prob_dens(inp_samp_prim[var.name])[:,np.newaxis], n_imp_hyc, axis=1)
            
        # probabilities are computed for pre-computed stochastic samples as the product of PDFs of the underlying RVs
        # each hypercube may be constructed from different hypervariables (RVs) -> has a different product probability
        # assemble probability weights from hypervariables (secondary)       
               
        hyp_dens = {}    
        for i_weight, i_imp_hyc in enumerate(it_imp):
            hyp_vars = hyc_hyp_vars[i_imp_hyc]
            for hyp_var in hyp_vars:
                if not isinstance(hyp_var, UncertainVariable): continue
                # p_weights[:, i_imp_hyc] *= hyp_var.prob_dens(inp_suppl_ale[hyp_var.name])
                if not hyp_var.name in hyp_dens: 
                    # caching probability densities if a variable is hypervariable of multiple variables
                    if hyp_var.primary:
                        hyp_dens[hyp_var.name] = hyp_var.prob_dens(inp_samp_prim[hyp_var.name])
                    else:
                        hyp_dens[hyp_var.name] = hyp_var.prob_dens(inp_suppl_ale[hyp_var.name])
                p_weights[:, i_weight] *= hyp_dens[hyp_var.name]
            # normalize
            p_weights[:, i_weight] /= np.sum(p_weights[:, i_weight])
        
        if i_imp is None:
            return p_weights
        else:
            return p_weights[:,0]
        '''
        ice_occ is a primary variable
        ice_occ is a hypervariable of ice_mass
        ice_mass is a primary variable
        
        assume ice_mass has two focal sets one precise set and one imprecise set
        for the precise focal set p_weights only result from primary aleatory variables
        for the imprecise focal set p_weights result from the aleatory hypervariables
        actually ice_occ is not a primary variable, that was just a hack
            epistemic samples are generated on the support (of ice_mass)
            for each aleatory sample, the epistemic samples are propagated
            if ice_occ==0 the model is run with all those ice_masses, even though, they are not present
            that is why ice_occ had to be passed to the mapping function and thus needed to be primary
        what to do about those cases?
            if we just use it twice, once as primary and once as hypervariable
            it's probability weight is doubled
            
        '''
        
# RBF: r will be the maximum distance between any two samples
# gaussian: e^(-epsilon*r^2)
# epsilon ist the shape parameter -> how fast gaussian kernel drops to zero, i.e. how much will each sample influence its neighbors
#     if too big interpolant will be oscillating (?)
#     if too small interpolant will not accurately follow the true behavior (?)
# epsilon is "inversely proportional to the average distance between the data points"
#
# A consequence of this choice, is that the interpolation matrix approaches the identity 
# matrix as ε → ∞ {\displaystyle \varepsilon \to \infty } leading to stability when solving 
# the matrix system. The resulting interpolant will in general be a poor approximation to 
# the function since it will be near zero everywhere, except near the interpolation points 
# where it will sharply peak – the so-called "bed-of-nails interpolant" (as seen in the 
# plot to the right).
#
# On the opposite side of the spectrum, the condition number of the interpolation matrix 
# will diverge to infinity as ε → 0 leading to ill-conditioning of the 
# system. In practice, one chooses a shape parameter so that the interpolation matrix is 
# "on the edge of ill-conditioning" (eg. with a condition number of roughly 10^{{12}} 
# for double-precision floating point).

    def estimate_imp(self, interp_fun='rbf', opt_meth='genetic', 
                     plot_res=False, plot_intp=False, print_res_ranges=False, 
                     intp_err_warn=10, # threshold in percent of interpolation domain
                     extrp_warn=5, # threshold in percent of interpolation domain
                     **kwargs):
        
        '''    
        Estimate imprecision for all aleatory samples and imprecision hypercubes
        using surrogate models / interpolators fit to pre-computed mapping
        outputs on a quasi Monte Carlo sequence / grid. 
        
        For each sample k in N_mcs_ale
            Retrieve imprecision input and output samples
            Setup surrogate model interp_fun on the unit hypercube
                 cross-validate by leave-one-out or leave-k-out depending on sample size 
            Compute imprecision interval boundaries / focal sets 
                which may depend on aleatory variables
            Generate imprecision hypercubes
                according to chosen dimension extension procedure (self.dim_ex)
            For each hypercube
                determine optimization variables (hypercube side length in the
                respective dimention > 0)
                compute output interval (by bounded optimization on interval_range)
                compute hypercube mass
                
        For reference the axis order of arrays and related variables are:
        0           1           2              3
        N_mcs_ale,  N_mcs_imp,  n_vars_imp,    intv_bound
                    n_imp_hyc,  n_vars_opt
                    S,
        n_stat (inc.), n_hyc,
        Not always all levels are present and actual dimensions change 
        accordingly, but not the order.
        
        
        Parameters:
        -----------
            interp_fun: str ['nearest', 'linear', 'rbf'] or callable
                The interpolation method to use as a surrogate. Needs to take arguments 
                unit_x (N_mcs, n_vars),  this_out (N_mcs,), **kwargs 
                
                NearestND interpolator exceeds hypercube bounds frequently during
                optimization and should only be used for rough estimates of 
                imprecision. RBF interpolator generally works very fast and reliable
                but requires parameter tuning, e.g. kernel function and its shape parameter
                to avoid extrapolation / overshoot / bed-of-nails / ill-conditioning problems.
                LinearND interpolator becomes very slow / stops working with 
                higher dimensions.
            opt_meth: str
                One of the available methods in scipy.optimize.minimize for 
                local optimization or 'genetic' (recommended) for global 
                optimization (vectorized differential evolution).
            plot_res: bool
                Show a scatterplot matrix of input and output variables. Best 
                used in an interactive jupyter environment
            plot_interp: bool
                For each hypercube, extend the scatterplot matrix with samples
                generated with the interpolator to verify good interpolation.
            print_res_ranges: bool
                Print the ranges, interval boundaries and corresponding input 
                points for each hypercube.
            intp_err_warn: float (0...100)
                Issue a warning if interpolator cross-validation error exceeds
                the given threshold in percent of the output variable range
            extrp_warn: float (0...100)
                 Issue a warning if interpolator is extrapolating by the given 
                 percentage outside of the output variable range in the obtained
                 optimum interval. However, extrapolations will always be replaced
                 by the minimum / maximum output value.
            **kwargs: dict
                Extra keywords passed to interp_fun, optimizer or used at various
                places in the code for evaluation purposes.
                
        
        Returns:
        --------
            imp_foc: np.ndarray (N_mcs_ale, n_imp_hyc, 2)
                sequence of output intervals
            val_samp_prim: np.ndarray (N_mcs_ale, n_imp_hyc, len(all_vars_prim), 2)
                input variables for each optimum found, for subsequent validation
                with the propagation mapping or adaptive resampling
            intp_errors: np.ndarray (N_mcs_ale,)
                cross-validation errors for each aleatory sample
            intp_exceed: np.ndarray (2,) # [count, exceed]
                when the interpolator has intp_exceed the maximum output value
                increment count and sum up exceedance differences over
                all aleatory samples and imprecision hypercubes
            intp_undershot: np.ndarray (2,) # [count, intp_undershot]
                when the interpolator has intp_undershot the mimimum output value
                increment count and sum up intp_undershot differences over
                all aleatory samples and imprecision hypercubes
        '''
            
        def interval_range(x, interp, temp_x, unit_bounds, temp_dists,
                    vars_opt, n_vars_opt, **kwargs):
            '''
            interpolator: 
                input is in unit space, 
                output is in physical space 
                    
            objective is to maximize interval
            i.e. minimize lower - upper -> in [-infty, 0]
            
            LinearND solutions may return nan, when outside the convex hull
                return 0
            
            NearestND solutions violate boundaries (intentionally in some cases)
            
                as there might be singular boundaries in some dimensions, it may happen
                that there are no points inside the boundaries of that dimension
                we would want to find a nearest point, that is inside the domain in as many dimensions as possible
        
                we can get the nearest point itself and compute the distance to the boundaries
                all computations must be performed in the unit cube
                get the nearest point
                for each dimension:
                    if outside bounds -> compute distance
                compute power(sum(distances**2), 1/20) -> distance in [0,1]
                distance may become > 1, when optimizer leaves boundaries
        
                multiply objective by (1-distance) -> 
                    for low/no distances keeps the objective
                    for high distances drives it towards 0
                    
            SHAPES:
            # interpolator needs input of shape (S, n_vars_imp)
            # init is shape (n_vars_imp, ) 
            # focals is shape (n_vars_imp, 2) and so is bounds and so is unit_bounds
            # optimizer needs input of shape (2 * n_vars_opt,) (minimize) or (S, 2 * n_vars_opt) (global)
            #
            # depending on the optimizer that calls interval_range
            # global: x is (2 * n_vars_opt, S), temp_x is (S, 2 * n_vars) is temp_dists
            # local: x is (2 * n_vars_opt,), temp_x is (1,n_vars_imp) is temp_dists
                
            '''
            
            dist = 0
            if x.ndim == 1: # scalar optimizer
                temp_x[:,vars_opt] = x[:n_vars_opt]
            else: # vectorized optimizer
                temp_x[:,vars_opt] = x[:n_vars_opt,:].T
            if isinstance(interp, scipy.interpolate.NearestNDInterpolator):
                nx,yl = interp.point_val(temp_x)
                
                temp_dists[:,:] = unit_bounds[:,0] - nx
                dist += np.sum(temp_dists[temp_dists>0]**2)
                temp_dists[:,:] = nx - unit_bounds[:,1]
                dist += np.sum(temp_dists[temp_dists>0]**2)
            else:
                yl = interp(temp_x)
                #if np.isnan(yl): return 0
        
            if x.ndim == 1: #scalar optimizer
                temp_x[:,vars_opt] = x[n_vars_opt:]
            else:# vectorized optimizer
                temp_x[:,vars_opt] = x[n_vars_opt:,:].T
        
            if isinstance(interp, scipy.interpolate.NearestNDInterpolator):
                nx,yh = interp.point_val(temp_x)
    
                temp_dists[:,:] = unit_bounds[:,0] - nx
                dist += np.sum(temp_dists[temp_dists>0]**2)
                temp_dists[:,:] = nx - unit_bounds[:,1]
                dist += np.sum(temp_dists[temp_dists>0]**2)
            else:
                yh = interp(temp_x)
                #if np.isnan(yh): return 0
            dist = np.power(dist, 1/20)
            # for testing purposes
        #     if dist > 1 and not kwargs.get('pr',False):
        #         print('wrapper failed, rerun with printouts enabled')
        #         interval_range(x,interp,temp_x,unit_bounds,temp_dists,vars_opt, n_vars_opt,pr=True,**kwargs)

            y = yl - yh
            y *= (1 - dist)
            y[np.isnan(y)]=0
            return y
        
        def scale(x):
            # subtraction always along the last axis (?)
            # x.shape = (n_obs, n_vars)
            return (x - x_supp[:,0]) / (x_supp[:,1] - x_supp[:,0])
        
        def unscale(x):
            return x * (x_supp[:,1] - x_supp[:,0]) + x_supp[:,0]
        
        logger.info(f"Estimating imprecision intervals by surrogate optimization ({opt_meth})...")
        
        if interp_fun in ['nearest', 'linear', 'rbf']:
            interp_fun = {'nearest': scipy.interpolate.NearestNDInterpolator,
                          'linear': scipy.interpolate.LinearNDInterpolator,
                          'rbf': scipy.interpolate.RBFInterpolator}[interp_fun]
        elif not callable(interp_fun):
            raise ValueError('interp_fun is neither in [nearest","linear","rbf"] nor a callable')
        
        # Variables
        vars_ale = self.vars_ale
        vars_imp = self.vars_imp
        all_vars_prim = [var for var in vars_ale if var.primary] + list(vars_imp)
        n_vars_imp = len(vars_imp)
        loop_ale = self.loop_ale
        
        # Epistemic Hypercubes
        imp_hyc_foc_inds = self.imp_hyc_foc_inds # no-imp: is a list containing a single empty tuple
        n_imp_hyc = len(imp_hyc_foc_inds) # no-imp: would be 1
        
        # Samples
        N_mcs_ale = self.N_mcs_ale
        N_mcs_epi = self.N_mcs_epi
        if vars_imp:
            N_mcs_imp = N_mcs_epi
        else:
            N_mcs_imp = 1
            # should be zero, but that would make hypercube_sample_indices empty
        
        inp_samp_prim = self.inp_samp_prim
        # inp_suppl_epi = self.inp_suppl_epi
        inp_suppl_ale = self.inp_suppl_ale
        
        out_samp = self.out_samp # Here we may choose from multiple output quantities, cluster output, etc.
        
        # extract underlying numpy array in the order of vars_epi for faster indexing 
        x_samp = inp_samp_prim[[var.name for var in vars_imp]].values[:N_mcs_imp, :] # shape (S,n_vars_imp)
        # TODO: eventually store it in in sample_qmc and use it here 
        x_supp = self.var_supp[[var.name for var in vars_imp]].values.T
        
        # x_supp_ = np.empty((n_vars_imp, 2))
        # x_supp_[:,0] = x_samp.min(axis=0) #shape (n_vars_imp,)
        # x_supp_[:,1] = x_samp.max(axis=0) #shape (n_vars_imp,)
        # print(x_supp, x_supp_)
        
        # quantify imprecise QMC samples for all stochastic samples
        # allocate arrays for interval optimization and subsequent statistical analyzes
        if self.imp_foc is None:
            imp_foc = np.full((N_mcs_ale, n_imp_hyc, 2), np.nan)
        elif self.imp_foc.shape[0]<N_mcs_ale:
            N_mcs_ale_prev = self.imp_foc.shape[0]
            assert self.imp_foc.shape[1] == n_imp_hyc
            imp_foc = np.full((N_mcs_ale, n_imp_hyc, 2), np.nan)
            imp_foc[:N_mcs_ale_prev,:,:] = self.imp_foc
        else:
            imp_foc = self.imp_foc
            assert np.all(imp_foc.shape[1:] == (n_imp_hyc, 2))
            
            
        # next-to-last axis corresponds to the order in all_vars_imp
        if self.val_samp_prim is None:
            val_samp_prim = np.full((N_mcs_ale, n_imp_hyc, len(all_vars_prim), 2), np.nan)
        elif self.val_samp_prim.shape[0]<N_mcs_ale:
            N_mcs_ale_prev = self.val_samp_prim.shape[0]
            assert self.val_samp_prim.shape[1] == n_imp_hyc
            assert self.val_samp_prim.shape[2] == len(all_vars_prim)
            val_samp_prim = np.full((N_mcs_ale, n_imp_hyc, len(all_vars_prim), 2), np.nan)
            val_samp_prim[:N_mcs_ale_prev,:,:,:] = self.val_samp_prim
            
        else:
            val_samp_prim = self.val_samp_prim
            assert np.all(val_samp_prim.shape[1:] == (n_imp_hyc, len(all_vars_prim), 2))
            
        # count the number of times the interpolator has extrapolated values
        if self.intp_exceed is None:
            intp_exceed = [0, 0] # count, exceed
        else:
            intp_exceed = self.intp_exceed
            
        if self.intp_undershot is None:
            intp_undershot = [0, 0] # count, exceed
        else:
            intp_undershot = self.intp_undershot
            
        if self.intp_errors is None:
            intp_errors = np.full(N_mcs_ale, np.nan)
        elif self.intp_errors.shape[0]<N_mcs_ale:
            N_mcs_ale_prev = self.intp_errors.shape[0]
            intp_errors = np.full(N_mcs_ale, np.nan)
            intp_errors[:N_mcs_ale_prev] = self.intp_errors
        else:
            intp_errors = self.intp_errors
            assert len(intp_errors) >= N_mcs_ale
        
        iter_ale = range(kwargs.pop('start_ale',0), kwargs.pop('end_ale',N_mcs_ale))
        pbar = simplePbar(n_imp_hyc * len(iter_ale))
        for n_ale in iter_ale:
            # each supplementary aleatory sample defines boundaries on imprecise variables
            #    do interval optimization using the pre-computed samples within these boundaries
            #    (pre-computed epistemic samples may be the same for each aleatory sample while only imprecise input boundaries differ)
            logger.debug(f'At sample {n_ale} out of {N_mcs_ale}')
            
            if loop_ale:
                this_out = out_samp[n_ale, :N_mcs_imp]
            else:
                this_out = out_samp[0, :N_mcs_imp]
                
            this_inp_suppl = inp_suppl_ale.iloc[n_ale]
            this_inp_prim = inp_samp_prim.iloc[n_ale]
            #
            # # freeze the aleatory variables
            # # to fix numeric focals of imprecise variables
            for var in vars_ale:
                if not var.primary:
                    var.freeze(this_inp_suppl[var.name])
                    logger.debug(f'{var}, {this_inp_suppl[var.name]}')
                else:
                    var.freeze(this_inp_prim[var.name])
                    logger.debug(f'{var}, {this_inp_prim[var.name]}')
                    
            # # assemble arg_vars for verification
            # arg_vars = kwargs.get('arg_vars', {})
            # arg_vals_l = {key:None for key,_ in arg_vars.items()}
            # arg_vals_h = {key:None for key,_ in arg_vars.items()}
            # for var in vars_ale:
            #     if var.primary:
            #         var.freeze(this_inp_prim[var.name])
            #         logger.debug(f'{var}, {this_inp_prim[var.name]}')
            #         for arg,var_ in arg_vars.items():
            #             if var_ == var:
            #                 arg_vals_l[arg] = this_inp_prim[var.name]
            #                 arg_vals_h[arg] = this_inp_prim[var.name]
            #                 break
            #         else:
            #             pass
            
            out_max = np.max(this_out)
            out_min = np.min(this_out)
            out_range = out_max - out_min
            
            x_out_max = x_samp[np.argmax(this_out),:]
            x_out_min = x_samp[np.argmin(this_out),:]
            
            if plot_res:
                df = inp_samp_prim[[var.name for var in vars_imp]].iloc[:N_mcs_imp,:]
                df[self.out_name] = this_out
                grid = plot_grid(df, self.out_name)
            
            now= time.time()
            
            interp =  interp_fun(scale(x_samp),  this_out, **kwargs)
            
            logger.debug(f'Took {time.time()-now:1.2f} s to build interpolator of type {type(interp)}.')
            
            now=time.time()
            # fit interpolator and estimate interpolator error 
            # by k-runs of leave-one-out cross validation
            # or k-fold cross validation, depending on the sample size 
            # finally use all samples for the interpolator
            # k = np.arange(N_mcs) #loo 
            # k = np.random.randint(0, N_mcs, int(N_mcs//10)) # 10 % num sample runs of loo 
            k = np.random.randint(0, N_mcs_imp, min(N_mcs_imp,100)) # 100 sample runs of loo 
            val_errs = []
            ind = np.ones(N_mcs_imp, dtype=bool)
            if N_mcs_imp<10*len(k): # do loo
                for k_i in k:
                    ind[k_i] = False
                    interp_loo =  interp_fun(scale(x_samp[ind,:]),  this_out[ind], **kwargs)
                    err = interp_loo(scale(x_samp[~ind,:])) - this_out[~ind]
                    val_errs.append(err)
                    ind[k_i] = True
                
                intp_err = np.sqrt(np.nanmean(np.power(val_errs, 2))) / out_range
                if intp_err*100 > intp_err_warn:
                    logger.warn(f'RMSE of interpolator using {len(k)} runs of leave-one-out cross-validation: {intp_err*100:1.3f} percent')
            else: # do lko
                ind[k]=False
                interp_lko =  interp_fun(scale(x_samp[ind,:]),  this_out[ind], **kwargs)
                val_errs = interp_lko(scale(x_samp[~ind,:])) - this_out[~ind]
                
                intp_err = np.sqrt(np.nanmean(np.power(val_errs, 2))) / out_range
                if intp_err*100 > intp_err_warn:
                    logger.warn(f'RMSE of interpolator using leave-{len(k)}-out cross-validation: {intp_err*100:1.3f} percent')
            intp_errors[n_ale] = intp_err
            
            logger.debug(f'Took {time.time()-now:1.2f} s to cross-validate interpolator.')
            
            now=time.time()
            numeric_focals = [var.numeric_focals for var in vars_imp] # no-imp => []
            # palette = sns.color_palette(n_colors=n_imp_hyc)
            for i_hyc, hypercube in enumerate(imp_hyc_foc_inds):
                # get focal sets / intervals
                focals = np.vstack([focals[ind,:] for focals, ind in zip(numeric_focals, hypercube)])
                vars_opt = focals[:,0]!=focals[:,1]# focals are not a singleton
                n_vars_opt = np.sum(vars_opt)
                
                if plot_intp:
                    engine = scipy.stats.qmc.Halton(n_vars_imp)
                    inputs_hyc = engine.random(3**n_vars_imp)
                    for i_var in range(n_vars_imp):
                        minvar, maxvar = focals[i_var,:]
                        inputs_hyc[:, i_var] *= maxvar - minvar
                        inputs_hyc[:, i_var] += minvar
                        
                    outputs_hyc = interp(scale(inputs_hyc))
                    # maxout_hyc = np.max(outputs_hyc)
                    # minout_hyc = np.min(outputs_hyc)
                    df_hyc = pd.DataFrame(inputs_hyc, columns = [var.name for var in vars_imp])
                    df_hyc[self.out_name] = outputs_hyc
                    plot_hyc_grid(df_hyc, grid, self.out_name, 
                                  #maxx, minx, 
                                  #maxout_hyc, minout_hyc
                                  )
                
                # transform boundaries to unit cube for penalization
                # will slightly exceed the unit cube due to samples being strictly inside the hypercube which was derived from all boundaries
                # but the transformation is done to the minimum and maximum sample
                unit_bounds = np.empty_like(focals)
                for lu in range(2):
                    unit_bounds[:,lu] = scale(focals[:,lu])
                    
                hyc_vol_frac = np.product(unit_bounds[:,1] - unit_bounds[:,0])

                # initialize memory for optimizing x in wrapper interval_range
                temp_x = np.copy(np.mean(unit_bounds, axis=1))[np.newaxis,:]
                temp_dists = np.empty((1,n_vars_imp))
                # interpolator needs input of shape (S, n_vars_imp)
                # optimizer needs input of shape (n_vars_opt,) (minimize) or (S,2 * n_vars_opt) (global)
                # init is shape (n_vars_imp,1) and so is temp_x 
                # focals is shape (n_vars_imp, 2) and so is focals and so is unit_bounds
                
                if opt_meth=='Nelder-Mead':
                    options = kwargs.get('options',{})
                    options['adaptive'] = True
                else:
                    options = kwargs.get('options',{})
                
                if isinstance(opt_meth, str) and opt_meth!='genetic':
                    
                    resl = scipy.optimize.minimize(interval_range, 
                                np.concatenate([x_out_min[vars_opt], x_out_max[vars_opt]]),  
                                method=opt_meth, 
                                args=(interp,temp_x,unit_bounds,temp_dists, vars_opt, n_vars_opt),
                                bounds=np.vstack([unit_bounds[vars_opt], unit_bounds[vars_opt]]), # (2 * n_opt_vars, 2)
                                options=options)
                    unit_x_low = resl.x[:n_vars_opt]
                    unit_x_up = resl.x[n_vars_opt:]

                    if not resl.success:
                        logger.warning(f'interval optimization failed on hypercube {i_hyc} of sample {n_ale} with message: {resl.message}')
                    
                elif opt_meth=='genetic':
                    popsize = 15
                    S = popsize * 2 * n_vars_opt
                    temp_x_vec = np.repeat(temp_x, S, axis=0) # (S, 2 * n_vars)
                    temp_dists_vec = np.repeat(temp_dists, S, axis=0) # (S, 2 * n_vars)
                    init = scipy.stats.qmc.Halton(2 * n_vars_opt).random(S) # (S, 2 * n_vars_opt)

                    resl = scipy.optimize.differential_evolution(interval_range, 
                                np.vstack([unit_bounds[vars_opt,:], unit_bounds[vars_opt,:]]), # (2 * n_vars_opt, 2)
                                args=(interp,temp_x_vec,unit_bounds,temp_dists_vec, vars_opt, n_vars_opt),
                                polish=False,
                                init=init,
                                vectorized=True, 
                                updating='deferred')
                    
                    unit_x_low = resl.x[:n_vars_opt]
                    unit_x_up = resl.x[n_vars_opt:]
                else:
                    logger.warning('Attempting Brute Force Interval optimization.')
                    unit_x_intv = scipy.optimize.brute(interval_range, 
                                np.vstack([unit_bounds[vars_opt,:], unit_bounds[vars_opt,:]]),
                                args=(interp,temp_x,unit_bounds,temp_dists, vars_opt, n_vars_opt),Ns=10)
                    unit_x_low = unit_x_intv[:n_vars_opt]
                    unit_x_up = unit_x_intv[n_vars_opt:]
                
                temp_x[:,vars_opt] = unit_x_low
                out_of_bounds = np.logical_or((temp_x - unit_bounds[:,0]) <-1e-8,
                                              (temp_x - unit_bounds[:,1]) > 1e-8)
                if np.any(out_of_bounds):
                    logger.warning(f'Minimum out of bounds at sample {n_ale} hypercube {i_hyc} for variable nr {np.where(out_of_bounds)[1]}')
                    # print(temp_x, unit_bounds, temp_x.T - unit_bounds,vars_opt)
                
                out_low = interp(temp_x)
                if out_low < out_min:
                    err = np.abs(out_min - out_low)[0]
                    intp_undershot[0] += 1
                    intp_undershot[1] += err
                    out_low = out_min
                    if err / out_range * 100 > extrp_warn:
                        logger.warning(f'Extrapolation by {err/out_range*100:1.3f} percent (interpolation domain min) at sample {n_ale} hypercube {i_hyc}.')
                    
                if isinstance(interp, scipy.interpolate.NearestNDInterpolator):
                    i, _ = interp.neighbor_dist(temp_x)
                    x_low = x_samp[i,:]
                else:
                    x_low = unscale(temp_x)
                
                temp_x[:,vars_opt] = unit_x_up
                out_of_bounds = np.logical_or((temp_x - unit_bounds[:,0]) <-1e-8,
                                              (temp_x - unit_bounds[:,1]) > 1e-8)
                if np.any(out_of_bounds):
                    logger.warning(f'Maximum out of bounds at sample {n_ale} hypercube {i_hyc} for variable nr {np.where(out_of_bounds)[1]}')
                    # print(temp_x, unit_bounds)
                    
                out_up = interp(temp_x)
                if out_up > out_max:
                    err = np.abs((out_up - out_max))[0]
                    intp_exceed[0] += 1
                    intp_exceed[1] += err
                    out_up = out_max
                    if err / out_range * 100 > extrp_warn:
                        logger.warning(f'Extrapolation by {err/out_range*100:1.3f} percent (interpolation domain max) at sample {n_ale} hypercube {i_hyc}.')
                    
                if isinstance(interp, scipy.interpolate.NearestNDInterpolator):
                    i, _ = interp.neighbor_dist(temp_x)
                    x_up = x_samp[i,:]
                else:
                    x_up = unscale(temp_x)
                    
                if plot_res:
                    plot_opt_res_grid(grid, vars_opt, 
                                      x_low, x_up, 
                                      focals, 
                                      self.out_valid[1], self.out_valid[0], 
                                      out_low, out_up)
                    
                imp_foc[n_ale, i_hyc, :] = out_low, out_up
                
                # assemble validation sample points (for linear or rbf interpolators)
                if not isinstance(interp, scipy.interpolate.NearestNDInterpolator):
                    # val_samp_prim axis 3 is ordered in as all_vars_prim 
                    # [var for var in vars_ale if var.primary] + list(vars_imp)
                    for i,var in enumerate([var for var in vars_ale if var.primary]):
                        assert var.value is not None 
                        val_samp_prim[n_ale, i_hyc, i, :] = var.value
                    i += 1
                    for j, b in enumerate(vars_opt):
                        # if b:
                        val_samp_prim[n_ale, i_hyc, j + i, :] = x_low[0,j], x_up[0,j]
                        # else:
                        #     val_samp_prim[n_ale, i_hyc, j + i, 0] = focals[j,0]
                
                if print_res_ranges:
                    bl=focals[vars_opt,0]
                    bh=focals[vars_opt,1]
                    opt_var=0
                    for focal, var, b in zip(focals, vars_imp, vars_opt):
                        if not b:
                            continue
                        pl=int((unit_x_low[opt_var]-bl[opt_var])/(bh[opt_var]-bl[opt_var])*100)
                        ph=int((unit_x_up[opt_var]-bl[opt_var])/(bh[opt_var]-bl[opt_var])*100)
                        s=f'{var.name[:6]}\t{bl[opt_var]:1.2g}\t'
                        for i in range(100):
                            if i==pl and i==ph: s+='+'
                            elif i==pl: s+='<'
                            elif i==ph: s+='>'
                            elif i==0: s+='|'
                            elif i==99: s+='|'
                            else: s+='.'
                        s+=f'{bh[opt_var]:1.2g}'
                        logger.info(s)
                        opt_var+=1
                        
                    pl=int((out_low-out_min)/(out_max-out_min)*100)
                    ph=int((out_up-out_min)/(out_max-out_min)*100)
                    s=f'out \t{out_min:1.2g}\t'
                    for i in range(100):
                        if i==pl and i==ph: s+='+'
                        elif i==pl: s+='<'
                        elif i==ph: s+='>'
                        elif i==0: s+='|'
                        elif i==99: s+='|'
                        else: s+='.'
                    s+=f'{out_max:1.2g}'
                    logger.info(s)
                
                # arg_vars = kwargs.get('arg_vars', None)
                # mapping_function = kwargs.get('mapping_function', None)
                # if arg_vars is not None and mapping_function is not None:
                #
                #     opt_var=0
                #     for focal, var, b in zip(focals, vars_imp, vars_opt):
                #         for arg,var_ in arg_vars.items():
                #             if var_ == var:
                #                 if b:
                #                     arg_vals_l[arg] = unit_x_low[opt_var]
                #                     arg_vals_h[arg] = unit_x_up[opt_var]
                #                     opt_var+=1
                #                 else:
                #                     arg_vals_l[arg] = focal[0]
                #                     arg_vals_h[arg] = focal[1]
                #                 break
                #         else:
                #             print(f'could not find {var} in {arg_vars}')
                #
                #     fmin_true = mapping_function(**arg_vals_l, working_dir='/dev/shm/womo1998',result_dir='/dev/shm/womo1998', skip_existing=False)
                #     fmax_true = mapping_function(**arg_vals_h, working_dir='/dev/shm/womo1998',result_dir='/dev/shm/womo1998', skip_existing=False)
                #     logger.debug(f'Approximated: /t {out_low:1.3f}...{out_up:1.3f}')
                #     logger.debug(f'True: /t {fmin_true:1.3f}...{fmax_true:1.3f}')
                #
                #     imp_foc[n_ale, i_hyc, :] = fmin_true, fmax_true
                
                next(pbar)
                if plot_res:
                    from IPython import display
                    display.display(plt.gcf())
                    display.clear_output(wait=True)
                    time.sleep(1)
                    input("Hit any key to continue...")
                    
            logger.debug(f'Took {time.time()-now:1.2f} s for interval optimization of all {n_imp_hyc} hypercubes.')
            logger.debug(imp_foc[n_ale, :, :])
        
        self.imp_foc = imp_foc
        self.val_samp_prim = val_samp_prim
        
        self.intp_errors = intp_errors
        self.intp_exceed = intp_exceed
        self.intp_undershot = intp_undershot
        
        if intp_exceed[0]:
            logger.warning(f'The interpolator has exceeded the maximum interpolation domain a total of {intp_exceed[0]} out of {n_imp_hyc * len(iter_ale)} times: Average error {intp_exceed[1]/intp_exceed[0]/(self.out_valid[1]-self.out_valid[0])*100:1.3f} percent (of valid domain)')
        if intp_undershot[0]:
            logger.warning(f'The interpolator has exceeded the minimum interpolation domain a total of {intp_undershot[0]} out of {n_imp_hyc * len(iter_ale)} times: Average error {intp_undershot[1]/intp_undershot[0]/(self.out_valid[1]-self.out_valid[0])*100:1.3f} percent (of valid domain)')
        
        return imp_foc, val_samp_prim, intp_errors, intp_exceed, intp_undershot
    
    def optimize_inc(self, stat_fun, n_stat, stat_fun_kwargs={}):
        
        '''
        stat_fun must accept the following arguments:
                samples, p_weights, i_stat (=the index of the statistic for multivalued statistics, e.g the number of a histogram bin)
        stat_fun may additionally accept stat_fun_kwargs
        
        '''
        
        def stat_eval(x, stat_fun, imp_foc, i_imp_hyc, sort_ind, i_stat, vars_opt, min_max, stat_fun_kwargs):
            
            for i, var in enumerate(vars_opt):
                var.freeze(x[i])
            p_weights = self.probabilities_imp(i_imp_hyc)
            # samp = imp_foc[sort_ind]

            stat_vals = np.empty(2)
            for bound in range(2):
                stat_vals[bound] = stat_fun(imp_foc[sort_ind[:,bound],i_imp_hyc, bound], p_weights[sort_ind[:,bound]], i_stat,  min_max, **stat_fun_kwargs) # min
            # # high boundary
            # stat_vals[1] = stat_fun(samp[:, 1], p_weights[sort_ind[:,1]], i_stat,  min_max, **stat_fun_kwargs) # min
            
            return stat_vals
        
        def interval_range(x, stat_fun, imp_foc, i_imp_hyc, sort_ind, i_stat, vars_opt, n_vars_opt, stat_fun_kwargs, ):
                    #        x, interp, temp_x, unit_bounds, temp_dists,
                    # **kwargs):
            '''
            optimizer might need scaling to unit boundaries... let's see 
            distribution parameters are usually in somewhat similar ranges 
            
            optimization has to be done for each statistic, boundary, imprecise and incomplete hypercube separately
            i.e. a scalar value must be returned from wrapper
            
            this creates a huge overhead: 
                p_weights is computed for all imp_hyc at once -> resolved
                stat_fun returns all statistics at once
                all but one entry of these are discarded -> resolved
            
            nevertheless it might be needed for verification purposes
            '''
            
            
            stat_vals = stat_eval(x[:n_vars_opt], stat_fun, imp_foc, i_imp_hyc, sort_ind, i_stat, vars_opt, 1, stat_fun_kwargs)
            if np.all(np.isnan(stat_vals)): 
                return 0
            stat_min = np.nanmin(stat_vals)
            
            stat_vals = stat_eval(x[n_vars_opt:], stat_fun, imp_foc, i_imp_hyc, sort_ind, i_stat, vars_opt, -1, stat_fun_kwargs)
            if np.all(np.isnan(stat_vals)): 
                return 0
            stat_max = np.nanmax(stat_vals)
            
            return stat_min - stat_max
        
        logger.info('Estimating incompleteness intervals by direct L-BFGS optimization of statistics over input hypercubes...')
        # Samples
        imp_foc = self.imp_foc
        
        # Variables
        vars_inc = self.vars_inc
        n_vars_inc = len(vars_inc)
        
        # Epistemic Hypercubes
        imp_hyc_foc_inds = self.imp_hyc_foc_inds # no-imp: is a list containing a single empty tuple
        n_imp_hyc = len(imp_hyc_foc_inds) # no-imp: would be 1
        imp_hyc_mass = self.imp_hyc_mass # no-imp: is a list containing a single 1.0
        
        inc_hyc_foc_inds = self.inc_hyc_foc_inds
        n_inc_hyc = len(inc_hyc_foc_inds)
        inc_hyc_mass = self.inc_hyc_mass 
        # no-inc: sizes are analogous to the no-imp case
        
        n_hyc = n_imp_hyc * n_inc_hyc
             
        numeric_focals = [var.numeric_focals for var in vars_inc]
        
        # compute belief functions for each statistic
        focals_stats = np.full((n_stat, n_hyc, 2), np.nan)
        hyc_mass = np.empty((n_hyc,))
        
        if vars_inc:
            pbar = simplePbar(n_imp_hyc * n_inc_hyc*n_stat)
            for i_imp_hyc in range(n_imp_hyc):
                for i_inc_hyc, hypercube in enumerate(inc_hyc_foc_inds):
                    i_hyc = i_imp_hyc * n_inc_hyc + i_inc_hyc
                    
                    focals = np.vstack([focals[ind,:] for focals, ind in zip(numeric_focals, hypercube)])
                    # vars_opt = focals[:,0]!=focals[:,1]# focals are not a singleton
                    # n_vars_opt = np.sum(vars_opt)
                    
                    bounds = focals
                    # bounds = [focs[ind] for focs, ind in zip(numeric_focals, inc_hyc_foc_inds[i_inc_hyc])]
                    init = np.mean(bounds, axis=1, keepdims=True)
                    
                    sort_ind = np.argsort(imp_foc[:,i_imp_hyc,:], axis=0)
                    
                    intervals = np.full((n_stat,2),np.nan)
                    # plt.figure()            
                    # for i, var in enumerate(vars_inc):
                    #     var.freeze(init[i,0])
                    # p_weights = self.probabilities_imp(i_imp_hyc)
                    # print(np.sum(p_weights))
                    # plt.hist(imp_foc[:,i_imp_hyc,0], weights=p_weights, alpha=0.5, bins=50, density=False)
                    # plt.hist(imp_foc[:,i_imp_hyc,1], weights=p_weights, alpha=0.5, bins=50, density=False)
                    # plt.show(block=True)
                    
                    now = time.time()
                    for i_stat in range(n_stat):
                        
                        out_low = stat_eval(init[:,0], stat_fun, imp_foc, i_imp_hyc, sort_ind, i_stat, vars_inc, 1, stat_fun_kwargs)
                        out_up  = stat_eval(init[:,0],  stat_fun, imp_foc, i_imp_hyc, sort_ind, i_stat, vars_inc, -1, stat_fun_kwargs)
                        
                        intervals[i_stat,0] = np.nanmin(out_low)
                        intervals[i_stat,1] = np.nanmax(out_up)
                        if np.isnan(intervals[i_stat,:]).all():
                            break
                        continue
                    
                        logging.disable(logging.INFO)
                        resl = scipy.optimize.minimize(fun=interval_range, x0=np.vstack((init,init)),
                                                args=(stat_fun, imp_foc, i_imp_hyc, sort_ind, i_stat, vars_inc, n_vars_inc, stat_fun_kwargs),
                                                bounds=np.vstack((bounds, bounds)))
                        if not resl.success:
                            logger.warning(f'Optimizer failed for hypercube {i_hyc} at i_stat {i_stat}. Breaking here.')
                            break
                        x_low = resl.x[:n_vars_inc]
                        x_up = resl.x[n_vars_inc:]
                        
                        out_low = stat_eval(x_low, stat_fun, imp_foc, i_imp_hyc, sort_ind, i_stat, vars_inc, 1, stat_fun_kwargs)
                        out_up  = stat_eval(x_up,  stat_fun, imp_foc, i_imp_hyc, sort_ind, i_stat, vars_inc, -1, stat_fun_kwargs)
                        
                        logging.disable(logging.NOTSET)
                        focals_stats[i_stat, i_hyc, 0] = np.nanmin(out_low)
                        focals_stats[i_stat, i_hyc, 1] = np.nanmax(out_up)
                        
                        # try:
                        #     logging.disable(logging.INFO)
                        #     resll = scipy.optimize.minimize(wrapper, init, ( 1, stat_fun, imp_foc[:,i_imp_hyc, 0], i_imp_hyc, i_stat, stat_fun_kwargs), bounds=bounds)
                        #     resul = scipy.optimize.minimize(wrapper, init, (-1, stat_fun, imp_foc[:,i_imp_hyc, 0], i_imp_hyc, i_stat, stat_fun_kwargs), bounds=bounds)
                        #     # high boundary
                        #     reslu = scipy.optimize.minimize(wrapper, init, ( 1, stat_fun, imp_foc[:,i_imp_hyc, 1], i_imp_hyc, i_stat, stat_fun_kwargs), bounds=bounds)
                        #     resuu = scipy.optimize.minimize(wrapper, init, (-1, stat_fun, imp_foc[:,i_imp_hyc, 1], i_imp_hyc, i_stat, stat_fun_kwargs), bounds=bounds)
                        #
                        #     for res in [resll, resul, reslu, resuu]:
                        #         if not res.success:
                        #             logger.warning(f'Interval optimization did not succeed on hypercube {i_hyc} with message: {res.message}')
                        # finally:
                        #     logging.disable(logging.NOTSET)
                        #     focals_stats[i_stat, i_hyc, 0] = min( resll.fun,  reslu.fun)
                        #     focals_stats[i_stat, i_hyc, 1] = max(-resul.fun, -resuu.fun)
                        
                        next(pbar)
                    
                    # plt.figure()
                    # cm = 0
                    # for j in range(n_stat):
                    #     r,l = intervals[j,:]
                    #     target_pdfs = stat_fun_kwargs['target_densities']
                    #     m = target_pdfs[1]-target_pdfs[0]
                    #     plt.bar(l, m, (r-l), bottom=cm + 0.05*m, align='edge', color='lightgrey', edgecolor='black', alpha=0.5)
                    #     cm += m
                    # plt.show(block=True)
                    break
                    
                    logger.debug(f'Took {time.time()-now:1.2f} s for interval optimization of {n_stat} statistics on hypercube {i_hyc}.')
                break        
                hyc_mass[i_imp_hyc * n_inc_hyc: (i_imp_hyc + 1 ) * n_inc_hyc ] = inc_hyc_mass * imp_hyc_mass[i_imp_hyc] 
        else: #no incompleteness
            raise NotImplementedError('Needs implementation, copy relevant code from above')
            # with HiddenPrints():
            #     p_weights = self.probabilities_imp() 
            # for i_imp_hyc in range(n_imp_hyc):
            #     for high_low in range(2):
            #         stat = stat_fun(imp_foc[:, i_imp_hyc, high_low], p_weights[:, i_imp_hyc], None, **stat_fun_kwargs)
            #         focals_stats[:, i_imp_hyc, high_low] = stat
            # hyc_mass = imp_hyc_mass
        
        self.focals_stats = focals_stats
        self.focals_mass = hyc_mass
        
        return focals_stats, hyc_mass
    
    def check_sample_sizes(self, vars_epi, samples, N_mcs_ale=None, N_mcs_epi=None):
        '''
        A function to check sample sizes for approximation of epistemic hypercubes
        '''
        vars_ale = self.vars_ale
        if N_mcs_ale is None:
            N_mcs_ale = samples.shape[0]
        if N_mcs_epi is None:
            N_mcs_epi = samples.shape[0]
        
        
        logger.info("Checking sample sizes for approximation of epistemic focal sets...")
        for var in vars_epi:
            if var.is_poly:
                sample_sizes=np.empty((N_mcs_ale, var.n_focals), dtype=int)
                for n_ale in range(N_mcs_ale):
                    for var_ale in vars_ale:
                        var_ale.freeze(samples[var_ale.name].iloc[n_ale])
                    for i_foc, (boundl, boundr) in enumerate(var.numeric_focals):
                        this_selector = np.logical_and(samples[var.name].iloc[:N_mcs_epi]>=boundl,
                                                       samples[var.name].iloc[:N_mcs_epi]<=boundr)
                        sample_sizes[n_ale, i_foc] = np.sum(this_selector)
            else:
                sample_sizes=np.empty((1, var.n_focals), dtype=int)
                for i_foc, (boundl, boundr) in enumerate(var.numeric_focals):
                    this_selector = np.logical_and(samples[var.name].iloc[:N_mcs_epi]>=boundl,
                                                   samples[var.name].iloc[:N_mcs_epi]<=boundr)
                    sample_sizes[0, i_foc] = np.sum(this_selector)
                    # print(f'Focal set {i_foc} of variable {var.name} contains {sample_sizes[0, i_foc]} epistemic samples.')
            for i_foc in range(var.n_focals):
                max_samples = np.max(sample_sizes[:,i_foc])
                if var.is_poly and max_samples:
                    bins = [10 ** i for i in range(-1, int(np.ceil(np.log10(max_samples))) + 1)]
                    bins[0]=0
                    #bins[1]=1
                    bins[-1] = max_samples
                    hist, bins = np.histogram(sample_sizes[:,i_foc],bins=bins)
                else:
                    hist, bins = [1,], [sample_sizes[0, i_foc], sample_sizes[0, i_foc]]
                    
                logger.info(f'Focal set {i_foc} of variable {var} epistemic sample size distribution:')
                for lbin, ubin, count in zip(bins[:-1], bins[1:], hist):
                    if count == 0: continue
                    logger.info(f'{lbin} < n < {ubin}: {count} aleatory samples')
                
    @property
    def hyc_hyp_vars(self, ):
        # each imprecise hypercube may be constructed from different 
        # aleatory hypervariables (RVs) -> has a different product probability
        
        # hyc_hyp_var indices refer to imprecise hypercubes
        if self._hyc_hyp_vars is not None: # cache
            return self._hyc_hyp_vars
        
        vars_imp = self.vars_imp
        hyc_foc_inds = self.imp_hyc_foc_inds
        n_hyc = len(hyc_foc_inds)
        
        #assemble hypervariables per hypercube (secondary)
        hyc_hyp_vars = [set() for i_hyc in range(n_hyc)] # use sets to avoid double counting of variables that may be hypervariables of more than one focal set
        for i_hyc in range(n_hyc):
            # walk over primary epistemic variables and their hypervariables  
            for i_epi, var_epi in enumerate(vars_imp):
                # skip if hypercube is defined by fixed values -> stochastic probabilities are not changing
                if not var_epi.is_poly: continue
                
                hyp_vars = var_epi._focals[hyc_foc_inds[i_hyc][i_epi]]
                hyc_hyp_vars[i_hyc].update(hyp_vars)      
                
        self._hyc_hyp_vars = hyc_hyp_vars
        return hyc_hyp_vars

    def hyc_foc_inds(self, vars_):
        
        if self.dim_ex=='cartesian':
            # cartesian product of input intervals / focal sets
            # for each hypercube: indices of focal sets of each variable
            return list(product(*[range(var.n_focals) for var in vars_]))
        elif self.dim_ex=='hadamard':
            # for fuzzy sets, an alpha level would be chosen, which results in a single focal set per variable and thus a single hypercube
            # each variable must have the same number of focal sets
            n_vars = len(vars_)
            if not n_vars:
                return [()]
            else:
                n_focals = vars_[0].n_focals
                return [[i_foc, ] * n_vars for i_foc in range(n_focals)]
        elif self.dim_ex=='vacuous':
            raise NotImplementedError('Vacuous extension currently not implemented')
        else:
            raise ValueError(f'Dimension extension {self.dim_ex} not understood.')
        
    @property
    def imp_hyc_foc_inds(self,):
        vars_imp = self.vars_imp
        return self.hyc_foc_inds(vars_imp)
    
    @property
    def inc_hyc_foc_inds(self):
        vars_inc = self.vars_inc
        return self.hyc_foc_inds(vars_inc)
        
    
    def hyc_mass(self, vars_):
        hyc_foc_inds = self.hyc_foc_inds(vars_)
        n_hyc = len(hyc_foc_inds)
        
        hyc_mass = np.empty((n_hyc,))
        for i_hyc, foc_inds in enumerate(hyc_foc_inds):
            # compute output prob_dens for this hypercube
            if self.dim_ex=='cartesian':
                #returns 1.0 for empty hypercube
                hyc_mass[i_hyc] = np.product([var.masses[ind] for var, ind in zip(vars_, foc_inds)])
            elif self.dim_ex=='hadamard':
                # returns nan for empty hypercube
                hyc_mass[i_hyc] = np.average([var.masses[ind] for var, ind in zip(vars_, foc_inds)])
            else:
                raise NotImplementedError(f'{self.dim_ex} dimension extension currently not implemented')
            
        return hyc_mass
    
    @property
    def imp_hyc_mass(self):
        vars_imp = self.vars_imp
        return self.hyc_mass(vars_imp)
    
    @property
    def inc_hyc_mass(self):
        vars_inc = self.vars_inc
        return self.hyc_mass(vars_inc)
    
    def hypercube_sample_indices(self, inp_samples, vars_epi, hyc_foc_inds=None, N_mcs_epi=None):
        '''
        generate the input sample indices for all hypercubes defined by
        the focal sets of the provided epistemic variables
        
        Parameters:
        vars_epi: epistemic variables to consider for the hypercubes
        inp_samples: pd.DataFrame n_vars x N_mcs_*
        hyc_foc_inds: list-of-list-of-indices for each hypercube the indices of the corresponding focal sets of each epistemic variable
        N_mcs_epi: number of epistemic input samples, inferred from inp_samples if not given 
        '''
        
        if hyc_foc_inds is None:
            hyc_foc_inds = self.hyc_foc_inds(vars_epi) # no-imp -> [()]
            
        n_hyc = len(hyc_foc_inds) # no-imp: 1
        
        if N_mcs_epi is None:
            N_mcs_epi = inp_samples.shape[0]
            
        # extract underlying numpy array in order of vars_epi for faster indexing 
        inp_samples = inp_samples[[var.name for var in vars_epi]].values[:N_mcs_epi, :]
        
        hyc_dat_inds = np.ones((n_hyc, N_mcs_epi), dtype=bool)
        numeric_focals = [var.numeric_focals for var in vars_epi] # no-imp []
        for i_hyc, hypercube in enumerate(hyc_foc_inds):
            # get focal sets / intervals
            focals = [focals[ind] for focals, ind in zip(numeric_focals, hypercube)] # no-imp []
            # select matching epistemic inp_samples
            selector = hyc_dat_inds[i_hyc,:]
            # for (boundl,boundr), var in zip(focals,vars_epi):
            #     this_selector = np.logical_and(inp_samples[var.name]>=boundl,
            #                                    inp_samples[var.name]<=boundr)
            for i_var, (boundl, boundr) in enumerate(focals):
                selector &= inp_samples[:, i_var]>=boundl
                selector &= inp_samples[:, i_var]<=boundr 
                # no-imp: [1,1,...,1,1] select all samples
                
        return hyc_dat_inds
    

    def naive_uq(self, N_mcs, mapping, arg_vars):
        
        def imp_stat(x):
            for i, var in enumerate(vars_inc):
                var.freeze(x[i])
            # print(x)
            hypercube = imp_hyc_foc_inds[i_imp_hyc]
            
            imp_foc = np.empty((N_mcs, 2))
            
            rvs_ale = np.empty((len(vars_ale), N_mcs))
            for i, var_ale in enumerate(vars_ale):
                rvs_ale[i, :] = var_ale.rvs(size=N_mcs)
            
            bound_vars = []
            for i, (arg, var) in enumerate(arg_vars.items()):
                for var_ale in vars_ale:
                    if not var_ale.primary: continue
                    if var_ale.name == var:
                        bound_vars.append(var_ale)
                        break
                else:
                    for j, var_imp in enumerate(vars_imp):
                        if not var_imp.primary: continue
                        if var_imp.name == var:
                            bound_vars.append((var_imp, j))
                            break
                    else:
                        raise RuntimeError(f'Could not find var {var} for argument {arg}, neither in {vars_imp} nor {vars_ale}')
            
            for n_ale in range(N_mcs):
                for i in range(len(vars_ale)):
                    vars_ale[i].freeze(rvs_ale[i,n_ale])
                    
                bounds = np.zeros((len(arg_vars), 2))
                
                for i in range(len(arg_vars)):
                    if isinstance(bound_vars[i], RandomVariable):
                        bounds[i, :] = bound_vars[i].value
                    else:
                        var_imp, j = bound_vars[i]
                        bounds[i, :] = var_imp.numeric_focal(hypercube[j])
                
                # init = [np.mean(bound) for bound in bounds]
                # resl = scipy.optimize.minimize(lambda x:  mapping(*x), init, bounds=bounds)
                # resu = scipy.optimize.minimize(lambda x: -mapping(*x), init, bounds=bounds)
                #
                # imp_foc[n_ale,:] = (resl.fun, -resu.fun)
                
                imp_foc[n_ale, :] = mapping(bounds[0, :], bounds[1, :])
            
            Pf = np.sum(imp_foc >= 260, axis=0) / N_mcs
            # print(Pf)
            return Pf
        # Variables
        vars_inc = self.vars_inc
        vars_ale = self.vars_ale
        vars_imp = self.vars_imp
        
        
        # Epistemic Hypercubes
        imp_hyc_foc_inds = self.imp_hyc_foc_inds # no-imp: is a list containing a single empty tuple
        n_imp_hyc = len(imp_hyc_foc_inds) # no-imp: would be 1
        imp_hyc_mass = self.imp_hyc_mass # no-imp: is a list containing a single 1.0
        
        inc_hyc_foc_inds = self.inc_hyc_foc_inds
        n_inc_hyc = len(inc_hyc_foc_inds)
        inc_hyc_mass = self.inc_hyc_mass 
        # no-inc: sizes are analogous to the no-imp case
        
        n_hyc = n_imp_hyc * n_inc_hyc
        
        n_stat = 1
        
        num_focals_inc = [var.numeric_focals for var in vars_inc]
                
        # compute belief functions for each statistic
        focals_stats = np.empty((n_stat, n_hyc, 2))
        hyc_mass = np.empty((n_hyc,))
        
        if vars_inc:
            # print(vars_inc)
            pbar = simplePbar(n_imp_hyc * n_inc_hyc * n_stat)
            for i_imp_hyc in range(n_imp_hyc):
                for i_inc_hyc in range(n_inc_hyc):
                    i_hyc = i_imp_hyc * n_inc_hyc + i_inc_hyc
                    
                    bounds = [focs[ind] for focs, ind in zip(num_focals_inc, inc_hyc_foc_inds[i_inc_hyc])]

                    init = [np.mean(bound) for bound in bounds]
                    for i_stat in range(n_stat):
                        
                        if np.all(np.diff(bounds, axis=1)==0): # complete variable
                            focals_stats[i_stat, i_hyc,:] = imp_stat(init)
                        else:
                            
                            '''
                            optimizing a  stochastic objective requires carefully chosen initial guesses, as it is very noisy due to randomness
                            '''
                            initial_simplex = np.empty((5, 4))
                            for i in range(2):
                                for j in range(2):
                                    initial_simplex[i*2 + j,:]=[bounds[0][i],bounds[1][j], bounds[2][j],bounds[3][i]]
                            initial_simplex[-1,:] = init
                            # lower boundary
                            # print(f'Minimizing within {bounds}')
                            resll = scipy.optimize.minimize(lambda x:  imp_stat(x)[0], init, options={'initial_simplex':initial_simplex}, method='Nelder-Mead', bounds=bounds)
                            # resul = scipy.optimize.minimize(lambda x: -imp_stat(x)[0], init, bounds=bounds)
                            # high boundary
                            # reslu = scipy.optimize.minimize(lambda x:  imp_stat(x)[1], init, bounds=bounds)
                            # print(f'Maximizing within {bounds}')
                            resuu = scipy.optimize.minimize(lambda x: -imp_stat(x)[1], init, options={'initial_simplex':initial_simplex}, method='Nelder-Mead', bounds=bounds)
                            
                            # focals_stats[i_stat, i_hyc, 0] = min( resll.fun,  reslu.fun)
                            # focals_stats[i_stat, i_hyc, 1] = max(-resul.fun, -resuu.fun)
                            focals_stats[i_stat, i_hyc, :] = (resll.fun, resuu.fun)
                
                        next(pbar)
                        
                hyc_mass[i_imp_hyc * n_inc_hyc: (i_imp_hyc + 1 ) * n_inc_hyc ] = inc_hyc_mass * imp_hyc_mass[i_imp_hyc] 
        else: #no incompleteness
            for i_imp_hyc in range(n_imp_hyc):
                stat = imp_stat(None)
                focals_stats[:, i_imp_hyc, :] = stat
            hyc_mass = imp_hyc_mass
        
        return focals_stats, hyc_mass
            
    def save_state(self, fname, differential=None):
        # differential: samp, prop, imp, inc

        dirname, _ = os.path.split(fname)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        logger.info(f'Saving state of PolyUQ to {fname}. Make sure to store your variable definitions script externally...')
        out_dict = {}
        if differential is None or differential=='samp':
            out_dict['self.dim_ex'] = self.dim_ex
            
            out_dict['self.N_mcs_ale'] = self.N_mcs_ale
            out_dict['self.N_mcs_epi'] = self.N_mcs_epi
            out_dict['self.percentiles'] = self.percentiles
            out_dict['self.seed'] = self.seed
            out_dict['self.var_supp'] = self.var_supp
            out_dict['self.inp_samp_prim'] = self.inp_samp_prim
            out_dict['self.inp_suppl_ale'] = self.inp_suppl_ale
            out_dict['self.inp_suppl_epi'] = self.inp_suppl_epi
            
            if self.var_supp is not None:
                out_dict['self.var_supp.columns'] = self.var_supp.columns
            else:
                out_dict['self.var_supp.columns'] = None
            if self.inp_samp_prim is not None:
                out_dict['self.inp_samp_prim.columns'] = self.inp_samp_prim.columns
            else:
                out_dict['self.inp_samp_prim.columns'] = None
            if self.inp_suppl_ale is not None:
                out_dict['self.inp_suppl_ale.columns'] = self.inp_suppl_ale.columns
            else:
                out_dict['self.inp_suppl_ale.columns'] = None
            if self.inp_suppl_epi is not None:
                out_dict['self.inp_suppl_epi.columns'] = self.inp_suppl_epi.columns
            else:
                out_dict['self.inp_suppl_epi.columns'] = None
        
        if differential is None or differential=='prop':
            out_dict['self.fcount'] = self.fcount
            out_dict['self.loop_ale'] = self.loop_ale
            out_dict['self.loop_epi'] = self.loop_epi
            out_dict['self.out_name'] = self.out_name
            out_dict['self.out_samp'] = self.out_samp
            out_dict['self.out_valid'] = self.out_valid
            
        if differential is None or differential=='imp':
            out_dict['self.imp_foc'] = self.imp_foc
            out_dict['self.val_samp_prim'] = self.val_samp_prim
            out_dict['self.intp_errors'] = self.intp_errors
            out_dict['self.intp_exceed'] = self.intp_exceed
            out_dict['self.intp_undershot'] = self.intp_undershot
        
        if differential is None or differential=='inc':
            out_dict['self.focals_stats'] = self.focals_stats
            out_dict['self.focals_mass'] = self.focals_mass
        
        with open(fname + '.tmp', 'wb') as f:
            np.savez_compressed(f, **out_dict)
        if os.path.exists(fname):
            os.remove(fname)
        os.rename(fname + '.tmp', fname)
        
    def load_state(self, fname, differential=None):
        
        def validate_array(arr):
            '''
            Determine whether the argument has a numeric datatype and if
            not convert the argument to a scalar object or a list.
        
            Booleans, unsigned integers, signed integers, floats and complex
            numbers are the kinds of numeric datatype.
        
            Parameters
            ----------
            array : array-like
                The array to check.
            
            '''
            _NUMERIC_KINDS = set('buifc')
            if arr is None:
                return arr
            if not arr.shape:
                return arr.item()
            elif arr.dtype.kind in _NUMERIC_KINDS:
                return arr
            else:
                return list(arr)
            
        def to_dataframe(arr, columns):
            if arr is None:
                return None
            else:
                return pd.DataFrame(data=arr, columns=columns)
        
        logger.info('Now loading previous results from  {}'.format(fname))
        
        in_dict = np.load(fname, allow_pickle=True)
        
        if differential is None or differential=='samp':
            self.dim_ex = validate_array(in_dict['self.dim_ex'])
            
            self.N_mcs_ale = validate_array(in_dict['self.N_mcs_ale'])
            self.N_mcs_epi = validate_array(in_dict['self.N_mcs_epi'])
            self.percentiles = validate_array(in_dict['self.percentiles'])
            self.seed = validate_array(in_dict['self.seed'])
            
            self.var_supp = to_dataframe(validate_array(in_dict.get('self.var_supp')),
                                              validate_array(in_dict.get('self.var_supp.columns')))
            self.inp_samp_prim = to_dataframe(validate_array(in_dict['self.inp_samp_prim']),
                                              validate_array(in_dict['self.inp_samp_prim.columns']))
            self.inp_suppl_ale = to_dataframe(validate_array(in_dict['self.inp_suppl_ale']),
                                              validate_array(in_dict['self.inp_suppl_ale.columns']))
            self.inp_suppl_epi = to_dataframe(validate_array(in_dict['self.inp_suppl_epi']),
                                              validate_array(in_dict['self.inp_suppl_epi.columns']))
        
        if differential is None or differential=='prop':
            self.fcount = validate_array(in_dict['self.fcount'])
            self.loop_ale = validate_array(in_dict['self.loop_ale'])
            self.loop_epi = validate_array(in_dict['self.loop_epi'])
            self.out_name = validate_array(in_dict.get('self.out_name'))
            self.out_samp = validate_array(in_dict['self.out_samp'])
            self.out_valid = validate_array(in_dict.get('self.out_valid'))
        
        if differential is None or differential=='imp':
            self.imp_foc = validate_array(in_dict['self.imp_foc'])
            self.val_samp_prim = validate_array(in_dict.get('self.val_samp_prim'))
            self.intp_errors = validate_array(in_dict.get('self.intp_errors'))
            self.intp_exceed = validate_array(in_dict.get('self.intp_exceed'))
            self.intp_undershot = validate_array(in_dict.get('self.intp_undershot'))
        
        if differential is None or differential=='inc':
            self.focals_stats = validate_array(in_dict['self.focals_stats'])
            self.focals_mass = validate_array(in_dict['self.focals_mass'])
        
        if self.var_supp is None:
            self.sample_qmc(percentiles= self.percentiles, supp_only=True)
            self.save_state(fname)
    
def generate_histogram_bins(data, axis=1, nbin_fact=1):
    # generate the bins
    # modified Freedman-Diaconis rule 
    # for each epistemic sample or hypercube/focal set boundary, compute the 25% and 75% quantiles
    # the axis parameter is not tested
    
    N_mcs = data.shape[int(not axis)]
    quantiles = np.apply_along_axis(np.nanquantile, axis, data, [0.75,0.25])
    # compute the maximum interquartile range over all epistemic samples
    iqr = np.max(quantiles[:,0]) - np.min(quantiles[:,1])
    # compute the bin width based on Freedman-Diaconis equation
    bin_width_dens = 2 * iqr * np.power(N_mcs, -1/3)
    # compute the bins
    start, stop = np.nanmin(data), np.nanmax(data)
    out_range = stop - start
    n_bins_dens = np.ceil(out_range / bin_width_dens * nbin_fact).astype(int)
    bins_densities = np.linspace(start, stop, n_bins_dens + 1) # that is what needs to be passed to histogram functions
    
    return bins_densities
        
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

def test_to_dm():

    
    
    logger2 = logging.getLogger('uncertainty.data_manager')
    logger2.setLevel(level=logging.WARNING)
    arg_vars = {'q1':'q1', 'q2':'q2'}
    
    def deterministic_mapping2(q1,q2, jid=None, result_dir='', working_dir=''):
        return np.array((189/500*q1+3*q2)*16/3),

    example_num, dim_ex, vars_ale, vars_epi = example_e()
    
    N_mcs_ale = 100
    N_mcs_epi = 100
    
    poly_uq = PolyUQ(vars_ale, vars_epi, dim_ex=dim_ex)
    poly_uq.sample_qmc(N_mcs_ale, N_mcs_epi, check_sample_sizes=False)
    dm_grid, dm_ale, dm_epi = poly_uq.to_data_manager('example',
                                                    working_dir='/dev/shm/womo1998/',
                                                    result_dir='/usr/scratch4/sima9999/work/modal_uq/poly-dm-test',
                                                    overwrite=True)
    dm_grid.evaluate_samples(deterministic_mapping2, arg_vars, {'stress': ()}, dry_run=True)
    ':type dm_grid: DataManager'
    dm_grid.clear_locks()
    poly_uq.from_data_manager(dm_grid, 'stress')
    intervals = poly_uq.estimate_imp(False)
    #%%snakeviz

    def stat_fun(a, weight, i_stat):
        exceed = a>=260
        return np.sum(weight[exceed])
    
    # def stat_fun(a, weight,):
    #     return [np.average(a, weights=weight),]
    
    
    n_stat = 1
    # focals_Pf, hyc_mass = poly_uq.estimate_inc(intervals, stat_fun, n_stat)
    focals_Pf, hyc_mass = poly_uq.optimize_inc(stat_fun, n_stat)
    
    if example_num==0: # Example 1
        print(f'Result: \t {focals_Pf[0,0,0]*1e4} - {focals_Pf[0,0,1]*1e4}')
        print('Reference:\t 10.03 - 10.03')
    elif example_num==1: # Example b
        print("Alpha-level 1 (Hypercube 0)")
        print(f'Result: \t {focals_Pf[0,0,0]*1e4} (- {focals_Pf[0,0,1]*1e4})')
        print('Reference:\t 10.27')
        
        print("Alpha-level 0 (Hypercube 3)")
        print(f'Result: \t {focals_Pf[0,1,0]*1e4} - {focals_Pf[0,1,1]*1e4}')
        print('Reference:\t 3.21 - 21.94')
    elif example_num==2: # Example c
        print("Alpha-level 1 (Hypercube 0)")
        print(f'Result: \t {focals_Pf[0,0,0]*1e4} (- {focals_Pf[0,0,1]*1e4})')
        print('Reference:\t 20.10')
        
        print("Alpha-level 0 (Hypercube 15)")
        print(f'Result: \t {focals_Pf[0,1,0]*1e4} - {focals_Pf[0,1,1]*1e4}')
        print('Reference:\t 2.33 - 120.16')
    else:
        n_hyc = len(hyc_mass)
        for i_hyc in range(n_hyc):
            print(f"(Hypercube {i_hyc})")
    #         print(f'Result: \t {focals_Pf[0,i_hyc,0]*1e4} - {focals_Pf[0,i_hyc,1]*1e4}')
            print(f'Result: \t {focals_Pf[0,i_hyc,0]*1e4} - {focals_Pf[0,i_hyc,1]*1e4}')
        '''
        Hypercube     Hypercube Imp     Hypercube Inc
        0             0 (precise)       0 (complete)      alpha=1, beta=1  19.12 -  19.12
        1             0 (precise)       1 (incomplete)    alpha=1, beta=0   1.66 - 111.55
        2             1 (imprecise)     0 (complete)      alpha=0, beta=1   7.6  -  37.07
        3             1 (imprecise)     1 (incomplete)    alpha=0, beta=0   0.54 - 183.12
        missing       (consolidated with hyc 3)           alpha=0, beta=0   4.29 -  53.56
         '''
        
            
    
    

if __name__ == '__main__':
    pass
