import os
import glob
import shutil
import traceback
import sys
import time
from datetime import date
import simpleflock
#import ray
import coloredlogs
import logging
from contextlib import contextmanager
import uuid
import numpy as np
import xarray as xr
#import itertools
#coloredlogs.install()
#global logger
#LOG = logging.getLogger('')
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

global pid
pid = str(os.getpid())
        
# importing here to resolve import errors in ray
# as usually, data_manager is part of the environment
# all of its dependencies are pickled and so on

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


import seaborn as sns
import scipy.stats

import ray

'''
PATH=$PATH:/usr/scratch4/sima9999/.local/bin/
export PATH

conda activate my-conda-env
export OMP_NUM_THREADS=1 # limit MKL to the number of workers / cpus to avoid hyperthreading
PYTHONPATH=/usr/scratch4/sima9999/code/:/usr/scratch4/sima9999/git/pyOMA/
export PYTHONPATH
export PYTHONUNBUFFERED=1
IPADDRESS=$(hostname --ip-address)
ray start --head --dashboard-host $IPADDRESS --dashboard-port=5990 --num-cpus 0

lsrun ray start --address=$IPADDRESS:6379 --redis-password='5241590000000000' --num-cpus=16 --block &

# connect to dashboard at  http://141.54.148.100:5990/

# encountered errors
# workers fail due to slurm killing them
# workers fail, if too many different jobs are submitted, i.e. this script is run too many times

'''

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class MultiLock():
    '''
    dbpath = '/vegas/scratch/womo1998/locktest.nc'
    for i in range(3):
        print(i)
        with MultiLock(dbpath):
            with open(f"/vegas/scratch/womo1998/locktest.file",'at') as f:
                f.write(f'{i}\n')
    '''

    def __init__(self, path):
        self._path = path

    def __enter__(self):
        self._this_lockfile = f'{self._path}.{pid}.lock'

        # simpleflock sometimes gives lock to two processes
        with simpleflock.SimpleFlock(f"{self._path}.lock"):
            while True:
                lockfile_list = glob.glob(f'{self._path}.*.lock')
                # print(lockfile_list)
                if len(lockfile_list) > 0:
                    if len(
                            lockfile_list) == 1 and lockfile_list[0] == self._this_lockfile:
                        # this processes lockfile is the only one, we can
                        # continue to modify the ds safely
                        logger.debug(f'Acquired lock on {self._path}.lock')
                        return
                    elif self._this_lockfile in lockfile_list:
                        # another process has created a lockfile meanwhile ->
                        # start over
                        os.remove(self._this_lockfile)
                        time.sleep(np.random.random())
                    else:
                        # another process currently holds the lock for this
                        # file
                        logger.warning(
                            'Wating for lockfile to release: {}'.format(lockfile_list))
                        time.sleep(np.random.random())
                else:
                    # if no other lockfile exists -> create one
                    # continue in while loop to check for race conditions with
                    # othe processes
                    _fd = open(self._this_lockfile, 'w+')
                    _fd.close()

    def __exit__(self, *args):

        os.remove(self._this_lockfile)


class DataManager(object):
    def __init__(self, title, dbfile_in=None, dbfile_out=None,
                 result_dir=None, working_dir=None, overwrite=False):
        '''
        initializes the object and checks all the provided directories and filenames
        '''
        assert isinstance(title, str)
        self.title = title

        if result_dir is None:
            logger.debug(
                'no result_dir specified, using /usr/scratch4/sima9999/work/modal_uq/')
            result_dir = '/usr/scratch4/sima9999/work/modal_uq/'

        if not os.path.isdir(result_dir):
            logger.debug(f'creating directory(s) {result_dir}')
            os.makedirs(result_dir, exist_ok=True)

        self.result_dir = result_dir

        if working_dir is None:
            working_dir = os.getcwd()

        if not os.path.isdir(working_dir):
            logger.debug(f'creating directory(s) {working_dir}')
            os.makedirs(working_dir, exist_ok=True)

        self.working_dir = working_dir

        if dbfile_in is None:
            # use the title, but clean it up a bit
            dbfile_in = "".join(
                i for i in title if i not in "\\/:*?<>|") + '.nc'

        self.dbfile_in = dbfile_in
        if os.path.exists(os.path.join(result_dir, dbfile_in)):
            if overwrite:
                logger.warning(
                    f'Input database file {os.path.join(result_dir, dbfile_in)} already exists and will be overwritten')
                os.remove(os.path.join(result_dir, dbfile_in))

        if dbfile_out is None:
            file, ext = os.path.splitext(dbfile_in)
            dbfile_out = file + '_out' + ext

        self.dbfile_out = dbfile_out
        if os.path.exists(os.path.join(result_dir, dbfile_out)):
            if overwrite:
                logger.warning(
                    f'Output database file {os.path.join(result_dir, dbfile_out)} already exists and will be overwritten')
                os.remove(os.path.join(result_dir, dbfile_out))

        if not os.path.exists(os.path.join(result_dir, dbfile_in)):
            # initialize database file
            with self.get_database(database='in', rw=True) as ds:
                pass

        if not os.path.exists(os.path.join(result_dir, dbfile_out)):
            # initialize database file
            with self.get_database(database='out', rw=True) as ds:
                pass

    def generate_sample_inputs(self, distributions, num_samples, names=None):
        '''
        generates num_distributions x num_samples samples following each
        distribution in distributions

        distributions = (dist_type, dist_params) were dist_type should
        follow the names of the np.random.generator methods
        https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.Generator
            try choice for categorical data and provide an array of possible values

        names may be provided for better mapping of model inputs and distributions

        results will be saved in a separate xarray database

        '''
        num_variables = len(distributions)
        if names is None:
            logger.debug('generating variable names')
            names = [f'var_{i:02d}' for i in range(num_variables)]
        assert len(names) == num_variables

        for dist_type, dist_params in distributions:
            #assert dist_type in np.random.__all__
            #print(dist_params)
            assert isinstance(dist_params, (tuple, list))

        with self.get_database(database='in', rw=True) as ds:

            # seed the RNG, to get the same results, when rerunning
            seeds = np.random.SeedSequence(ds.attrs['entropy']).spawn(len(
                ds.data_vars) + num_variables)[-num_variables:]  # skip over seeds for previously generated samples

            if 'ids' not in ds.coords:
                # we could seed that too, but I think, that might not be
                # necessary
                ids = [str(uuid.uuid4()).split('-')[-1]
                       for i in range(num_samples)]
                #num_params= 3
                ds.coords['ids'] = ids
            else:
                assert len(ds.coords['ids']) == num_samples
                ids = ds.coords['ids']

            for seed, name, (dist_type, dist_params) in zip(
                    seeds, names, distributions):
                rng = np.random.default_rng(seed)
                values = getattr(rng, dist_type)(
                    *dist_params, size=(num_samples,))
                ds[name] = ('ids', values, {
                            'dist_type': dist_type, 'dist_params': dist_params})

        # populate output db with the ids at least
        with self.get_database(database='out', rw=True) as ds:
            ds.coords['ids'] = ids
            ds['_runtimes'] = (['ids'], np.full(
                shape=(num_samples,), fill_value=np.nan))

    def provide_sample_inputs(self, arrays, names=None, attrs=None):
        '''
        provide pre-generated sample input, i.e. for LH or full-factorial sampling
        names may be provided for better mapping of model inputs and distributions
        results will be saved in a separate xarray database

        '''
        num_variables = len(arrays)
        if names is None:
            logger.debug('generating variable names')
            names = [f'var_{i:02d}' for i in range(num_variables)]
        assert len(names) == num_variables

        num_samples = arrays[0].size
        for array in arrays:
            assert array.size == num_samples

        with self.get_database(database='in', rw=True) as ds:

            if not 'ids' in ds.coords:
                if "jid" in names:
                    ind = names.index("jid")
                    ids = arrays[ind]
                    del names[ind]
                    del arrays[ind]
                elif "ids" in names:
                    ind = names.index("ids")
                    ids = arrays[ind]
                    del names[ind]
                    del arrays[ind]
                else:
                    ids = [str(uuid.uuid4()).split('-')[-1]
                           for i in range(num_samples)]
                ds.coords['ids'] = ids
            else:
                assert len(ds.coords['ids']) == num_samples
                assert "jid" not in names
                assert "ids" not in names
                ids = ds.coords['ids']

            for values, name in zip(arrays, names):

                ds[name] = ('ids', values, {})
                #ds['samples'][ds.names==name,:] = values

        # populate output db with the ids at least
        with self.get_database(database='out', rw=True) as ds:
            ds.coords['ids'] = ids
            ds['_runtimes'] = (['ids'], np.full(
                shape=(num_samples,), fill_value=np.nan))

    def enrich_sample_set(self, total_samples):
        '''
        Add additional Monte Carlo samples to a previously evaluated dataset
        does not support provided samples
        '''

        logger.warn("Enrichment currently untested.")
        with self.get_database(database='in', rw=False) as ds:

            new_ds = xr.Dataset()
            num_variables = len(ds.data_vars)
            num_existing_samples = len(ds.coords['ids'])
            num_samples = total_samples - num_existing_samples

            # seed the RNG, to get the same results, when rerunning
            seeds = np.random.SeedSequence(
                ds.attrs['entropy']).spawn(num_variables)

            ids = [str(uuid.uuid4()).split('-')[-1]
                   for i in range(num_existing_samples, total_samples)]
            #num_params= 3
            new_ds.coords['ids'] = ids

            for seed, name in zip(seeds, ds.data_vars):
                da = ds[name]
                dist_type = da.attrs['dist_type']
                dist_params = da.attrs['dist_params']

                rng = np.random.default_rng(seed)
                # theoretically, if all variables are in the same order, etc.
                # this should generate the first num_existing_samples to be
                # equal to the existing
                values = getattr(rng, dist_type)(
                    *dist_params, size=(total_samples,))
                new_ds[name] = ('ids', values[num_existing_samples:])
                # print(name, dist_type, dist_params, values)
                # print('new',new_ds[name], 'old',ds[name])
            #ds_new = ds.combine_first(new_ds)
            ds_new = xr.concat([ds,new_ds], dim='ids')
            # for name in ds_new.data_vars:
                # print('comb',ds_new[name])
            
        dbpath = os.path.join(self.result_dir, self.dbfile_in)
        ds_new.to_netcdf(dbpath, engine='h5netcdf')
        
        # populate output db with the ids at least
        with self.get_database(database='out', rw=False) as ds:
            new_ds = xr.Dataset()
            new_ds.coords['ids'] = ids
            new_ds['_runtimes'] = (['ids'], np.full(
                shape=(num_samples,), fill_value=np.nan))
            for name in ds:
                new_ds[name] = (['ids'], np.full(
                    shape=(num_samples,), fill_value=np.nan))
            #ds_new = ds.combine_first(new_ds)
            
            ds_new = xr.concat([ds,new_ds], dim='ids')
            
        dbpath = os.path.join(self.result_dir, self.dbfile_out)
        ds_new.to_netcdf(dbpath, engine='h5netcdf')

    def evaluate_samples(self, func, arg_vars, ret_names,
                         chwdir=True, re_eval_sample=None, 
                         dry_run=False, default_len=30, **kwargs):
        '''

        func is a function that
            takes jid, result_dir and working_dir, arguments
            takes arguments as named in arg_vars with values from the input db
            returns values as named and in the order of ret_names
            may take additional kwargs

        arg_vars  is a list of mappings (function argument, dataarray name in input dataset)
        
        ret_names Dict {'name1':('dimension1','dimension2',...),'name2':(),...} 
            (as of python 3.7 dicts keep their order)
        
        if chwdir:
            working directory will be changed for each sample to self.working_dir+jid
            and cleaned up upon successful completion ? or on every completion
        else:
            working_directory will be changed to self.working_dir once
        
        default_len is used when setting up new dimensions in an array to specify the
            maximum number of values, that will be stored in that dimension
        
        dry_run may be a boolean, or a specific id_number for debugging purposes
        
        uses ray to distribute tasks to a number of workers.
        a ray head process must be started on stratos
        see top of file for a sbatch file to start a ray worker

        '''
        if not dry_run and not ray.is_initialized():
            if os.path.exists(os.path.expanduser('~/ipaddress.txt')):
                address = open(os.path.expanduser('~/ipaddress.txt'),'rt').read().splitlines()[0]+':6379'
            else:
                address = 'auto'
            ray.init(address=address, _redis_password='5241590000000000')
        '''
        Ray fails with import error .. static TLS
        this can be reconstructed by
        import ray
        ray.init()
        from PyQt5 import QtCore, QtGui, QtWidgets
        from scipy.linalg import _fblas
        
        probably because of intel optimized scipy ? 
        but probably this is caused for another reason
        
        also happens, when matplotlib is being loaded because it wants to load pyqt
        does not happen on a fresh python 3.9 install
        '''
        # @ray.remote
        def setup_eval(func, jid, fun_kwargs,
                       result_dir=self.result_dir, working_dir=None, **kwargs):

            # lock files will stay there, make sure to delete them afterwards
            with simpleflock.SimpleFlock(os.path.join(self.result_dir, f'{jid}.lock'), timeout=1):
                logger.info(f'start computing sample {jid}')

                now = time.time()
                # create the working directory

                if not os.path.exists(result_dir):
                    os.makedirs(result_dir, exist_ok=True)
                    
                try: #  to get LSF temporary working directory
                    working_dir = os.path.join(f'/usr/tmp/{os.environ["LSB_JOBID"]}.tmpdir', jid)
                except KeyError:
                    working_dir = os.path.join('/dev/shm/womo1998/', jid)
                
                if working_dir is not None:
                    cwd = os.getcwd()
                    if not os.path.exists(working_dir):
                        os.makedirs(working_dir, exist_ok=True)
                    os.chdir(working_dir)

                # call model evaluation
                # save results in filesystem -> should be done in func, also
                # with a shortcut in func to return, what has been done
                # previously
                error = False
                try:
                    ret_vals = func(jid=jid, result_dir=result_dir,
                                    working_dir=working_dir, **fun_kwargs, **kwargs)
                    if not isinstance(ret_vals, tuple):
                        raise RuntimeError('The evaluation function must return a tuple.')
                except Exception as e:
                    logger.warning(f'sample {jid} failed')
                    traceback.print_exc()
                    ret_vals = repr(e)
                    error = True
                    if working_dir is not None:
                        err_file = os.path.join(working_dir, jid + '.err')
                        dst = os.path.join(result_dir, jid + '.err')
                        if os.path.exists(err_file):
                            if os.path.exists(dst):
                                os.remove(dst)
                            shutil.move(err_file, dst)
                finally:
                    if working_dir is not None:
                        os.chdir(cwd)
                        subdirs = next(os.walk(working_dir))[1]
                        if len(subdirs) == 0:
                            logger.info(f"Removing working_dir {working_dir}")
                            shutil.rmtree(working_dir, ignore_errors=True)
                        else:
                            logger.warning(
                                f"Cannot remove working_dir {working_dir} (contains subdirectories)")
                        
                        base_dir,_ = os.path.split(working_dir)
                        freedisk=shutil.disk_usage(base_dir).free/(1024**3)
                        nblock = 0
                        while freedisk < 1 and nblock < 30: # other mysterious errors are caused by a full RAM disk (which we use as a working dir)
                            logger.warning(f'Working dir disk is almost full: {freedisk} GB free. Blocking for 30 s.')
                            time.sleep(30)
                            freedisk=shutil.disk_usage(base_dir).free/(1024**3)
                            nblock+=1
                            
                    if error:
                        free = int(os.popen('free -t -g').readlines()[-1].split()[-1])
                        nblock = 0
                        while free < 2 and nblock < 30: # the cause of mysterious RuntimeError("") is OutOfMemory, when pexpect fails to startup ANSYS properly
                            logger.warning(f"System memory low: {free} GB. Blocking further execution for 30 s.")
                            time.sleep(30)
                            free = int(os.popen('free -t -g').readlines()[-1].split()[-1])
                            nblock+=1
                    
                    runtime = time.time() - now        
                    logger.info(
                        f'done computing sample {jid}. Runtime was {runtime} s')
                    
                return jid, ret_vals, runtime
            
        def save_samples(ret_sets, ret_names, num_samples, default_len):
            with self.get_database(database='out', rw=True) as out_ds:
                
                if not ret_sets:
                    return
                
                for jid, ret_vals, runtime in ret_sets:  # first may have thrown an exception, then len() fails
                    if isinstance(ret_vals, (list, tuple)):
                        num_variables = len(ret_vals)
                        break
                else:
                    logger.warning(f"All ret_sets are empty {ret_sets}")
                    #return

                for jid, ret_vals, runtime in ret_sets:

                    if '_exceptions' not in out_ds.data_vars:
                        out_ds['_exceptions'] = (['ids'], np.full(
                            shape=(num_samples,), fill_value=''))
                    if isinstance(ret_vals, str):  # exception repr
                        out_ds['_exceptions'][out_ds.ids == jid] = ret_vals
                        continue
                    else:
                        # reset previous exceptions
                        out_ds['_exceptions'][out_ds.ids == jid] = ''

                    assert len(ret_names) == len(ret_vals)

                    for (name, dims), value in zip(ret_names.items(), ret_vals):
                        if not isinstance(value, np.ndarray):
                            logger.warning(f'Output for {name} should be a (0- or higher-dimensional) numpy array.')
                            value = np.array(value)
                        if name not in out_ds.data_vars:
                            ndims = len(dims)
                            dtype = value.dtype
                            if isinstance(default_len,(int,float)):
                                shape = (num_samples, *[default_len] * ndims)
                            elif isinstance(default_len, (dict)):  # dict of {dim:len,...}
                                shape = [num_samples]
                                for dim in dims:
                                    shape.append(default_len[dim])
                                
                            out_ds[name] = (('ids', *dims), np.full(
                                shape=shape, fill_value=np.nan, dtype=dtype))
                            # for dim in dims:
                            #     if dim not in out_ ds.coords:
                            #         out_ds.coords[dim] = np.arange(default_len)
                        
                        pos_dict = {dim:slice(None,siz) for dim, siz in zip(dims, value.shape)}
                            
                        out_ds[name].loc[jid][pos_dict] = value

                        logger.debug(out_ds[name][out_ds.ids == jid])

                    out_ds['_runtimes'][out_ds.ids == jid] = runtime

                    logger.debug(out_ds['_runtimes'][out_ds.ids == jid])
        
        if isinstance(dry_run, str):
            if re_eval_sample is None:
                re_eval_sample = dry_run
            if re_eval_sample!=dry_run:
                logger.warning(f'Trying to debug sample {dry_run} but re-evaluating {re_eval_sample}')
                     
        # open database read-only, without locking
        with self.get_database(database='in') as in_ds, self.get_database(database='out', rw=False) as out_ds:
            
            if not chwdir:
                os.chdir(self.working_dir)
                logger.debug(f'current working directory {self.working_dir}')

            num_samples = in_ds.ids.size
            
            if re_eval_sample is not None:
                in_ds = in_ds.sel(ids=[re_eval_sample])

            futures = []

            # TODO: Improvement: Estimate job size and submit big jobs first,
            # i.e. sort key = jobsize, but also add some smallest jobs
            # to the beginning to see any errors quickly
            
            for jid_ind in sorted(range(in_ds.ids.size),
                                  key=lambda _: np.random.random()):

                if (not out_ds['_runtimes'][jid_ind].isnull()) \
                    and re_eval_sample is None \
                    and out_ds['_exceptions'][jid_ind].data.item()=='':
                    
                    continue

                jid = in_ds['ids'][jid_ind].copy().item()

                fun_kwargs = {}
                for arg, var in arg_vars.items():
                    fun_kwargs[arg] = in_ds[var][jid_ind].copy().item()

                if chwdir:
                    working_dir = os.path.join(self.working_dir, jid)
                else:
                    working_dir = None

                if dry_run:
                    if isinstance(dry_run, str) and dry_run!=jid:
                        continue
                    # evaluates samples in the regular way, i.e. one at a time
                    ret_sets = [setup_eval(func, jid, fun_kwargs, working_dir=working_dir, **kwargs)]
                    save_samples(ret_sets, ret_names, num_samples, default_len)
                else:
                    #print(fun_kwargs)
                    worker_ref = setup_eval.remote(
                        func, jid, fun_kwargs, working_dir=working_dir, **kwargs)
                    
                    futures.append(worker_ref)
        
        futures = set(futures)
        logger.info(f'{len(futures)} jobs have been submitted for evaluation.')

        if dry_run:
            return

        while True:
            ready, wait = ray.wait(
                list(futures), num_returns=min(len(futures), 10), timeout=300)
            try:
                ret_sets = ray.get(ready)
            except ray.exceptions.RayTaskError as e:
                traceback.print_exc()
                logger.warning(repr(e))
                ret_sets = []
                
            
            save_samples(ret_sets, ret_names, num_samples, default_len)

            if len(wait) == 0:
                break

            size_before = len(futures)
            futures.difference_update(ready)
            logger.info(
                f"Finished {len(ready)} samples. Remaining {len(futures)} samples. (before {size_before})")
        ray.shutdown()
        return

    def post_process_samples(self, db='merged', func=None, 
                             names=None, labels=None,
                             max_categories=30, draft=True, **kwargs):
        '''
        drawing scatterplot matrices, bar charts, box plots etc.
        '''

        
        def categorize_data(datax, datay, categories):
            categories = np.array(categories)
            num_categories = len(categories)
            datay_cat = [[] for _ in range(num_categories)]

            for i_cat in range(num_categories):
                indexer = datax == categories[i_cat]
                if indexer.any():
                    datay_cat[i_cat] = datay[indexer]

            return datay_cat

        def count_data(datax, datay, categoriesx, categoriesy):
            categoriesx = np.array(categoriesx)
            categoriesy = np.array(categoriesy)

            num_catx = len(categoriesx)
            num_caty = len(categoriesy)

            datax_cat = [np.nan for _ in range(
                num_catx * num_caty)]  # mean -> bin center
            datay_cat = [np.nan for _ in range(num_catx * num_caty)]
            sizes = [0 for _ in range(num_catx * num_caty)]
            for i_catx in range(num_catx):
                indexerx = datax == categoriesx[i_catx]
                for i_caty in range(num_caty):
                    indexery = datay == categoriesy[i_caty]
                    indexer = np.logical_and(indexerx, indexery)
                    if indexer.any():
                        # copy for convenience
                        datax_cat[i_catx * num_caty +
                                  i_caty] = categoriesx[i_catx]
                        # copy for convenience
                        datay_cat[i_catx * num_caty +
                                  i_caty] = categoriesy[i_caty]
                        sizes[i_catx * num_caty + i_caty] = np.sum(indexer)

            return datax_cat, datay_cat, sizes

        def scatterplot_matrix(data, all_categories={}, nbins=20, scales={}, draft=True, labels=None, **kwargs):
            """Plots a scatterplot matrix of subplots.  Each row of "data" is plotted
            against other rows, resulting in a nrows by nrows grid of subplots with the
            diagonal subplots labeled with "names".

            For categorical data (usually integer  data) the category values may be
            provided as all_categories, these are also usedto generate the bins for
            the histograms. In general, the number of bins in the histograms can be
            changed to each variable by providing all_nbins as an array.

            Additional keyword arguments are
            passed on to matplotlib's "plot" command. Returns the matplotlib figure
            object containg the subplot grid."""
            
            numdata, numvars = data.shape
            names = data.keys()
            #print(numdata,numvars,names)
            
            fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=
                                     kwargs.pop('figsize',(10, 10)), sharex='col', sharey='row', squeeze=False)

            for ax in axes.flat:
                # Hide all ticks and labels
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)

                # Set up ticks only on one side for the "edge" subplots...
                if ax.is_first_col() and not ax.is_first_row():
                    ax.yaxis.set_ticks_position('left')
                    ax.yaxis.set_visible(True)
                if ax.is_last_row():
                    ax.xaxis.set_ticks_position('bottom')
                    ax.xaxis.set_visible(True)

            data_limits = {}
            
            all_bins={}
            
            if scales is not None:
                assert len(scales) == numvars

            for ij in range(numvars):
                # can we deduce from the data if it is categorical?
                # it is visible in the histograms, i.e. many empty bins?
                # relies on the binning algorithm of hist
                # should use clustering or provide the categories as an
                # argument
                ax = axes[ij, ij].twinx()
                axes[ij, ij] = ax
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                
                name = names[ij]
                
                logger.debug(f'hist {name}')
                if name in all_categories:
                    bin_centers = all_categories[name]
                    bin_widths = (bin_centers[1:] - bin_centers[:-1]) / 2
                    bins = np.concatenate(
                        (bin_centers[0:1] - bin_widths[0:1], bin_centers[:-1] + bin_widths, bin_centers[-1:] + bin_widths[-1:]))
                    range_ = None
                else:
                    data_limits[name] = np.nanquantile(data[name], [0.02, 0.98])
                    bin_centers=None
                    bins = nbins
                    range_ = data_limits[name]
                    
                sample_counts, bins_gen, _ = ax.hist(
                    data[name], bins=bins, range=range_, density=False, zorder=-10, facecolor='lightgrey', edgecolor='dimgrey', alpha=.5)
                # sns.distplot may be another alternative to hist
                #print(name,sample_counts, bin_centers)
                # save for later to assign boxplot widths
                all_bins[name] = bins_gen
                # ax.set_ylim((0,max(sample_counts)))

                

            # Plot the data.
            for x, y in zip(*np.triu_indices_from(axes, k=1)):
                
                namex = names[x]
                namey = names[y]
                logger.debug(f'scatter {namex},{namey}')
                
                this_data = data[[namex, namey]].dropna()
                # categorical (i.e. integer) data should be displayed as box
                # plots rather than scatter plots
                if namex in all_categories and namey in all_categories:  # both categorical -> draw bubbles

                    datax, datay, sizes = count_data(
                        this_data[namex], this_data[namey], all_categories[namex], all_categories[namey])

                    # sizes must be normalized, scaled to the binsize and to
                    # the axes size in points
                    bbox = axes[y, x].get_window_extent().transformed(
                        fig.dpi_scale_trans.inverted())
                    width, height = bbox.width * fig.dpi, bbox.height * fig.dpi
                    binsize = min(
                        width / (len(all_bins[namex]) - 1),
                        height / (len(all_bins[namey]) - 1)) * .6

                    # sizes are in points**2
                    sizes = np.power(
                        np.array(sizes) / np.nanmean(sizes) * binsize, 2)

                    axes[y, x].scatter(datax, datay, s=sizes,
                                       alpha=.5, color='dimgrey')
                elif namex in all_categories or namey in all_categories:  # xdata or ydata categorical -> draw normal boxplots
                    if namex in all_categories:
                        widths = np.diff(all_bins[namex]) * 0.75
                        pos = all_categories[namex]
                        bp_data = categorize_data(
                            this_data[namex], this_data[namey], pos)
                        vert=True
                        
                    elif namey in all_categories:
                        widths = np.diff(all_bins[namey]) * 0.75
                        pos = all_categories[namey]
                        bp_data = categorize_data(
                            this_data[namey], this_data[namex], pos)
                        vert=False
                    
                    if True:
                        
                        axes[y, x].boxplot(bp_data,
                                           positions=pos,
                                           vert=vert,
                                           widths=widths,
                                           sym='',
                                           boxprops=dict(color='dimgrey'),
                                           medianprops=dict(color='black'),
                                           whiskerprops=dict(color='grey'))
                    else:
                        # strip plot plots the categories at x values 0 ... num_categories and does not obey numerical categories
                        # this will be fixed in a future release: https://github.com/mwaskom/seaborn/issues/2429 -> possible v0.12.0
                        # There will be a parameter (fixed_scale) to disable the forced categorical mapping,
                        # meaning that the placement along the "categorical" axis will respect the numeric values of that variable.
                        # Fixed scaling will remain the default behavior.
                        sns.stripplot(
                            x=namex, y=namey, ax=axes[y, x], data=this_data, orient=('h','v')[vert], color='dimgrey', jitter=0.2,size=1)

                else:  # draw hexbin plots
                    
                    #axes[y, x].plot(data[x], data[y], ls='none', marker='.', color='dimgrey', **kwargs)
                    color_s = matplotlib.colors.to_rgba('dimgrey', alpha=0)
                    color_e = matplotlib.colors.to_rgba('dimgrey', alpha=1)


                    cmap = LinearSegmentedColormap.from_list(
                        'CustomCmap', colors=[color_s, color_e])  # fff white with alpha
                    
                    if scales is not None:
                        xscale = scales[x]
                        yscale = scales[y]
                    else:
                        xscale='linear'
                        yscale='linear'
                    
                    limsx = data_limits[namex]
                    limsy = data_limits[namey]
                    
                    selx = np.logical_and(this_data[namex] >= limsx[0], this_data[namex] <= limsx[1])
                    sely = np.logical_and(this_data[namey] >= limsy[0], this_data[namey] <= limsy[1])
                    sel = np.logical_and(selx, sely)
                    
                    if draft:
                        axes[y, x].scatter(this_data[namex][sel], this_data[namey][sel],marker='.', c='dimgrey', edgecolors='none',s=1, )
                    else:
                        axes[y, x].hexbin(this_data[namex][sel], this_data[namey][sel], gridsize=200, bins='log', cmap=cmap,
                                      xscale=xscale, yscale=yscale,
                                      edgecolors='face')

                rho, pval = scipy.stats.spearmanr(this_data[namex], this_data[namey])
                #corr_coef = np.corrcoef(x,y)
                #rho = corr_coef[1,0]
                #corr_coef = rho
                axes[x, y].annotate('${:1.3f}$'.format(rho), (0.5, 0.5),
                                    xycoords='axes fraction',
                                    ha='center', va='center')

            # Label the diagonal subplots...
            for i, label in enumerate(labels):
                if "_" in label and not '$' in label and matplotlib.rcParams["text.usetex"] is True:
                    # might want to use pylatexenc in the future
                    label = label.replace("_", "\_")
                ab = axes[i, i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                                         ha='center', va='center', zorder=10, backgroundcolor='#FFFFFFAA')
            # print(data_limits)
            # # set the data limits
            # for i, name in enumerate(names):
                # if name not in all_categories:
                    # continue
                    # lims = data_limits[name]
                    # axes[-1, i].set_xlim(lims)
                    # axes[i, 0].set_ylim(lims)
                

            if scales is not None:
                for i, scale in enumerate(scales):
                    axes[0, i].set_xscale(scale)
                    if i > 0:  # skip yscale on first row, there are just correlation coefficients anyway
                        axes[i, 0].set_yscale(scale)
            
            fig.subplots_adjust(hspace=0.05, wspace=0.05, top=0.98, bottom=0.05, left=0.05, right=0.98)
            
            return fig
        
        with self.get_database(database=db, rw=False) as ds:
            
            '''
            we might consider seaborn for statistical visualizaztion:
                pairgrid instead of scatterplotmatrix
                histplot for hist (with kde=True)
                stripplot instead of boxplot for categorical data
                hexbinplot instead of scatter
                histplot for double categorical data
                
                
            
            seaborn likes to use pandas.DataFrame
            xarray.DataArrays can be converted to_pandas
                0D -> xarray.DataArray
                1D -> pandas.Series
                2D -> pandas.DataFrame
            we will deal with 2D data for plotting anyway. for 3D data, they will have to be reduced somehow
            
            advantages:
                can use native seaborn functionality
                can use pandas functionality, e.g. for grouping values
                might reduce work for usage, when other plots should be integrated
                arrays for normal pyplot can still be extracted
            disadvantages:
                another framework (pandas, seaborn) to "learn"
                might not be flexible enough in case, more complex cases have to be handled
                requires more work for implementation
                
            go for it:
                routine for sorting out 3D data
                implement and test to_pandas
                implement and test DataFrame.melt() as an alternative to categorization
                    check double categorization etc.
                integrate with scatterplot_matrix

            '''
            if func is not None:
                rw = kwargs.pop('rw', False)
                logger.info(f'Applying user-supplied function {func.__name__} to dataset and save results ({rw})...')
                    
                    
                ds = func(ds, **kwargs)
                if ds is None:
                    logger.warning(f'{func} did not return a dataset. Exiting!')
                
                # workaround, since func may have returned a new object of ds, also with new indices etc.
                # adding a underscore to be on the safe side
                if rw:
                    file, ext = os.path.splitext(self.dbfile_in)
                    dbfile_out = file + '_processed' + ext
                    dbpath = os.path.join(self.result_dir, dbfile_out)
                    logger.debug(f'Saving database to {dbpath}')
                    ds.close()
                    ds.to_netcdf(dbpath, engine='h5netcdf')
                    ds.close
                    logger.info('We saved the database outside the safe loop, ensure to load the "processed" db for future use')
            
            if '_exceptions' in ds.data_vars:
                ind = ds._exceptions=='' # no exception
                ds = ds.isel(ids=ind)    
            print(f"number of samples {len(ds.ids)}")
            if names is None:
                names = [name for name in ds.data_vars]
            
            # Sort out 3D data
            try:
                for i in reversed(range(len(names))):
                    name = names[i]
                    if name.startswith('_'):
                        del names[i]
                    elif name in ds.coords:
                        del names[i]
                    elif not name in ds.data_vars:
                        logger.warning(f'Variable {name} is not in the dataset and will be skipped')
                        del names[i]
                    elif len(ds[name].data.shape) != 1:
                        logger.warning(f'DataArray {name} has {len(ds[name].data.shape)} dimensions. Try to pre-process (flatten, subdivide, etc.) and remove it')
                        del names[i]
                    elif ds[name].dtype == np.dtype('O'):
                        logger.warning(f'DataArray {name} is an object dtype, must be preprocessed!')
                        #continue
                        del names[i]
                    elif np.isnan(ds[name].data).all():
                        logger.warning(f'DataArray {name} is empty. Try to remove it from names!')
                        del names[i]
            except:
                print(name)
                raise
            
            
            logger.info(f'Categorizing data arrays ...')
            categorical_dists = ['integers', 'choice', 'binomial',
                                 'hypergeometric', 'geometric', 'poisson']
            # geometric, poisson, are discrete but unbound, an overflow bin would have to be assigned, maybe if needed later
            all_categories = {}
            for name in names:
                da = ds[name]
                if 'dist_type' in da.attrs:
                    dist_type = da.attrs['dist_type']
                    if dist_type in categorical_dists:
                        dist_params = da.attrs['dist_params']
                        if dist_type == 'integers':
                            categories = np.arange(*dist_params)
                        elif dist_type == 'choice':
                            categories = dist_params[0]
                        else:
                            raise NotImplementedError(f'Category generation for the {dist_type} distribution is currently not implemented.')
                        
                        if len(categories) <= max_categories:
                            all_categories[name] = categories
                        else:
                            logger.warning(f'Number of categories exceeds max_categories {max_categories} for parameter {name}')
            
           
        
            numvars = len(names)
            
            # TODO: might have to remove empty coordinates before
            #df = ds.reset_coords(['ids_','modes'])[names].to_dataframe()
            df = ds[names].to_dataframe()
            #print(ds)
            
            
            
        # g = sns.PairGrid(df, diag_sharey=True, despine=False)
        #
        # #g.map_diag(sns.histplot)
        # print(np.triu_indices(numvars, 0))
        # for indices in zip(*np.tril_indices(numvars, k=0)):
            # x, y = indices
            # print(indices)
            # if x == y:
                # continue
                # g._map_bivariate(sns.histplot, [indices])
            # elif names[x] in all_categories and names[y] in all_categories: # categorical data
                # ax = self.axes[x,y]
                #
                #
                # #g._map_bivariate(plt.scatter, [indices])
            # elif names[x] in all_categories or names[y] in all_categories: #categorical data
                # g._map_bivariate(sns.stripplot, [indices])
        #
        #g.map_diag(sns.kdeplot)
        #g.map_offdiag(sns.stripplot)
        #g._map_bivariate(func, indices) # for categorical data
        # plt.show()
        logger.info(f'Creating the scatterplot matrix... ')
        #logger.debug(f'{df}')
        #logger.debug(f'{all_categories}')
        #logger.debug(f'{kwargs.get("scales",None)}{draft}')
        if labels is None:
            labels=names
        else:
            assert len(labels)==len(names)
            
        fig = scatterplot_matrix(data=df,
                           all_categories=all_categories,
                           scales=kwargs.pop('scales', None),
                           draft=draft,
                           labels=labels,
                           **kwargs)
        
        return fig

    @classmethod
    def from_existing(cls, dbfile_in,
                      result_dir='/usr/scratch4/sima9999/work/modal_uq/',
                      working_dir=None):
        assert os.path.exists(os.path.join(result_dir, dbfile_in))

        cls.result_dir = result_dir
        cls.dbfile_in = dbfile_in
        with cls.get_database(cls) as ds:
            if working_dir is None:
                working_dir = ds.attrs['working_dir']
            if not os.path.exists(working_dir):
                logger.warning(f'Working directory {working_dir} does not exist. Creating!')
                os.makedirs(working_dir)
                
            cls.working_dir = working_dir

            title = ds.attrs['title']
            cls.title = title

            result_dir_ = ds.attrs['result_dir']
            if result_dir_ != result_dir:
                logger.warning(
                    f'result dir from db {result_dir_} differs from given result_dir {result_dir}')

            dbfile_in_ = ds.attrs['dbfile_in']
            if dbfile_in != dbfile_in_:
                logger.warning(
                    f'dbfile_in from db {dbfile_in_} differs from given dbfile_in {dbfile_in}')

            dbfile_out = ds.attrs['dbfile_out']
            cls.dbfile_out = dbfile_out
            # create and  populate output db with the ids at least
            if not os.path.exists(os.path.join(result_dir, dbfile_out)):
                logger.info(
                    f'output db {os.path.join(result_dir,dbfile_out)} has been deleted. Recreating!')
                with cls.get_database(cls, database='out', rw=True) as out_ds:
                    ids = ds.ids.data
                    out_ds.coords['ids'] = ids
                    out_ds['_runtimes'] = (['ids'], np.full(
                        shape=(len(ids),), fill_value=np.nan))

        return cls(title, dbfile_in, dbfile_out, result_dir, working_dir)

    @contextmanager
    def get_database(self, database='in', rw=False):
        '''
        get a handle to the database
        if rw is required, lock the db

        dbfile_in can be safely accessed concurrently
        dbfile_out must be locked before operation

        this should be used as a contextmanager, to ensure proper file closing and lock removal
        '''

        assert database in ['in', 'out', 'merged', 'processed']

        if database == 'in':
            dbfile = self.dbfile_in
        elif database == 'out':
            dbfile = self.dbfile_out
            if not rw:
                logger.warning(
                    'You have chosen to open the output-db in read-only mode, changes will not be saved.')
        elif database == 'merged':
            if rw:
                logger.warning('merged database can not be opend in rw mode')
            try:
                with self.get_database(database='in', rw=False) as in_ds, self.get_database(database='out', rw=False) as out_ds:
                    ds = in_ds.combine_first(out_ds)
                    yield ds
            except BaseException:
                raise
            finally:
                ds.close()
            return
        elif database == 'processed':
            file, ext = os.path.splitext(self.dbfile_in)
            dbfile = file + '_processed' + ext

        dbpath = os.path.join(self.result_dir, dbfile)

        if not os.path.exists(dbpath):
            logger.debug(f'file {dbfile} will be created')
            with MultiLock(dbpath):
                # create database, if it does not exist
                if not os.path.exists(dbpath):
                    ds = xr.Dataset()
                    ds.attrs['working_dir'] = self.working_dir
                    ds.attrs['result_dir'] = self.result_dir
                    ds.attrs['dbfile_in'] = self.dbfile_in
                    ds.attrs['dbfile_out'] = self.dbfile_out
                    ds.attrs['date'] = date.today().__repr__()
                    ds.attrs['title'] = self.title
                    ds.attrs['entropy'] = np.random.randint(
                        np.iinfo(np.int32).max)
                    ds.to_netcdf(dbpath, engine='h5netcdf')
                    #ds.to_netcdf(dbpath, format='netcdf4')
                    ds.close()
        else:
            logger.debug(f'opening existing file {dbfile}')

        if not rw:
            try:
                # open database
                ds = xr.open_dataset(dbpath, engine='h5netcdf')
                #ds = xr.open_dataset(dbpath)
                ds.load()
                ds.close()
                # yield database
                yield ds
            except BaseException:
                raise
            finally:
                # close database
                ds.close()
        else:
            # create lock
            # TODO: locks seem to race on each other....
            with MultiLock(dbpath):
                try:
                    # open database
                    ds = xr.open_dataset(dbpath, engine='h5netcdf')
                    #ds = xr.open_dataset(dbpath)
                    ds.load()
                    # ds.close()
                    # yield database
                    yield ds
                except BaseException:
                    raise
                finally:
                    # save and close database
                    ds.close()
                    logger.debug(f'Saving database to {dbpath}')
                    ds.to_netcdf(dbpath, engine='h5netcdf')
                    #ds.to_netcdf(dbpath, format='netcdf4')
                    ds.close()
            # remove lock automatically

        return

    def clear_locks(self):
        lock_list = glob.glob(os.path.join(self.result_dir, '*.lock'))
        with self.get_database(database='out', rw=False) as out_ds:
            for file in lock_list:
                jid = os.path.splitext(os.path.split(file)[-1])[0]
                if np.isnan(out_ds['_runtimes'][out_ds.ids == jid]):
                    logger.warning(
                        f'A lock exists for {jid} but no results are in the db')
                logger.debug(f'removing file {file}')
                os.remove(file)

    def clear_wdirs(self, delnonempty=False):
        dir_list = glob.glob(os.path.join(self.result_dir, '*/'))
        with self.get_database(database='out', rw=False) as out_ds:
            for dir_ in dir_list:
                jid = os.path.split(os.path.dirname(dir_))[-1]
                if not jid in out_ds.ids:
                    continue
                if np.isnan(out_ds['_runtimes'][out_ds.ids == jid]):
                    logger.warning(
                        f'A lock exists for {jid} but no results are in the db')
                logger.debug(f'removing file {dir_}')
                try:
                    os.removedirs(dir_)
                except Exception as e:
                    if e.errno == 39:
                        logger.warning(f'Working directory not empty {dir_}')
                        if delnonempty:
                            shutil.rmtree(dir_)
                    else:
                        logger.warning(repr(e))

    def clear_failed(self, dryrun=True):
        lock_list = glob.glob(os.path.join(self.result_dir, '*.lock'))
        with self.get_database(database='out', rw=False) as out_ds:
            for file in lock_list:
                jid = os.path.splitext(os.path.split(file)[-1])[0]
                if '_exceptions' in out_ds.data_vars:
                    if out_ds['_exceptions'][out_ds.ids == jid].data:
                        print(out_ds['_exceptions']
                              [out_ds.ids == jid].data, file)
                        if not dryrun:
                            os.remove(file)
                if np.isnan(out_ds['_runtimes'][out_ds.ids == jid]):
                    print(out_ds['_runtimes'][out_ds.ids == jid], file)
                    if not dryrun:
                        os.remove(file)
            for jid in out_ds.ids:
                if '_exceptions' in out_ds.data_vars:
                    if out_ds['_exceptions'][out_ds.ids == jid].data:
                        print(out_ds['_exceptions']
                              [out_ds.ids == jid].data)
                        if not dryrun:
                            os.remove(file)
                if np.isnan(out_ds['_runtimes'][out_ds.ids == jid]):
                    print(out_ds['_runtimes'][out_ds.ids == jid])
                    if not dryrun:
                        os.remove(file)

    def as_text(self, fname, var_names=None):
        print(fname)

        with self.get_database(database='merged', rw=False) as ds:

            if var_names is None:
                var_names = list(ds.data_vars) + list(ds.coords)
            # var_arrays=[]
            for var in var_names:
                assert var in ds.data_vars or var in ds.coords
                # var_arrays.append(ds[var].data)
            ds[var_names].to_dataframe().to_csv(self.result_dir + fname, '\t')


def test_fun(a, b, c, result_dir=None, factor=1, **kwargs):
    shutil.rmtree(result_dir)
    time.sleep(0.1)
    return a + b + c * factor, a * b / factor


def process_model_perf(ds):

    for name in ['mean', 'min', 'max', 'std']:
        ds[f'{name}_per_nnod'] = ds[name] / ds['num_nodes']
    for name in ['mean', 'min', 'max', 'std']:
        ds[f'{name}_per_ts'] = ds[name] / ds['chunksize']

    ds['frac_nodes'] = ds['num_meas_nodes'] / ds['num_nodes']
    del_names = [  # 'd0','f_scale',
        'mean', 'min', 'max', 'std']

    for name in ds.data_vars:
        if name.startswith('_'):
            del_names.append(name)

    for name in del_names:
        del ds[name]

    return ds


def student_manager(ambient, nonlinear, friction):
    if ambient and nonlinear and not friction:
        title = "nonlinear_ambient"
    elif ambient and not nonlinear and not friction:
        title = "linear_ambient"
    elif not ambient and not nonlinear and not friction:
        title = "linear_decay"
    elif not ambient and nonlinear and not friction:
        title = "nonlinear_decay"
    elif not ambient and not nonlinear and friction:
        title = "friction_decay"
    elif ambient and not nonlinear and friction:
        title = "friction_ambient"
    elif not ambient and nonlinear and friction:
        title = "general_decay"
    elif ambient and nonlinear and friction:
        title = "general_ambient"
    else:
        raise RuntimeError(
            f'This combination of inputs is not supported: {ambient}, {nonlinear}, {friction}')
    print(title)
    savefolder = '/vegas/scratch/womo1998/data_hadidi/datasets_' + title + '/'
    result_dir = '/vegas/scratch/womo1998/data_hadidi2/datasets_' + title + '/'

    if not os.path.exists(result_dir + title + '.nc') or False:
        data_manager = DataManager(title=title, working_dir='/dev/shm/womo1998/',
                                   result_dir=result_dir,
                                   overwrite=True)

#         jids,ks,cs,d_maxs,d_means,fric_visc_rats,nl_itys,deltats = [],[],[],[],[],[],[],[]
#         with open(f'{savefolder}description.txt','tr') as descr:#, open(f'{source_folder}description_new.txt','tw') as descr_new:
#             descr.readline()
#             for i,line in enumerate(descr):
#                 jid,k,c,d_max,d_mean,fric_visc_rat,nl_ity,deltat = [float(s.strip()) if j>0 else s.strip() for j,s in enumerate(line.split(',')) ]
#                 for l,v in zip([jids,ks,cs,d_maxs,d_means,fric_visc_rats,nl_itys,deltats],[jid,k,c,d_max,d_mean,fric_visc_rat,nl_ity,deltat]):
#                     l.append(v)
#
#         jids=np.array(jids)
#         ks=np.array(ks)
#         cs=np.array(cs)
#         d_maxs = np.array(d_maxs)
#         d_means=np.array(d_means)
#         fric_visc_rats=np.array(fric_visc_rats)
#         nl_itys=np.array(nl_itys)
#         deltats=np.array(deltats)
#
#
#         data_manager.provide_sample_inputs([jids,ks,cs,d_maxs,d_means,fric_visc_rats,nl_itys,deltats],
#                                            ['jid','k','c','d_max','d_mean','fric_visc_rat','nl_ity','deltat'])
#         data_manager.generate_sample_inputs(distributions=[('uniform',(-20,20)),],
#                                                        num_samples=len(jids),
#                                                        names=['snr'])

        data_manager.generate_sample_inputs(names=['omega',
                                                   'zeta',
                                                   'dt_fact',
                                                   'snr'],
                                            distributions=[('uniform', (1, 15)),
                                                           ('uniform', (0.1, 10)),
                                                           ('uniform',
                                                            (0.001, 0.015)),
                                                           ('uniform', (-20, 20)),
                                                           ],
                                            num_samples=1000 - 722)

        if nonlinear:
            data_manager.generate_sample_inputs(names=['nl_ity', ],
                                                distributions=[
                                                    ('uniform', (-0.5, 0.5)), ],
                                                num_samples=1000 - 722)
        else:
            data_manager.generate_sample_inputs(names=['nl_ity', ],
                                                distributions=[
                                                    ('uniform', (0, 0)), ],
                                                num_samples=1000 - 722)
        if friction:
            data_manager.generate_sample_inputs(names=['fric_visc_rat', ],
                                                distributions=[
                                                    ('uniform', (0, 1)), ],
                                                num_samples=1000 - 722)
        else:
            data_manager.generate_sample_inputs(names=['fric_visc_rat', ],
                                                distributions=[
                                                    ('uniform', (0, 0)), ],
                                                num_samples=1000 - 722)
        if ambient:
            data_manager.provide_sample_inputs([np.array([None for _ in range(1000 - 722)])],
                                               ['d0'])
            data_manager.generate_sample_inputs(names=['num_cycles',
                                                       'f_scale', ],
                                                distributions=[('integers', (300, 2000)),
                                                               ('uniform',
                                                                (0, 10)),
                                                               ],
                                                num_samples=1000 - 722)
        else:
            data_manager.generate_sample_inputs(names=['num_cycles',
                                                       'd0', ],
                                                distributions=[('integers', (3, 20)),
                                                               ('uniform', (1e-3, 100)), ],
                                                num_samples=1000 - 722)
            data_manager.provide_sample_inputs([np.array([None for _ in range(1000 - 722)])],
                                               ['f_scale'])

        data_manager.post_process_samples(db='in')

    elif False:
        # add_samples
        data_manager = DataManager.from_existing(dbfile_in="".join(
            i for i in title if i not in "\\/:*?<>|") + '.nc', result_dir=result_dir)
        data_manager.enrich_sample_set(total_samples=1000)

    elif True:

        from model import mechanical
        # evaluate input samples
        data_manager = DataManager.from_existing(dbfile_in="".join(
            i for i in title if i not in "\\/:*?<>|") + '.nc', result_dir=result_dir)

        #data_manager.evaluate_samples(func=manipulate_student_fun, arg_vars={'snr_db':'snr'}, ret_names=['signal_power', 'noise_power','deltat'], chwdir = True, readfolder=savefolder)
        data_manager.evaluate_samples(func=mechanical.student_data_part2, arg_vars={'omega': 'omega',
                                                                                    'zeta': 'zeta',
                                                                                    'dt_fact': 'dt_fact',
                                                                                    'num_cycles': 'num_cycles',
                                                                                    'f_scale': 'f_scale',
                                                                                    'd_scale': 'd0',
                                                                                    'nl_ity': 'nl_ity',
                                                                                    'fric_visc_rat': 'fric_visc_rat',
                                                                                    'snr_db': 'snr'},
                                      ret_names=[
                                          'k', 'c', 'd_max', 'fsl', 'signal_power', 'noise_power', 'deltat'],
                                      chwdir=True, dry_run=True)
        #mechanical.student_data_part2(jid, result_dir, omega, zeta, dt_fact,num_cycles, f_scale, d_scale, nl_ity, fric_visc_rat, snr_db, **kwargs)
        #
    elif True:
        data_manager = DataManager.from_existing(dbfile_in="".join(
            i for i in title if i not in "\\/:*?<>|") + '.nc', result_dir=result_dir)
        data_manager.post_process_samples(db='in')
        data_manager.post_process_samples(db='out')
        data_manager.post_process_samples(db='merged')
        data_manager.clear_locks()
        # data_manager.clear_wdirs()
        data_manager.as_text('test.txt', ['ids', 'k', 'c', 'd_max', 'fsl',
                                          'fric_visc_rat', 'nl_ity', 'noise_power', 'signal_power', 'deltat'])


def manipulate_student_fun(jid, snr_db, readfolder, result_dir):
    import scipy.io
    print(jid)
#     rn=np.random.rand(1)
#     print(rn)
#     if rn>0.9: raise RuntimeError

    do_plot = False

    snr = 10**(snr_db / 10)

    file = readfolder + jid + '.csv'
    array = np.loadtxt(file)

    power = np.mean(array[:, 1]**2)
    if do_plot:
        fig, axes = plt.subplots(3, 1, sharex=True)
        axes[0].psd(array[:, 1], Fs=array.shape[0] /
                    (array[-1, 0] - array[0, 0]))
    # decimate
    array = array[1::6, :]
    N = array.shape[0]
    # add noise
    noise_power = power / snr
    noise = np.random.normal(0, np.sqrt(noise_power), N)
    power_noise = np.mean(noise**2)

    snr_actual = power / power_noise
    snr_actual_db = 10 * np.log10(snr_actual)

    array[:, 1] += noise
    if 'ambient' in result_dir:
        cov = np.zeros((N, 2))
        for i in range(N):
            cov[i, 1] = array[:N - i, 1].dot(array[i:, 1]) / (N - i)
        cov[:, 0] = array[:, 0]
        if do_plot:
            axes[1].psd(array[:, 1], Fs=array.shape[0] /
                        (array[-1, 0] - array[0, 0]))
            axes[2].plot(cov[:, 0], cov[:, 1])
            axes[1].set_xlim(
                (0, array.shape[0] / (array[-1, 0] - array[0, 0]) / 2))
            plt.show()

        mdict = {'disp': array, 'corr': cov}
    else:
        mdict = {'disp': array}
    file = os.path.join(result_dir, jid + '.mat')
    scipy.io.savemat(file, mdict, appendmat=True, format='5',
                long_field_names=False, do_compression=True, oned_as='row')

    return power, power_noise, (array[-1, 0] -
                                array[0, 0]) / (array.shape[0] - 1)


def test_categorical():
    title = 'test_categorical'
    savefolder = '/usr/scratch4/sima9999/work/modal_uq/'
    result_dir = '/usr/scratch4/sima9999/work/modal_uq/'

    if not os.path.exists(result_dir + title + '.nc') or False:
        data_manager = DataManager(title=title, working_dir='/dev/shm/womo1998/',
                                   result_dir=result_dir,
                                   overwrite=True)
        data_manager.generate_sample_inputs(names=['N',
                                                   'dec_fact',
                                                   'numtap_fact',
                                                   'nyq_rat'],
                                            distributions=[('choice', [2**np.arange(5, 18)]),
                                                           ('integers', (2, 15)),
                                                           ('integers', (5, 61)),
                                                           ('uniform', (2, 4)),
                                                           ],
                                            num_samples=1000)

        data_manager.post_process_samples(db='in', names=['N','dec_fact','numtap_fact','nyq_rat'], scales = ['log','linear','linear','linear',])

    elif False:
        # add_samples
        data_manager = DataManager.from_existing(dbfile_in="".join(
            i for i in title if i not in "\\/:*?<>|") + '.nc', result_dir=result_dir)
        data_manager.enrich_sample_set(total_samples=1000)

    elif True:
        data_manager = DataManager.from_existing(dbfile_in="".join(
            i for i in title if i not in "\\/:*?<>|") + '.nc', result_dir=result_dir)
        data_manager.post_process_samples(db='merged')
        data_manager.clear_locks()
        


    
def test_imports():
    #import ray
    def import_tester_fun(**kwargs):
        # The following imports must be on top of the file before import ray
        import scipy.linalg, scipy.integrate, scipy.optimize, scipy.signal
        import PyQt5.QtCore, PyQt5.QtGui
        #from ansys.mapdl.core import launch_mapdl
        #from ansys.dpf.core import Model
        from model.mechanical import Mechanical, MechanicalDummy
        import pyansys
        import seaborn
        
        
        
        return (1,)

    title = 'test_imports'
    savefolder = '/usr/scratch4/sima9999/work/modal_uq/'
    result_dir = '/usr/scratch4/sima9999/work/modal_uq/'
    #ray.init(address='auto', _redis_password='5241590000000000')
    #
    #ray.shutdown()
    if not os.path.exists(result_dir + title + '.nc') or True:
        data_manager = DataManager(title=title, working_dir='/dev/shm/womo1998/',
                                   result_dir=result_dir,
                                   overwrite=True)
        data_manager.generate_sample_inputs(names=['N',],
                                            distributions=[('uniform', (2, 4)),
                                                           ],
                                            num_samples=1)
                                            
        data_manager.evaluate_samples(func=import_tester_fun, 
                                      arg_vars={'N':'N'},
                                      ret_names=['snrs'],
                                      chwdir=True, dry_run=False)
    data_manager = DataManager.from_existing(result_dir + title + '.nc', result_dir)
    data_manager.post_process_samples()
   

def test():

    # generate input samples
    title = 'test3'
    dbfile_in = f'{title}.nc'
    result_dir = f'/usr/scratch4/sima9999/work/modal_uq/{title}/'
    # result_dir=f'/vegas/users/staff/womo1998/Projects/2019_OMA_UQ/data/{title}/'

    if False:
        data_manager = DataManager(title=title, working_dir='/dev/shm/womo1998/',
                                   result_dir=result_dir,
                                   overwrite=True)

        data_manager.generate_sample_inputs(distributions=[('uniform', (-10, 10)),
                                                           ('normal', (0, 10)),
                                                           ('exponential', (10,))],
                                            num_samples=100,
                                            names=['mrofinu', 'lamron', 'laitnenopxe'])
    elif False:
        # evaluate input samples
        data_manager = DataManager.from_existing(dbfile_in=dbfile_in, result_dir=result_dir
                                                 )
        # data_manager.post_process_samples(db='in')
        data_manager.evaluate_samples(func=test_fun, arg_vars=[('a', 'mrofinu'), ('b', 'lamron'), (
            'c', 'laitnenopxe')], ret_names=['test_res', 'res_test'], chwdir=False, factor=10)
        #
    else:
        #data_manager = DataManager.from_existing(dbfile_in=dbfile_in,result_dir = result_dir)
        #data_manager=DataManager.from_existing(dbfile_in='model_perf.nc', result_dir='/usr/scratch4/sima9999/work/modal_uq/')
        data_manager = DataManager.from_existing(
            dbfile_in='model_perf2.nc', result_dir='/usr/scratch4/sima9999/work/modal_uq/')
        with data_manager.get_database('merged') as ds:
            ds.load()
        ds = process_model_perf(ds)
        plt.figure()
        for chunksize in [500, 1000, 2000, 4000]:
            fracsel = np.logical_and(
                ds['frac_nodes'] < 0.7, ds['frac_nodes'] > 0.4)
            chunksel = ds['chunksize'] == chunksize
            force_sel = np.isnan(ds['f_scale'])
            min_ = ds.where(fracsel, drop=True).where(
                chunksel, drop=True).where(force_sel, drop=True)['min_per_ts']
            mean_ = ds.where(fracsel, drop=True).where(
                chunksel, drop=True).where(force_sel, drop=True)['mean_per_ts']
            max_ = ds.where(fracsel, drop=True).where(
                chunksel, drop=True).where(force_sel, drop=True)['max_per_ts']
            nnod = ds.where(fracsel, drop=True).where(
                chunksel, drop=True).where(force_sel, drop=True)['num_nodes']
            plt.plot(nnod.values, mean_.values, ls='none',
                     marker='.', label=f'{chunksize}')
        plt.show()
        # data_manager.clear_failed(dryrun=False)
        # data_manager.clear_locks()
        # data_manager.clear_wdirs(delnonempty=False)
        #data_manager.post_process_samples(db='merged', func=process_model_perf)
        # data_manager.clear_locks()
        
if __name__ == '__main__':
    test_imports()
    #test_categorical()