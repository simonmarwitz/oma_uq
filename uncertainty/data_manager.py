
import os
import glob
import shutil
import sys
import time
from datetime import date
import simpleflock
import coloredlogs, logging
coloredlogs.install()
#global logger
#LOG = logging.getLogger('')
logging.basicConfig(level=logging.INFO)

from contextlib import contextmanager

import uuid

import numpy as np

import matplotlib.pyplot as plot

import xarray as xr
import scipy.io as sio


global pid
pid=str(os.getpid())


import ray

'''
PATH=$PATH:/ismhome/staff/womo1998/.local/bin/
export PATH

module load python/intelpython3.7 > /dev/zero  2>&1
export OMP_NUM_THREADS=1 # limit MKL to the number of workers / cpus to avoid hyperthreading
PYTHONPATH=/vegas/users/staff/womo1998/Projects/2019_OMA_UQ/code/
export PYTHONPATH
export PYTHONUNBUFFERED=1

ray start --head --dashboard-host 141.54.148.100 --dashboard-port=5998 --num-cpus 0

# run as many times as workers are needed, each adds 16 workers
srun --nodes=1 --exclusive ray start --address='141.54.148.100:6379' --redis-password='5241590000000000' --num-cpus=16 --block &

# connect to dashboard at  http://141.54.148.100:5998/

# encountered errors
# workers fail due to slurm killing them
# workers fail, if too many different jobs are submitted, i.e. this script is run too many times

'''






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
        
        #simpleflock sometimes gives lock to two processes
        with simpleflock.SimpleFlock(f"{self._path}.lock"):
            while True:
                lockfile_list = glob.glob(f'{self._path}.*.lock')
                #print(lockfile_list)
                if len(lockfile_list)>0:
                    if len(lockfile_list) == 1 and lockfile_list[0] == self._this_lockfile:
                        # this processes lockfile is the only one, we can continue to modify the ds safely
                        logging.debug(f'Acquired lock on {self._path}.lock')
                        return
                    elif self._this_lockfile in lockfile_list:
                        # another process has created a lockfile meanwhile -> start over
                        os.remove(self._this_lockfile)
                        time.sleep(np.random.random())
                    else:
                        # another process currently holds the lock for this file
                        logging.warning('Wating for lockfile to release: {}'.format(lockfile_list))
                        time.sleep(np.random.random())
                else:
                    # if no other lockfile exists -> create one
                    # continue in while loop to check for race conditions with othe processes       
                    _fd=open(self._this_lockfile, 'w+')
                    _fd.close()
    def __exit__(self, *args):
        
        os.remove(self._this_lockfile)
        
class DataManager(object):
    def __init__(self, title, dbfile_in=None, dbfile_out=None, result_dir=None, working_dir=None, overwrite=False):
        '''
        initializes the object and checks all the provided directories and filenames
        '''
        assert isinstance(title, str)
        self.title = title
        
        
        if result_dir is None:
            logging.debug('no result_dir specified, using /vegas/scratch/womo1998/modal_uq/')
            result_dir = '/vegas/scratch/womo1998/modal_uq/'
                    
        if not os.path.isdir(result_dir):
            logging.debug(f'creating directory(s) {result_dir}')
            os.makedirs(result_dir, exist_ok=True)
            
        self.result_dir = result_dir
        
        if working_dir is None:
            working_dir = os.getcwd()
            
        if not os.path.isdir(working_dir):
            logging.debug(f'creating directory(s) {working_dir}')
            os.makedirs(working_dir, exist_ok=True)
        
        self.working_dir = working_dir
        
        if dbfile_in is None:
            # use the title, but clean it up a bit
            dbfile_in = "".join(i for i in title if i not in "\/:*?<>|")+'.nc'
            
        self.dbfile_in = dbfile_in
        if os.path.exists(os.path.join(result_dir, dbfile_in)):
            if overwrite:
                logging.warning(f'Input database file {os.path.join(result_dir, dbfile_in)} already exists and will be overwritten')
                os.remove(os.path.join(result_dir, dbfile_in))
        
        
        if dbfile_out is None:
            file,ext = os.path.splitext(dbfile_in)
            dbfile_out = file+'_out'+ext
        
        self.dbfile_out = dbfile_out
        if os.path.exists(os.path.join(result_dir, dbfile_out)):
            if overwrite:
                logging.warning(f'Output database file {os.path.join(result_dir, dbfile_out)} already exists and will be overwritten')
                os.remove(os.path.join(result_dir, dbfile_out))

        
        if not os.path.exists(os.path.join(result_dir, dbfile_in)):
            #initialize database file
            with self.get_database(database='in', rw=True) as ds:
                pass
        
        if not os.path.exists(os.path.join(result_dir, dbfile_out)):
            #initialize database file
            with self.get_database(database='out', rw=True) as ds:
                pass
        
        
    def generate_sample_inputs(self, distributions, num_samples, names=None):
        '''
        generates num_distributions x num_samples samples following each distribution in distributions
        names may be provided for better mapping of model inputs and distributions
        results will be saved in a separate xarray database
        
        '''
        num_variables = len(distributions)
        if names is None:
            logging.debug('generating variable names')
            names = [f'var_{i:02d}' for i in range(num_variables)]
        assert len(names) == num_variables
        
        for dist_type, dist_params  in distributions:
            #assert dist_type in np.random.__all__
            print(dist_params)
            assert isinstance(dist_params, (tuple,list))
            
        
        with self.get_database(database='in', rw=True) as ds:
            
            # seed the RNG, to get the same results, when rerunning
            seeds = np.random.SeedSequence(ds.attrs['entropy']).spawn(len(ds.data_vars)+num_variables)[-num_variables:] # skip over seeds for previously generated samples
            
            if not 'ids' in ds.coords:
                # we could seed that too, but I think, that might not be necessary
                ids = [str(uuid.uuid4()).split('-')[-1] for i in range(num_samples)]
                #num_params= 3
                ds.coords['ids'] = ids
            else:
                assert len(ds.coords['ids'])==num_samples
                ids = ds.coords['ids']
                
            for seed, name, (dist_type, dist_params) in zip(seeds, names, distributions):
                rng = np.random.default_rng(seed)
                values = getattr(rng, dist_type)(*dist_params, size=(num_samples,))
                ds[name]=('ids',values, {'dist_type':dist_type, 'dist_params':dist_params})

        # populate output db with the ids at least
        with self.get_database(database='out', rw=True) as ds:
            ds.coords['ids'] = ids
            ds['_runtimes'] = (['ids'],np.full(shape=(num_samples,), fill_value=np.nan))
            
    def provide_sample_inputs(self, arrays, names=None, attrs=None):
        '''
        generates num_distributions x num_samples samples following each distribution in distributions
        names may be provided for better mapping of model inputs and distributions
        results will be saved in a separate xarray database
        
        '''
        num_variables = len(arrays)
        if names is None:
            logging.debug('generating variable names')
            names = [f'var_{i:02d}' for i in range(num_variables)]
        assert len(names) == num_variables
                    
        num_samples = arrays[0].size
        for array in arrays:
            assert array.size == num_samples
        
        with self.get_database(database='in', rw=True) as ds:
            
            if not 'ids' in ds.coords:
                if "jid" in names:
                    ind = names.index("jid")
                    ids=arrays[ind]
                    del names[ind]
                    del arrays[ind]
                elif "ids" in names:
                    ind = names.index("ids")
                    ids=arrays[ind]
                    del names[ind]
                    del arrays[ind]
                else:
                    ids = [str(uuid.uuid4()).split('-')[-1] for i in range(num_samples)]
                ds.coords['ids'] = ids
            else:
                assert len(ds.coords['ids'])==num_samples
                assert "jid" not in names
                assert "ids" not in names
                ids=ds.coords['ids']
                
            #ds.coords['names'] = names
            #ds['samples'] = (['names','ids'],np.full(shape=(num_variables, num_samples), fill_value=np.nan))
            #ds.coords['dists_params']=['dist_type','p1','p2','p3','p4','p5']
            for values,name in zip(arrays,names):
                
                ds[name]=('ids',values, {})
                #ds['samples'][ds.names==name,:] = values
                
        # populate output db with the ids at least
        with self.get_database(database='out', rw=True) as ds:
            ds.coords['ids'] = ids
            ds['_runtimes'] = (['ids'],np.full(shape=(num_samples,), fill_value=np.nan))
    
    def enrich_sample_set(self, total_samples):
        
        logging.warn("Enrichment currently untested.")
        with self.get_database(database='in', rw=False) as ds:
            
            new_ds = xr.Dataset()
            num_variables = len(ds.data_vars)
            num_existing_samples = len(ds.coords['ids'])
            num_samples = total_samples - num_existing_samples 
            
            # seed the RNG, to get the same results, when rerunning
            seeds = np.random.SeedSequence(ds.attrs['entropy']).spawn(num_variables)
            
            
            ids = [str(uuid.uuid4()).split('-')[-1] for i in range(num_existing_samples,total_samples)] 
                #num_params= 3
            new_ds.coords['ids'] = ids


#             for seed, name, (dist_type, dist_params) in zip(seeds, names, distributions):
                
            for seed, name in zip(seeds, ds.data_vars):
                da = ds[name]
                dist_type=da.attrs['dist_type']
                dist_params = da.attrs['dist_params']
                
                
                rng = np.random.default_rng(seed)
                # theoretically, if all variables are in the same order, etc.  this should generate the first num_existing_samples to be equal to the existing
                values = getattr(rng, dist_type)(*dist_params, size=(total_samples,)) 
                new_ds[name]=('ids',values[num_existing_samples:])
            
            ds.combine_first(new_ds)
            
        
        # populate output db with the ids at least
        with self.get_database(database='out', rw=False) as ds:
            new_ds = xr.Dataset()
            new_ds.coords['ids'] = ids
            new_ds['_runtimes'] = (['ids'],np.full(shape=(num_samples,), fill_value=np.nan))
            ds.combine_first(new_ds)
            
    def evaluate_samples(self, func, arg_vars, ret_names=None, chwdir=True, re_eval_sample=None, dry_run=False,**kwargs):
        '''
        
        func is a function that 
            takes jid, result_dir and working_dir, arguments
            takes arguments as named in arg_vars with values from the input db
            returns values as named and in the order of ret_names
            may take additional kwargs
        
        arg_vars  is a list of mappings (function argument, dataarray name in input dataset)
        
        if chwdir:
            working directory will be changed for each sample to self.working_dir+jid
            and cleaned up upon successful completion ? or on every completion
        else:
            working_directory will be changed to self.working_dir once
        
        this could be automated with dask -> workflow:
            specify your inputs
            specify your function and expected outputs
            assign function to input
            
            setup dask to use slurm as a job scheduler
            
            start a number of workers and assign the previously generated workloads
            results should be readily available in the dask arrays
            question is: how often will I really have such an automated workflow? 
            That is probably better for production not for development
        alternatively manual sbatch could be uses:
            advantage: 
                don't have to learn dask
                more control and manual inspection possible
            disadvantage: 
                manual configuration and file locking needed
                hard to debug function errors
                won't learn dask
        '''
        if not dry_run:
            ray.init(address='auto', _redis_password='5241590000000000')
        
        @ray.remote
        def setup_eval(func, jid, fun_kwargs, result_dir=self.result_dir, working_dir=None, **kwargs):
           
            # lock files will stay there, make sure to delete them afterwards
            with simpleflock.SimpleFlock(os.path.join(self.result_dir,f'{jid}.lock'), timeout=1):
                logging.info(f'start computing sample {jid}')
                 
                now = time.time()
                # create the working directory
                
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir, exist_ok=True)
                    
                if working_dir is not None:
                    cwd=os.getcwd()
                    if not os.path.exists(working_dir):
                        os.makedirs(working_dir, exist_ok=True)
                    os.chdir(working_dir)
                 
                 
                # call model evaluation
                # save results in filesystem -> should be done in func, also with a shortcut in func to return, what has been done previously
                try:
                    ret_vals = func(jid=jid, result_dir=result_dir, working_dir=working_dir, **fun_kwargs, **kwargs)
                except Exception as e:
                    logging.warning(f'function failed with exit message {e}')
                    ret_vals=repr(e)
                finally:
                    if working_dir is not None:
                        os.chdir(cwd)
                        subdirs= next(os.walk(working_dir))[1]
                        if len(subdirs)==0:
                            logging.info(f"Removing working_dir {working_dir}")
                            shutil.rmtree(working_dir, ignore_errors=True)
                        else:
                            logging.warning(f"Cannot remove working_dir {working_dir} (contains subdirectories)")
                        
                    runtime = time.time() - now
                     
                    logging.info(f'done computing sample {jid}. Runtime was {runtime} s')
                
                return jid, ret_vals, runtime
            

        # open database read-only, without locking
        with self.get_database(database='in') as in_ds, self.get_database(database='out',rw=True) as out_ds:
            
            if not chwdir:
                os.chdir(self.working_dir)
                logging.debug(f'current working directory {self.working_dir}')
            
            if re_eval_sample is not None:
                in_ds = in_ds.sel(ids =[re_eval_sample])
            
            num_samples = in_ds.ids.size
            
            futures=[]
            
            # TODO: Improvement: Estimate job size and submit big jobs first, 
            # i.e. sort key = jobsize, but also add some smallest jobs 
            # to the beginning to see any errors quickly
            
            for jid_ind in sorted(range(num_samples), key=lambda _: np.random.random()):
                
                if (not out_ds['_runtimes'][jid_ind].isnull()) and re_eval_sample is None: continue
                
                jid=in_ds['ids'][jid_ind].item()

                fun_kwargs = {}
                for arg,var in arg_vars.items():
                    fun_kwargs[arg]=in_ds[var][jid_ind].item()
                
                if chwdir:
                    working_dir = self.working_dir+jid
                else:
                    working_dir = None
                
                if dry_run:
#                     if jid=='5db7e4ed1b09':
#                         if not os.path.exists(working_dir):
#                             os.makedirs(working_dir, exist_ok=True)
#                         func(jid, **fun_kwargs, working_dir=working_dir, result_dir=working_dir, **kwargs)
#                         return
                    continue
                
                worker_ref = setup_eval.remote(func, jid, fun_kwargs, working_dir=working_dir, **kwargs)
                
                futures.append(worker_ref)
                
        if dry_run:
            return
        
        futures=set(futures)
        
        while True:
            ready, wait = ray.wait(list(futures), num_returns=min(len(futures),25))
            
            ret_sets = ray.get(ready)
            
            
            if not ret_sets:
                logging.info("All jobs already computed!")
                break
    
            with self.get_database(database='out',rw=True) as out_ds:
                
                for jid,ret_vals, runtime in ret_sets: # first may have thrown an exception, then len() fails
                    if isinstance(ret_vals, (list,tuple)):
                        num_variables =len(ret_vals)
                        break
                else:
                    raise RuntimeError(f"All ret_sets are empty {ret_sets}")
                 
                if ret_names is None:
                    logging.debug('generating variable names')
                    ret_names = [f'var_{i:02d}' for i in range(num_variables)]
                
                for jid,ret_vals, runtime in ret_sets:
                    
                    if isinstance(ret_vals, str): # exception repr
                        if '_exceptions' not in out_ds.data_vars:
                            out_ds['_exceptions'] = (['ids'],np.full(shape=(num_samples,), fill_value=''))
                        out_ds['_exceptions'][out_ds.ids==jid] = ret_vals
                        continue
                        
                    assert len(ret_names)==len(ret_vals)
                    
                    for name, value in zip(ret_names, ret_vals):
                        if name not in out_ds.data_vars:
                            out_ds[name]=('ids',np.full(shape=(num_samples), fill_value=np.nan))
                        out_ds[name][out_ds.ids==jid] = value
                         
                        logging.debug(out_ds[name][out_ds.ids==jid])
                    
                    out_ds['_runtimes'][out_ds.ids==jid] = runtime
                    
                    logging.debug(out_ds['_runtimes'][out_ds.ids==jid]) 
            
            if len(wait) == 0:
                break
            
            size_before=len(futures)
            futures.difference_update(ready)
            logging.info(f"Finished {len(ready)} samples. Remaining {len(futures)} samples. (before {size_before})")
            
        return
        
    def post_process_samples(self, db='merged', func=None, **kwargs):
        '''
        drawing scatterplot matrices, bar charts, box plots etc.
        '''
        def scatterplot_matrix(data, names, **kwargs):
            import itertools
            """Plots a scatterplot matrix of subplots.  Each row of "data" is plotted
            against other rows, resulting in a nrows by nrows grid of subplots with the
            diagonal subplots labeled with "names".  Additional keyword arguments are
            passed on to matplotlib's "plot" command. Returns the matplotlib figure
            object containg the subplot grid."""
            data = np.array(data)
            numvars, numdata = data.shape
            fig, axes = plot.subplots(nrows=numvars, ncols=numvars, figsize=(12,12))
            fig.subplots_adjust(hspace=0.05, wspace=0.05)
        
            for ax in axes.flat:
                # Hide all ticks and labels
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
        
                # Set up ticks only on one side for the "edge" subplots...
                if ax.is_first_col():
                    ax.yaxis.set_ticks_position('left')
                if ax.is_last_col():
                    ax.yaxis.set_ticks_position('right')
                if ax.is_first_row():
                    ax.xaxis.set_ticks_position('top')
                if ax.is_last_row():
                    ax.xaxis.set_ticks_position('bottom')
        
            # Plot the data.
            for i, j in zip(*np.triu_indices_from(axes, k=1)):
                for x, y in [(i,j), (j,i)]:
                    axes[y,x].plot(data[x], data[y], ls='none', marker='.',**kwargs)
            for ij in range(numvars):
                #print(names[ij])
                #print(data[ij])
                if not np.issubdtype(data[ij].dtype, np.number):continue
                if np.isnan(data[ij]).all(): continue
                if np.isinf(data[ij]).all(): continue
                axes[ij,ij].hist(data[ij], bins=20)
        
            # Label the diagonal subplots...
            for i, label in enumerate(names):
                axes[i,i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                        ha='center', va='center')
        
            # Turn on the proper x or y axes ticks.
            for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
                axes[j,i].xaxis.set_visible(True)
                axes[i,j].yaxis.set_visible(True)
        
            return fig
        
        with self.get_database(database=db,rw=False) as ds:
             
            if func is not None:
                ds = func(ds,**kwargs)
#             
            #scatterplot_matrix(np.array(data), names)
            
            scatterplot_matrix([ds[name].data for name in ds.data_vars], [name for name in ds.data_vars])
            plot.show()
    
    @classmethod
    def from_existing(cls, dbfile_in,result_dir='/vegas/scratch/womo1998/modal_uq/'):
        assert os.path.exists(os.path.join(result_dir,dbfile_in))
        
        cls.result_dir = result_dir
        cls.dbfile_in = dbfile_in
        with cls.get_database(cls) as ds:
            working_dir = ds.attrs['working_dir']
            assert os.path.exists(working_dir)
            cls.working_dir = working_dir
            
            title = ds.attrs['title']
            cls.title = title
            
            result_dir_ = ds.attrs['result_dir']
            if result_dir_!= result_dir:
                logging.warning(f'result dir from db {result_dir_} differs from given result_dir {result_dir}')
            
            dbfile_in_ = ds.attrs['dbfile_in']
            if dbfile_in != dbfile_in_:
                logging.warning(f'dbfile_in from db {dbfile_in_} differs from given dbfile_in {dbfile_in}')
            
                
            dbfile_out = ds.attrs['dbfile_out']
            cls.dbfile_out = dbfile_out
            #create and  populate output db with the ids at least
            if not os.path.exists(os.path.join(result_dir,dbfile_out)):
                logging.info(f'output db {os.path.join(result_dir,dbfile_out)} has been deleted. Recreating!')
                with cls.get_database(cls, database='out', rw=True) as out_ds:
                    ids = ds.ids.data
                    out_ds.coords['ids'] = ids
                    out_ds['_runtimes'] = (['ids'],np.full(shape=(len(ids),), fill_value=np.nan))
        
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
        
        assert database in ['in','out','merged']
        
        if database == 'in':
            dbfile = self.dbfile_in
        elif database == 'out':
            dbfile = self.dbfile_out
            if not rw:
                logging.warning('You have chosen to open the output-db in read-only mode, changes will not be saved.')
        elif database == 'merged':
            if rw: logging.warning('merged database can not be opend in rw mode')
            try:
                with self.get_database(database='in',rw=False) as in_ds, self.get_database(database='out',rw=False) as out_ds:
                    ds = out_ds.combine_first(in_ds)
                    yield ds
            except:
                raise
            finally:
                ds.close()
            return
                
        dbpath = os.path.join(self.result_dir, dbfile)
        
        if not os.path.exists(dbpath):
            logging.debug(f'file {dbfile} will be created')
            with MultiLock(dbpath):
                # create database, if it does not exist
                if not os.path.exists(dbpath):
                    ds = xr.Dataset()
                    ds.attrs['working_dir']=self.working_dir
                    ds.attrs['result_dir']=self.result_dir
                    ds.attrs['dbfile_in']=self.dbfile_in
                    ds.attrs['dbfile_out']=self.dbfile_out
                    ds.attrs['date']=date.today().__repr__()
                    ds.attrs['title']=self.title
                    ds.attrs['entropy']=np.random.randint(np.iinfo(np.int32).max)
                    ds.to_netcdf(dbpath, engine='h5netcdf')
                    #ds.to_netcdf(dbpath, format='netcdf4')
                    ds.close()
        else:
            logging.debug(f'opening existing file {dbfile}')
            
        if not rw:
            try:
                # open database
                ds = xr.open_dataset(dbpath, engine='h5netcdf')
                #ds = xr.open_dataset(dbpath)
                ds.load()
                ds.close()
                # yield database
                yield ds
            except:
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
                    #ds.close()
                    # yield database
                    yield ds
                except:
                    raise
                finally:
                    # save and close database
                    ds.close()
                    
                    ds.to_netcdf(dbpath, engine='h5netcdf')
                    #ds.to_netcdf(dbpath, format='netcdf4')
                    ds.close()
            # remove lock automatically 
            
        return
    
    def clear_locks(self):
        lock_list = glob.glob(os.path.join(self.result_dir,'*.lock'))
        with self.get_database(database='out', rw=False) as out_ds:
            for file in lock_list:
                jid = os.path.splitext(os.path.split(file)[-1])[0]
                if np.isnan(out_ds['_runtimes'][out_ds.ids==jid]):
                    logging.warning(f'A lock exists for {jid} but no results are in the db')
                logging.debug(f'removing file {file}')
                os.remove(file)
                
    
    def clear_wdirs(self, delnonempty=False):
        dir_list = glob.glob(os.path.join(self.result_dir,'*/'))
        with self.get_database(database='out', rw=False) as out_ds:
            for dir_ in dir_list:
                jid = os.path.split(os.path.dirname(dir_))[-1]
                if not jid in out_ds.ids: continue
                if np.isnan(out_ds['_runtimes'][out_ds.ids==jid]):
                    logging.warning(f'A lock exists for {jid} but no results are in the db')
                logging.debug(f'removing file {dir_}')
                try:
                    os.removedirs(dir_)
                except Exception as e:
                    if e.errno == 39:
                        logging.warning(f'Working directory not empty {dir_}')
                        if delnonempty:
                            shutil.rmtree(dir_)
                    else:
                        logging.warning(repr(e))
    
    def clear_failed(self, dryrun=True):
        lock_list = glob.glob(os.path.join(self.result_dir,'*.lock'))
        with self.get_database(database='out', rw=False) as out_ds:
            for file in lock_list:
                jid = os.path.splitext(os.path.split(file)[-1])[0]
                if '_exceptions' in out_ds.data_vars:
                    if out_ds['_exceptions'][out_ds.ids==jid].data:
                        print(out_ds['_exceptions'][out_ds.ids==jid].data, file)
                        if not dryrun: os.remove(file)
                if np.isnan(out_ds['_runtimes'][out_ds.ids==jid]):
                    print(out_ds['_runtimes'][out_ds.ids==jid], file)
                    if not dryrun: os.remove(file)
                    
    def as_text(self,fname, var_names = None):
        print(fname)

        with self.get_database(database='merged', rw=False) as ds:

            if var_names is None:
                var_names=list(ds.data_vars)+list(ds.coords)
            #var_arrays=[]
            for var in var_names:
                assert var in ds.data_vars or var in ds.coords
                #var_arrays.append(ds[var].data)
            ds[var_names].to_dataframe().to_csv(self.result_dir+fname, '\t')
            



def test_fun(a,b,c, result_dir=None,factor=1, **kwargs):
    shutil.rmtree(result_dir)
    time.sleep(0.1)
    return a+b+c*factor, a*b/factor

def process_model_perf(ds):
    
    for name in ['mean','min','max','std']:
        ds[f'{name}_per_nnod']=ds[name]/ds['num_nodes']
    for name in ['mean','min','max','std']:
        ds[f'{name}_per_ts']=ds[name]/ds['chunksize']
    
    ds['frac_nodes']=ds['num_meas_nodes']/ds['num_nodes']
    del_names = [#'d0','f_scale',
                 'mean','min','max','std']
    
    for name in ds.data_vars:
        if name.startswith('_'):
            del_names.append(name)
    
    for name in del_names:
        del ds[name]
            
    #print(ds)
    
    return ds

def student_manager(ambient, nonlinear, friction):
    if ambient and nonlinear and not friction:
        title="nonlinear_ambient"
    elif ambient and not nonlinear and not friction:
        title="linear_ambient"
    elif not ambient and not nonlinear and not friction:
        title="linear_decay"
    elif not ambient and nonlinear and not friction:
        title="nonlinear_decay"
    elif not ambient and not nonlinear and friction:
        title="friction_decay"
    elif ambient and not nonlinear and friction:
        title="friction_ambient"
    elif not ambient and nonlinear and friction:
        title="general_decay"
    elif ambient and nonlinear and friction:
        title="general_ambient"
    else:
        raise RuntimeError(f'This combination of inputs is not supported: {ambient}, {nonlinear}, {friction}')
    print(title)
    savefolder = '/vegas/scratch/womo1998/data_hadidi/datasets_'+title+'/'
    result_dir='/vegas/scratch/womo1998/data_hadidi2/datasets_'+title+'/'
    
    
    if not os.path.exists(result_dir+title+'.nc') or False:
        data_manager = DataManager(title=title, working_dir='/dev/shm/womo1998/',
                                   result_dir = result_dir,
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
                                            distributions=[('uniform',(1,15)),
                                                           ('uniform',(0.1,10)),
                                                           ('uniform',(0.001,0.015)),
                                                           ('uniform',(-20,20)),
                                                           ], 
                                            num_samples=1000-722)        
        
        if nonlinear:
            data_manager.generate_sample_inputs(names=[ 'nl_ity',],
                                                distributions=[('uniform',(-0.5,0.5)),], 
                                                num_samples=1000-722)
        else:
            data_manager.generate_sample_inputs(names=[ 'nl_ity',],
                                                distributions=[('uniform',(0,0)),], 
                                                num_samples=1000-722)
        if friction:
            data_manager.generate_sample_inputs(names=[ 'fric_visc_rat',],
                                            distributions=[('uniform',(0,1)),], 
                                            num_samples=1000-722)
        else:
            data_manager.generate_sample_inputs(names=[ 'fric_visc_rat',],
                                            distributions=[('uniform',(0,0)),], 
                                            num_samples=1000-722)
        if ambient:
            data_manager.provide_sample_inputs([np.array([None for _ in range(1000-722)])], 
                                           ['d0'])
            data_manager.generate_sample_inputs(names=['num_cycles',
                                                   'f_scale',],
                                            distributions=[('integers',(300,2000)),
                                                           ('uniform',(0,10)),
                                                           ],
                                            num_samples=1000-722)
        else:
            data_manager.generate_sample_inputs(names=['num_cycles',
                                                        'd0',],
                                            distributions=[('integers',(3,20)),
                                                           ('uniform',(1e-3,100)),], 
                                            num_samples=1000-722)
            data_manager.provide_sample_inputs([np.array([None for _ in range(1000-722)])], 
                                           ['f_scale'])
        

        data_manager.post_process_samples(db='in')
    
    elif False:
        #add_samples
        data_manager = DataManager.from_existing(dbfile_in="".join(i for i in title if i not in "\/:*?<>|")+'.nc',result_dir = result_dir)
        data_manager.enrich_sample_set(total_samples=1000)
        
    
        
    elif True:
        
        
        from model import mechanical
        #evaluate input samples
        data_manager = DataManager.from_existing(dbfile_in="".join(i for i in title if i not in "\/:*?<>|")+'.nc',result_dir = result_dir)
        

        
        #data_manager.evaluate_samples(func=manipulate_student_fun, arg_vars={'snr_db':'snr'}, ret_names=['signal_power', 'noise_power','deltat'], chwdir = True, readfolder=savefolder)
        data_manager.evaluate_samples(func=mechanical.student_data_part2, arg_vars={'omega':'omega', 
                                                                       'zeta':'zeta', 
                                                                       'dt_fact':'dt_fact',
                                                                       'num_cycles':'num_cycles',
                                                                       'f_scale':'f_scale', 
                                                                       'd_scale':'d0', 
                                                                       'nl_ity':'nl_ity', 
                                                                       'fric_visc_rat':'fric_visc_rat', 
                                                                       'snr_db':'snr'}, 
                                                                       ret_names=['k','c','d_max','fsl','signal_power', 'noise_power','deltat'], 
                                                                       chwdir = True, dry_run=True)
        #mechanical.student_data_part2(jid, result_dir, omega, zeta, dt_fact,num_cycles, f_scale, d_scale, nl_ity, fric_visc_rat, snr_db, **kwargs)
        #
    elif True:
        data_manager = DataManager.from_existing(dbfile_in="".join(i for i in title if i not in "\/:*?<>|")+'.nc',result_dir = result_dir)
        data_manager.post_process_samples(db='in')
        data_manager.post_process_samples(db='out')
        data_manager.post_process_samples(db='merged')
        data_manager.clear_locks()
        #data_manager.clear_wdirs()
        data_manager.as_text('test.txt',['ids','k','c','d_max','fsl','fric_visc_rat','nl_ity','noise_power','signal_power','deltat'])



def manipulate_student_fun(jid, snr_db, readfolder, result_dir):
    
    print(jid)
#     rn=np.random.rand(1)
#     print(rn)
#     if rn>0.9: raise RuntimeError
    
    do_plot=False
    
    snr=10**(snr_db/10)
    
    file=readfolder+jid+'.csv'
    array=np.loadtxt(file)
    
    
    power=np.mean(array[:,1]**2)
    if do_plot:
        fig,axes=plot.subplots(3,1, sharex=True)
        axes[0].psd(array[:,1],Fs=array.shape[0]/(array[-1,0]-array[0,0]))
    #decimate
    array=array[1::6,:]
    N=array.shape[0]
    # add noise
    noise_power = power/snr
    noise=np.random.normal(0,np.sqrt(noise_power),N)
    power_noise=np.mean(noise**2)
    
    snr_actual=power/power_noise
    snr_actual_db= 10*np.log10(snr_actual)
    
    array[:,1]+=noise
    if 'ambient' in result_dir:
        cov = np.zeros((N,2))
        for i in range(N):
            cov[i,1]=array[:N-i,1].dot(array[i:,1])/(N-i)
        cov[:,0]=array[:,0]
        if do_plot:
            axes[1].psd(array[:,1],Fs=array.shape[0]/(array[-1,0]-array[0,0]))
            axes[2].plot(cov[:,0],cov[:,1])
            axes[1].set_xlim((0,array.shape[0]/(array[-1,0]-array[0,0])/2))
            plot.show()
        
        mdict={'disp':array, 'corr':cov}
    else:
        mdict={'disp':array}
    file=os.path.join(result_dir,jid+'.mat')
    sio.savemat(file, mdict, appendmat=True, format='5', long_field_names=False, do_compression=True, oned_as='row')

    return power, power_noise, (array[-1,0]-array[0,0])/(array.shape[0]-1)
    
    
if __name__ == '__main__':
    
    if len(sys.argv)>1:
        num=int(sys.argv[1])
    else:
        num=4
    #for num in range(8):
    # 0x linear_decay, 1x friction_decay, 2x nonlinear_decay, 3x general_decay, 4x linear_ambient, 5x friction_ambient, 6x nonlinear_ambient, 7x general_ambient


    
    student_manager(num//4, num//2%2, num%2)
    
    
    
    
def test():
    
    # generate input samples
    title='test3'
    dbfile_in = f'{title}.nc'
    result_dir=f'/vegas/scratch/womo1998/modal_uq/{title}/'
    #result_dir=f'/vegas/users/staff/womo1998/Projects/2019_OMA_UQ/data/{title}/'
                                   
    if False:
        data_manager = DataManager(title=title, working_dir='/dev/shm/womo1998/',
                                   result_dir = result_dir,
                                   overwrite=True)
    
        data_manager.generate_sample_inputs(distributions=[('uniform',(-10,10)),
                                                           ('normal',(0,10)),
                                                           ('exponential',(10,))], 
                                                           num_samples=100, 
                                                           names=['mrofinu','lamron','laitnenopxe'])
    elif False:
        #evaluate input samples
        data_manager = DataManager.from_existing(dbfile_in=dbfile_in,result_dir = result_dir
                                                 )
        #data_manager.post_process_samples(db='in')
        data_manager.evaluate_samples(func=test_fun, arg_vars=[('a','mrofinu'),('b','lamron'),('c','laitnenopxe')], ret_names=['test_res','res_test'], chwdir = False, factor=10)
        #
    else:
        #data_manager = DataManager.from_existing(dbfile_in=dbfile_in,result_dir = result_dir)
        #data_manager=DataManager.from_existing(dbfile_in='model_perf.nc', result_dir='/vegas/scratch/womo1998/modal_uq/')
        data_manager=DataManager.from_existing(dbfile_in='model_perf2.nc', result_dir='/vegas/scratch/womo1998/modal_uq/')
        with data_manager.get_database('merged') as ds:
            ds.load()
        ds = process_model_perf(ds)
        plot.figure()
        for chunksize in [500,1000,2000,4000]:
            fracsel=np.logical_and(ds['frac_nodes'] <0.7, ds['frac_nodes'] >0.4)
            chunksel=ds['chunksize']==chunksize
            force_sel=np.isnan(ds['f_scale'])
            min_ =ds.where(fracsel, drop=True).where(chunksel, drop=True).where(force_sel,drop=True)['min_per_ts']
            mean_=ds.where(fracsel, drop=True).where(chunksel, drop=True).where(force_sel,drop=True)['mean_per_ts']
            max_ =ds.where(fracsel, drop=True).where(chunksel, drop=True).where(force_sel,drop=True)['max_per_ts']
            nnod =ds.where(fracsel, drop=True).where(chunksel, drop=True).where(force_sel,drop=True)['num_nodes']
            plot.plot(nnod.values,mean_.values, ls='none',marker='.', label=f'{chunksize}')
        plot.show()
        #data_manager.clear_failed(dryrun=False)
        #data_manager.clear_locks()
        #data_manager.clear_wdirs(delnonempty=False)
        #data_manager.post_process_samples(db='merged', func=process_model_perf)
        #data_manager.clear_locks()