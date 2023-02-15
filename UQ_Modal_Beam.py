# import sys
import os
# sys.path.append("/usr/wrk/people9/sima9999/code/")
# sys.path.append("/vegas/users/staff/womo1998/Projects/2019_OMA_UQ/code/")
import numpy as np
import zlib, zipfile
from model.mechanical import Mechanical, MechanicalDummy
from uncertainty.polymorphic_uncertainty import MassFunction,RandomVariable,PolyUQ
import logging
import psutil
import struct
import time

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.WARNING)

logger_mech=logging.getLogger('model.mechanical')
logger_mech.setLevel(level=logging.WARNING)

global ansys

def default_mapping(E=2.1e11, a=0.9, b=0.875, t=0.009, rho=7850, N_wire=1.1e+05, 
                     A_wire=0.00075, add_mass=55, zeta=0.0428, dD=198, # structural
                     ice_occ=1, ice_mass=75, # environmental
                     num_nodes=200, num_modes=14, fs=10, N=2048, meas_locs=np.array([40,80,120,160,200]), # algorithmic
                     jid='abcdef123', result_dir=None, working_dir='/dev/shm/womo1998/', skip_existing=False):
    
    return mapping_function(E, a, b, t, rho, N_wire, A_wire, add_mass, zeta, dD, ice_occ, ice_mass, num_nodes, num_modes, fs, N, meas_locs, jid, result_dir, working_dir, skip_existing)

def mapping_pass(b,t,N_wire,A_wire,add_mass,zeta,dD,ice_occ,ice_mass, jid='abcdef123', result_dir=None, working_dir='/dev/shm/womo1998/'):
    time.sleep(0.001)
    return np.array(b),np.array(t),np.array(N_wire),np.array(A_wire),np.array(add_mass),np.array(zeta),np.array(dD),np.array(ice_occ),np.array(ice_mass),np.array(jid),np.array(result_dir)

def mapping_validate(E=2.1e11, a=0.9, b=None, t=None, rho=7850, N_wire=None, 
                     A_wire=None, add_mass=None, zeta=None, dD=None, # structural
                     ice_occ=None, ice_mass=None, # environmental
                     num_nodes=200, num_modes=14, fs=10, N=2048, meas_locs=np.array([40,80,120,160,200]), # algorithmic
                     jid='abcdef123', result_dir=None, working_dir=None, skip_existing=True):
    A = np.pi*(a*b-(a-t)*(b-t))
    
    Iy = np.pi/4*(a*b**3 -(a-t)*(b-t)**3)
    Iz = np.pi/4*(b*a**3 -(b-t)*(a-t)**3)
    
    Aeq = A_wire/(1 + (A_wire*rho*9.819*70/N_wire)**2*E*A_wire/12/N_wire)

    keq = (E * Aeq * (70**2 / (70**2 + 160**2)) + N_wire)/np.sqrt(70**2 + 160**2)
    
    rho_pipe = rho + add_mass / A
     #incremental bernoulli gives [0...0, 0...1] -> [0...0, 50...100
    if ice_occ:
        ice_mass *= 50
        ice_mass += 50
        rho_pipe += ice_mass / A
    
    struct_parms = {
            'L'         : 200,

            'E'         : E,
            'A'         : A,
            'rho'       : rho_pipe,
            'Iy'        : Iy,
            'Iz'        : Iz,

            'kz_nl'     : 1.7 * keq,
            'ky_nl'     : 2 * keq,
            'x_knl'     : 160,

            'm_tmd'     : 800,
            'ky_tmd'    : 1025.48,
            'kz_tmd'    : 1025.48,
            'dy_tmd'    : dD,
            'dz_tmd'    : dD,
            'x_tmd'     : 200,
            }
    # print(struct_parms)
    if result_dir is None:
        result_dir=os.getcwd()
    savefolder = os.path.join(result_dir, jid)
    
    # load/generate  mechanical object:
    # ambient response of a n dof rod,
    if os.path.exists(os.path.join(savefolder, f'{jid}_mechanical.npz')):
        
        tim = os.stat(os.path.join(savefolder, f'{jid}_mechanical.npz')).st_ctime
        try:
            mech = MechanicalDummy.load(jid, savefolder)
            # print(mech.struct_parms, type(mech.struct_parms))
            for parm,val in struct_parms.items():
                if not np.isclose(val, mech.struct_parms[parm]):
                    print(f'{jid} \t Parameter {parm} values differ {val} != {mech.struct_parms[parm]} {time.ctime(tim)}')
                    # print(os.stat(os.path.join(savefolder, f'{jid}_mechanical.npz')).st_ctime)
                    
                    return (False,tim)
        except (EOFError, KeyError, zlib.error, zipfile.BadZipFile, OSError):
            print(f'File {os.path.join(savefolder, f"{jid}_mechanical.npz")} corrupted. Deleting.')
            #os.remove(os.path.join(savefolder, f'{jid}_mechanical.npz'))
            return (False,tim)
        return (True,tim)
    else:
        return (False,0)
    

def mapping_function(E=2.1e11, a=0.9, b=None, t=None, rho=7850, N_wire=None, 
                     A_wire=None, add_mass=None, zeta=None, dD=None, # structural
                     ice_occ=None, ice_mass=None, # environmental
                     num_nodes=200, num_modes=14, fs=10, N=2048, meas_locs=np.array([40,80,120,160,200]), # algorithmic
                     jid='abcdef123', result_dir=None, working_dir=None, skip_existing=True):
    
    # logger.warning('MEAS_NODES definition has changed to include TMD node (total 6).')
    '''
    The TMD FRF might be more interesting than the 160 m FRF
    160 m FRF was originally chosen to have some effect due to the mode shapes, 
    which should mostly be affected by wire properties,
    otherwise only frequencies and damping really play a role
    
    we would have to blow up the datamanager database and save it again
    from_datamanager should then work with space_ind=2 for the TMD dof
    which means, we could just operate on the smaller set of 8327 samples that
    are yet to be computed, while using all 13717 samples for the other two frf nodes
    without having to recompute everything again
    
    in estimate_imp we have to make sure to start at 5390 and ignore all previous
    samples (which would be nan)
    also in optimize_inc we have to make sure the first 5390 samples are ignored
    a general procedure to ignore all-nan samples would be useful anyway
    
    
    '''
    # print(E,a,b,t,rho,N_wire,A_wire,add_mass,zeta,dD,ice_occ,ice_mass)
    
    A = np.pi*(a*b-(a-t)*(b-t))
    
    Iy = np.pi/4*(a*b**3 -(a-t)*(b-t)**3)
    Iz = np.pi/4*(b*a**3 -(b-t)*(a-t)**3)
    
    Aeq = A_wire/(1 + (A_wire*rho*9.819*70/N_wire)**2*E*A_wire/12/N_wire)

    keq = (E * Aeq * (70**2 / (70**2 + 160**2)) + N_wire)/np.sqrt(70**2 + 160**2)
    
    rho_pipe = rho + add_mass / A
     #incremental bernoulli gives [0...0, 0...1] -> [0...0, 50...100
    if ice_occ:
        ice_mass *= 50
        ice_mass += 50
        rho_pipe += ice_mass / A
    
    struct_parms = {
            'L'         : 200,

            'E'         : E,
            'A'         : A,
            'rho'       : rho_pipe,
            'Iy'        : Iy,
            'Iz'        : Iz,

            'kz_nl'     : 1.7 * keq,
            'ky_nl'     : 2 * keq,
            'x_knl'     : 160,

            'm_tmd'     : 800,
            'ky_tmd'    : 1025.48,
            'kz_tmd'    : 1025.48,
            'dy_tmd'    : dD,
            'dz_tmd'    : dD,
            'x_tmd'     : 200,
            }
    # print(struct_parms)
    if result_dir is None:
        result_dir=os.getcwd()
    savefolder = os.path.join(result_dir, jid)
    
    # load/generate  mechanical object:
    # ambient response of a n dof rod,
    mech = None
    if skip_existing:
        if os.path.exists(os.path.join(savefolder, f'{jid}_mechanical.npz')):
            try:
                mech = MechanicalDummy.load(jid, savefolder)
                for parm,val in struct_parms.items():
                    if not np.isclose(val, mech.struct_parms[parm]):
                        # print(f'{jid} \t Parameter {parm} values differ {val} != {mech.struct_parms[parm]}')
                        mech = None
                        break
            except Exception as e:
                # logger.warning(f'File {os.path.join(savefolder, f"{jid}_mechanical.npz")} corrupted. Deleting.')
                print(e)
                os.remove(os.path.join(savefolder, f'{jid}_mechanical.npz'))
            
    if mech is None or len(mech.meas_nodes)<6:
        'python docs: The expression x or y first evaluates x; if x is true, its value is returned; otherwise, y is evaluated and the resulting value is returned.'
        
        global ansys
        if 'ansys' not in globals():
            ansys = Mechanical.start_ansys(working_dir=working_dir, jid=jid)
        #ansys = Mechanical.start_ansys(working_dir, jid)
        mech = Mechanical(ansys=ansys, jobname=jid, wdir=working_dir)
        
        mech.build_conti(struct_parms, Ldiv=num_nodes, damping=zeta, 
                         num_modes=num_modes, meas_locs=meas_locs)
        
        inp_node = num_nodes # should be the tip node
        _, frf_z = mech.frequency_response(N, inp_node,'uz',fmax=fs//2, out_quant='a')
        _, frf_y = mech.frequency_response(N, inp_node,'uy',fmax=fs//2, out_quant='a')
        
        # simulate unit force at 45° angle
        frf = (frf_z + frf_y) / np.sqrt(2)
        
        # hack to save the "angled frf" into the mech object
        mech.frf = frf 
        mech.save(savefolder)
        mech.ansys.finish()
        
        for open_file in psutil.Process().open_files():
            if working_dir in open_file.path and 'rst' in open_file.path:
                # print(f'closing {open_file}')
                os.close(open_file.fd)
        # else:
        #     print(f'leaving {open_file}')
        
    
    fd = mech.damped_frequencies
    zetas = mech.modal_damping
    frf = mech.frf[:,-3:]
    # print(mech.meas_nodes[-3:])
    # print(E,a,b,t,rho,N_wire,A_wire, add_mass, zeta, dD, ice_occ, ice_mass, fd[0])
    # print(fd.astype('float32'))
    
    return fd.astype('float32'), zetas.astype('float32'), np.abs(frf).astype('float32')
    
def test_interpolation(ret_name, ret_ind, N_mcs_ale):
    
    b = MassFunction('b',[(1.7/2,1.9/2),],[1,], primary=True) # meter
    tnorm = RandomVariable('norm','tnorm', [6e-3,1e-4], primary=False) # meter
    t = MassFunction('t',[(5.9e-3,6.1e-3),(tnorm,),],[0.7,0.3], primary=True) # meter
    add_mass = MassFunction('add_mass', [(20,100),(40,50)],[0.8, 0.2], primary=True) # kilogram per meter
    muNwire = MassFunction('muNwire',[(60000,),(40000,180000)],[0.75,0.25], primary=False) # Newton
    N_wire = RandomVariable('norm','N_wire', [muNwire, 2655], primary=True) # Newton
    A_wire = MassFunction('A_wire', [(0.0007,0.0008)],[1,], primary=True) # meter^2
    zeta = MassFunction('zeta', [(0.0075,0.0092), (0.0075,0.0134), (0.0016,0.015)], [0.5,0.3,0.2], primary=True) # -
    stddD = MassFunction('vardD',[(0,25),(10,15)],[0.2,0.8], primary=False) # Newton second per meter
    dD = RandomVariable('norm', 'dD',[197.61, stddD], primary=True)  # Newton second per meter
    ice_days = MassFunction('ice_days', [(28.2/365,), (1/365,77/365)],[0.3,0.7], primary=False) # days
    ice_occ = RandomVariable('bernoulli', 'ice_occ',  [ice_days], primary=True) # boolean
    ice_mass = MassFunction('ice_mass', [(0, ice_occ),],[1], primary=True, incremental=True) # 
    
    vars_epi = [b, t, add_mass, muNwire, A_wire, zeta, stddD, ice_mass, ice_days]
    vars_ale = [tnorm, N_wire, dD, ice_occ]
    
    dim_ex = 'cartesian'
    
    result_dir = '/usr/scratch4/sima9999/work/modal_uq/uq_modal_beam/'
    
    poly_uq = PolyUQ(vars_ale, vars_epi, dim_ex=dim_ex)
    
    ret_dir = f'{ret_name}-{".".join(str(e) for e in ret_ind.values())}'
    
    poly_uq.load_state(os.path.join(result_dir, 'estimations', f'{ret_dir}/polyuq_prop.npz'))
    
    poly_uq.N_mcs_ale=N_mcs_ale
    poly_uq.estimate_imp(
        interp_fun='rbf',
        opt_meth='genetic',
        plot_res=False,
        plot_intp=False,
        intp_err_warn = 20,
        extrp_warn = 10,
        start_ale = poly_uq.N_mcs_ale - 10,
        kernel='gaussian',
        epsilon={'frf':4,'zetas':2,'damp_freqs':2}[ret_name]
        )
    # poly_uq.save_state(os.path.join(result_dir,'estimations', f'{ret_dir}/polyuq_imp.npz'))
    return poly_uq.intp_errors, poly_uq.intp_exceed, poly_uq.intp_undershot

def test_domain():

    # ansys.exit()
    '''
    E           1.84e+11   2.3e+11    2.07e+11
    a           0.875      0.925      0.9
    b           0.875      0.925      0.9
    t           0.00569    0.00631    0.006
    rho         7.7e+03    7.85e+03   7.78e+03
    N_wire      3.18e+04   1.88e+05   1.1e+05
    A_wire      0.0007     0.0008     0.00075
    add_mass    10         100        55
    zeta        0.0016     0.084      0.0428
    dD          74         321        198
    ice_occ     0          1
    ice_mass    50         100        75
    '''
    titles = ['160 m', '200 m', 'TMD']
    frf_ind = 0
    import matplotlib.pyplot as plt
    import time
    # nominal
    ymin=0
    ymax=0.01
    freqs = np.linspace(0, 5, 2048 // 2 + 1, False)
    now=time.time()
    f,zeta,frf = mapping_function(b=0.9, t=0.006, N_wire=110000, 
                     A_wire=0.00075, add_mass=60, zeta=0.0083, dD=197.61, # structural
                    ice_occ=1, ice_mass=0.5, # environmental
                   jid='nominal12', working_dir='/dev/shm/womo1998/', skip_existing=True)
    print(time.time()-now)

    
    for frf_ind in range(3):
        frf_ = frf[:,frf_ind]
        plt.figure()
        plt.plot(freqs,frf_, color='dimgrey')
        plt.vlines(x=f, ymin=ymin, ymax=ymax, colors='lightgrey', zorder=-10)
        plt.ylim((ymin,ymax))
        plt.twinx()
        plt.plot(f,zeta, ls='none', marker='x', color='black')
        plt.title(titles[frf_ind])
        plt.ylim((0,0.14))
        plt.xlim((0,5))
    # max f,d
    now=time.time()
    f,zeta,frf = mapping_function(b=0.95, t=0.00637190164854557,N_wire=189873.988768885, 
                     A_wire=0.0008, add_mass=20, zeta=0.0016, dD=104.634587863608, # structural
                    ice_occ=0, ice_mass=0, # environmental
                     jid='max12', working_dir='/dev/shm/womo1998/', skip_existing=True)
    print(time.time()-now)
    for frf_ind in range(3):
        frf_ = frf[:,frf_ind]
        plt.figure()
        plt.plot(freqs,frf_, color='dimgrey')
        plt.vlines(x=f, ymin=ymin, ymax=ymax, colors='lightgrey', zorder=-10)
        plt.ylim((ymin,ymax))
        plt.twinx()
        plt.plot(f,zeta, ls='none', marker='x', color='black')
        plt.title(titles[frf_ind])
        plt.ylim((0,0.14))
        plt.xlim((0,5))
    # min f,d
    now=time.time()
    f,zeta,frf = mapping_function(b=0.85, t=0.00562809835145443, N_wire=30126.0112311152, 
                     A_wire=0.0007, add_mass=100, zeta=0.015, dD=290.585412136393, # structural
                    ice_occ=1, ice_mass=1, # environmental
                     jid='min12', working_dir='/dev/shm/womo1998/', skip_existing=True)
    print(time.time()-now)
    for frf_ind in range(3):
        frf_ = frf[:,frf_ind]
        plt.figure()
        plt.plot(freqs,frf_, color='dimgrey')
        plt.vlines(x=f, ymin=ymin, ymax=ymax, colors='lightgrey', zorder=-10)
        plt.ylim((ymin,ymax))
        plt.twinx()
        plt.plot(f,zeta, ls='none', marker='x', color='black')
        plt.title(titles[frf_ind])
        plt.ylim((0,0.14))
        plt.xlim((0,5))
    
    plt.show()

def vars_definition():
    b = MassFunction('b',[(1.7/2,1.9/2),],[1,], primary=True) # meter
    tnorm = RandomVariable('norm','tnorm', [6e-3,1e-4], primary=False) # meter
    t = MassFunction('t',[(5.9e-3,6.1e-3),(tnorm,),],[0.7,0.3], primary=True) # meter
    add_mass = MassFunction('add_mass', [(20,100),(40,50)],[0.8, 0.2], primary=True) # kilogram per meter
    muNwire = MassFunction('muNwire',[(60000,),(40000,180000)],[0.75,0.25], primary=False) # Newton
    N_wire = RandomVariable('norm','N_wire', [muNwire, 2655], primary=True) # Newton
    A_wire = MassFunction('A_wire', [(0.0007,0.0008)],[1,], primary=True) # meter^2
    zeta = MassFunction('zeta', [(0.0075,0.0092), (0.0075,0.0134), (0.0016,0.015)], [0.5,0.3,0.2], primary=True) # -
    stddD = MassFunction('vardD',[(5,20),(10,15)],[0.2,0.8], primary=False) # Newton second per meter
    dD = RandomVariable('norm', 'dD',[197.61, stddD], primary=True)  # Newton second per meter
    ice_days = MassFunction('ice_days', [(28.2/365,), (1/365,77/365)],[0.3,0.7], primary=False) # days, incompleteness
    ice_occ = RandomVariable('bernoulli', 'ice_occ',  [ice_days], primary=True) # boolean, variability
    #actually ice_occ is not a primary variable, that was just a hack
    # epistemic samples are generated on the support (of ice_mass)
    # for each aleatory sample, the epistemic samples are propagated
    # if ice_occ==0 the model is run with all those ice_masses, even though, they are not present
    # that is why ice_occ had to be passed to the mapping function and thus needed to be primary
    ice_mass = MassFunction('ice_mass', [(0, ice_occ),],[1], primary=True, incremental=True) #, imprecision
    
    vars_epi = [b, t, add_mass, muNwire, A_wire, zeta, stddD, ice_mass, ice_days]
    # vars_epi = [E, A, rho, L]
    vars_ale = [tnorm, N_wire, dD, ice_occ]
    # vars_ale = [Elognorm, Enorm, Anorm]
    
    # E, a, b, t, rho, add_mass, N_wire, A_wire, zeta, dD, ice_occ, ice_mass,
    arg_vars = {'b':b.name,
                't':t.name,
                'add_mass':add_mass.name,
                'N_wire':N_wire.name,
                'A_wire':A_wire.name,
                'zeta':zeta.name,
                'dD':dD.name,
                'ice_occ':ice_occ.name,
                'ice_mass':ice_mass.name,}
    return vars_ale, vars_epi, arg_vars

def export_datamanager():
    from uncertainty.data_manager import DataManager
    vars_ale, vars_epi, arg_vars = vars_definition()
    dim_ex = 'cartesian'
    result_dir = '/usr/scratch4/sima9999/work/modal_uq/uq_modal_beam/'
    
    dm_grid = DataManager.from_existing('uq_modal_beam.nc',
                                    result_dir=os.path.join('/dev/shm/womo1998/', 'testsamples'), 
                                    working_dir='/dev/shm/womo1998/')
    with dm_grid.get_database('out',False) as out_ds:
        out_ds_keep = out_ds.copy()
    out_ds = out_ds_keep
    
    logger= logging.getLogger('uncertainty.polymorphic_uncertainty')
    logger.setLevel(level=logging.DEBUG)
    
    for ret_name in ['damp_freqs','zetas','frf']:
        if ret_name == 'frf':
            n = 1025*2
        else:
            n=14
            
        for i_n in range(n):
            if ret_name == 'frf':
                ret_ind = {'frequencies':i_n//2, 'space':i_n%2}
            else:
                ret_ind = {'modes':i_n}
            ret_dir = f'{ret_name}-{".".join(str(e) for e in ret_ind.values())}'
            
            print(ret_dir)
    
            poly_uq = PolyUQ(vars_ale, vars_epi, dim_ex=dim_ex)
            
            samp_path = os.path.join(result_dir,'polyuq_samp.npz')
            prop_path = os.path.join(result_dir, 'estimations', f'{ret_dir}/polyuq_prop.npz')
            imp_path = os.path.join(result_dir, 'estimations', f'{ret_dir}/polyuq_imp.npz')
            
            if os.path.exists(prop_path):
                continue
            
            
            for path in [imp_path, prop_path, samp_path]:
                if os.path.exists(path):
                    poly_uq.load_state(path)
                    
                    poly_uq.from_data_manager(None, ret_name, ret_ind, out_ds)
                    if ret_name == 'damp_freqs':
                        poly_uq.out_valid = [np.nanmin(poly_uq.out_samp),np.nanmax(poly_uq.out_samp)]
                    if path == samp_path:
                        path = prop_path
                    poly_uq.save_state(path)
                    break

def main():
    # default_mapping()
    test_domain()


if __name__ == '__main__':
    # test_interpolation('frf', {'frequencies':127,'space':1}, 3062)
    # export_datamanager()
    main()