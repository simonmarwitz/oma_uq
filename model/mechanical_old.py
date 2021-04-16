import numpy as np
import matplotlib
from mpldatacursor import datacursor
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plot
import os
import glob
import shutil
import pyansys
import scipy.stats
import scipy.optimize
import scipy.signal
import time
import uuid
import warnings

print_context_dict ={'text.usetex':False,
                     'text.latex.preamble':r"\usepackage{siunitx},\usepackage{nicefrac}",
                     'font.size':10,                                                                                 
                     'legend.fontsize':10, 
                     'xtick.labelsize':10,
                     'ytick.labelsize':10,
                     'axes.labelsize':10,
                     'font.family':'serif',
                     'legend.labelspacing':0.1,
                     'axes.linewidth':0.5,                                                                           
                     'xtick.major.width':0.2,                                                                        
                     'ytick.major.width':0.2,                                                                        
                     'xtick.major.width':0.5,                                                                        
                     'ytick.major.width':0.5, 
                     'figure.figsize':(5.906,5.906/1.618),#print #150 mm \columnwidth
                     #'figure.figsize':(5.53/2,2.96),#beamer
                     #'figure.figsize':(5.53/2*2,2.96*2),#beamer
                     'figure.dpi':100}
    #figsize=(5.53,2.96)#beamer 16:9
    #figsize=(3.69,2.96)#beamer 16:9
    #plot.rc('axes.formatter',use_locale=True) #german months
# must be manually set due to some matplotlib bugs
if print_context_dict['text.usetex']:
    plot.rc('text.latex',unicode=True)
    plot.rc('text',usetex=True)

class Mechanical(object):
    def __init__(self, ansys=None):
        if ansys is None:
            ansys = pyansys.Mapdl(exec_file='/vegas/apps/ansys/v190/ansys/bin/ansys190', run_location='/dev/shm/', jobname='file', override=True, loglevel='INFO',nproc=2)
        assert isinstance(ansys, pyansys.Mapdl)
        self.ansys = ansys
        #ansys.finish()
        ansys.clear()
        ansys.config(lab='NOELDB',value=1)
#         ansys.config(lab='NORSTGM',value=1)
        ansys.output(fname='null',loc='/dev/')
        ansys.nopr() #Suppresses the expanded interpreted input data listing.
        ansys.nolist() #Suppresses the data input listing.
        ansys.finish()
        
        self.last_analysis = None
    
    def nonlinear(self, nl_ity=1, d_max=1.5, k_lin=1e5, k_nl=None, **kwargs):
        '''
        generate a nonlinear spring 
        with a linear part and a cubic nonlinear part
        nl_ity: the fraction of linear and nonlinear parts
            for nl_ity = (-0.5,0] it is a nonlinear softening spring
            for nl_ity = 0 it is a linear spring
            for nl_ity = [0,0.5) it is a nonlinar hardening spring
        d_max: maximal displacement, up to which the force-displacement curve will be defined. ansys will interpolate further points and (probably) issue a warning.
        k_lin: linear stiffness
        k_nl: nonlinear stiffness, if not automatically determined: k_lin and k_nl should match at d_max
        '''
        if k_nl is not None:
            warnings.warn('Are you sure, you want to set k_nl manually. Make sure to not overshoot d_max')
        if nl_ity<-0.5 or nl_ity >0.5:
            raise RuntimeError('The fraction of nonlinearity should not exceed/subceed 0.5/-0.5. It currently is set to {}.'.format(nl_ity))
        
        ansys = self.ansys
        ansys.prep7()
        
        ansys.run('nset=rlinqr(0,14)')
        ansys.run('itype=etyiqr(0,14)')
        ansys.load_parameters()
        

        if k_nl is None:
            k_at_d_max = k_lin/d_max**2
            k_nl = k_at_d_max
        ansys.et(itype= ansys.parameters['ITYPE']+1, ename='COMBIN39',kop3='3',inopr=1)
        d = np.linspace(0,d_max,20,endpoint=True)
        F=k_lin*d*(1-nl_ity)+k_nl*d**3*(nl_ity)
#         plot.plot(d,F)
#         plot.show()
        d=list(d)
        F=list(F)
        command = 'R, {}'.format(ansys.parameters['NSET']+1)
        i=0
        while len(d)>=1:
            command+=''.join(', {}, {}'.format(d.pop(0),F.pop(0)))
            i+=1
            if i == 3:
                ansys.run(command)
                #print(command)
                command='RMORE'
                i=0
        else:
            ansys.run(command)
            #print(command)
        
        ansys.run('nset=rlinqr(0,14)')
        ansys.run('itype=etyiqr(0,14)')
        ansys.load_parameters()
        
        ansys.finish()
        return ansys.parameters['ITYPE'], ansys.parameters['NSET'] #itype, nset
    
        #ansys.r(nset=1, r1=d.pop(0), r2=F.pop(0), r3=d.pop(0), r4=F.pop(0), r5=d.pop(0), r6=F.pop(0))
        
    def voigt_kelvin(self,k = 100000, d= 150, **kwargs):
        ansys = self.ansys
        ansys.prep7()
        
        ansys.run('nset=rlinqr(0,14)')
        ansys.run('itype=etyiqr(0,14)')
        ansys.load_parameters()
        
        
        ansys.et(itype= ansys.parameters['ITYPE']+1, ename='COMBIN14',inopr=1, kop2='3')
        #              k,    cv1, cv2
        ansys.r(nset= ansys.parameters['NSET']+1, r1=k, r2=d)
        
#         Omega = 1
#         s=1
#         Wd = d*np.pi*Omega*s**2 # petersen eq. 555
#         print(f"Damping energy at unit displacement and unit circular frequency: {Wd}")
        
        
        ansys.run('nset=rlinqr(0,14)')
        ansys.run('itype=etyiqr(0,14)')
        ansys.load_parameters()
        
        ansys.finish()
        return ansys.parameters['ITYPE'], ansys.parameters['NSET'] #itype, nset
        
    def coulomb(self, k_1=90000, d=0, f_sl=15000, k_2=10000 ,**kwargs):
        '''
        k_1 and slider are in series
        k_1 is the sticking stiffness, and defines the displacement required 
        to reach the force, at which the slider breaks loose should be k_2*10
            
        
        f_sl determines the constant friction damping force that is applied each half-cycle
            a lower f_sl means less damping, longer decay
            a higher f_sl means higher damping, shorter decay
            should be d_0/k_2
        
        high f_sl or low k_1 have the same effect
            
        k_2 is the direct stiffness and in parallel with k_1+slider and d
            determines the oscillation frequency if a mass is attached to the element
        
        apdl modal solver uses k_tot = k_1 + k_2
        '''
        ansys = self.ansys
        ansys.prep7()
        
        k_tot = kwargs.pop('k_tot',k_2+k_1)
        k_2 = k_tot - k_1
        if k_tot:
            f_sl_in = f_sl*k_1/k_tot
        else:
            f_sl_in = f_sl
        
#         print('Displacement amplitude above which friction force becomes active and below which no more dissipation happens {}'.format(f_sl/k_1))
#         s=1
#         Wd = 4*f_sl_in*(s-f_sl_in/k_1) # Petersen eq 637
#         print(f"Damping energy at {s} displacement: {Wd}")
        
        
        ansys.run('nset=rlinqr(0,14)')
        ansys.run('itype=etyiqr(0,14)')
        ansys.load_parameters()
        
        ansys.et(itype= ansys.parameters['ITYPE']+1, ename='COMBIN40',kop3='3', inopr=1)
        #               K1,     C,     M,    GAP,  FSLIDE, K2
        #print(k_1, d,f_sl_in, k_2)
        ansys.r(nset= ansys.parameters['NSET']+1, r1=k_1, r2=d, r3=0.0, r4=0.0, r5=f_sl_in, r6=k_2)
        #               K1,     C,     M,    GAP,  FSLIDE, K2
        #ansys.r(nset=1, r1=0, r2=1000, r3=0, r4=0, r5=0, r6=100)#
#         print(f"nset= {ansys.parameters['NSET']+1}, r1={k_1}, r2={d}, r3={0.0}, r4={0.0}, r5={f_sl_in}, r6={k_2}")
        ansys.run('nset=rlinqr(0,14)')
        ansys.run('itype=etyiqr(0,14)')
        ansys.load_parameters()
        
        ansys.finish()
        return ansys.parameters['ITYPE'], ansys.parameters['NSET'] #itype, nset
    
    def mass(self, m=100):
        ansys=self.ansys
        ansys.prep7()
        
        ansys.run('nset=rlinqr(0,14)')
        ansys.run('itype=etyiqr(0,14)')
        ansys.load_parameters()
        
        ansys.et(itype= ansys.parameters['ITYPE']+1, ename='MASS21',inopr=1)
        ansys.r(nset= ansys.parameters['NSET']+1, r3=m)
        
        ansys.run('nset=rlinqr(0,14)')
        ansys.run('itype=etyiqr(0,14)')
        ansys.load_parameters()
        
        ansys.finish()
        return ansys.parameters['ITYPE'], ansys.parameters['NSET'] #itype, nset
    
    def beam(self, E, PRXY, A, Iyy, Izz, Iyz, ):
        ansys=self.ansys
        ansys.prep7()
        
        ansys.run('nset=rlinqr(0,14)')
        ansys.run('itype=etyiqr(0,14)')
        ansys.load_parameters()
        
        ansys.et(itype= ansys.parameters['ITYPE']+1, ename='BEAM188',kop3=3, inopr=1)
#         ansys.r(nset= ansys.parameters['NSET']+1, r3=m)
        ansys.sectype(1, "BEAM", "ASEC", "tube", 0)
        '''
        Arbitrary: User-supplied integrated section
        properties instead of basic geometry data.
        Data to provide in the value fields:
        A, Iyy, Iyz, Izz, Iw, J, CGy, CGz, SHy,
        SHz, TKz, TKy
        where
        A = Area of section
        Iyy = Moment of inertia about the y axis
        Iyz = Product of inertia
        Izz = Moment of inertia about the z axis
        Iw = Warping constant
        J = Torsional constant
        CGy = y coordinate of centroid
        CGz = z coordinate of centroid
        SHy = y coordinate of shear center
        SHz = z coordinate of shear center
        TKz = Thickness along Z axis (maximum
        height)
        TKy = Thickness along Y axis (maximum
        width
        '''
        CGy=0
        CGz=0
        SHz=0
        SHy=0
        Tkz=0
        Tky=0
        # validate the above at some point
        
        IW = 0
        J = 0
        # no warping, no torsion, maybe has to be set to infty or whatever
        
        ansys.secdata(A,Iyy,Iyz,Izz,IW,J,CGy,CGz,SHy,SHz,Tkz,Tky)
        #SECOFFSET, CENT
        
        ansys.run('nset=rlinqr(0,14)')
        ansys.run('itype=etyiqr(0,14)')
        ansys.load_parameters()
        
        ansys.finish()
        return ansys.parameters['ITYPE'], ansys.parameters['NSET'] #itype, nset
    
    def build_conti(self, parameters, Ldiv, initial=None, meas_locs=None):
        ansys=self.ansys
        ansys.prep7()
        assert Ldiv >= 3
        
        #Nodes
        L = parameters['L']
        x_nodes = np.linspace(0,L,Ldiv) 
        
        x_knl = parameters['x_knl']
        x_nodes[np.argmin(np.abs(x_nodes-x_knl))] = x_knl
        
        if initial is not None:
            x_d0 = initial['x_d0']
            x_nodes[np.argmin(np.abs(x_nodes-x_d0))] = x_d0
        
        print(x_nodes, x_knl, x_d0)
        
        for x_node in x_nodes:
            ansys.n(x=x_node, y=0, z=0)
        
        
        
        #boundary conditions
        ansys.nsel(type, item='LOC', comp='x', vmin=0, vmax=0)
        ansys.get('bcnode','node',0,"num","min")
        ansys.d(node='bcnode', value=0,lab='UX',lab2='UY',lab3='UZ')
        
        
        ansys.nsel(type, item='LOC', comp='x', vmin=x_knl, vmax=x_knl)
        ansys.get('knlnode','node',0,"num","min")
        ansys.d(node='knlnode', value=0,lab2='UY',lab3='UZ')
        
        
        if initial is not None:
            x_d0 = initial['x_d0']
            d0y = initial['d0y']
            d0z = initial['d0z']
            
            ansys.nsel(type, item='LOC', comp='x', vmin=x_d0, vmax=x_d0)
            ansys.get('ininode','node',0,"num","min")
            ansys.ic(node='ininode', lab='UY', value=d0y)
            ansys.ic(node='ininode', lab='UZ', value=d0z)
            
            
        if meas_locs is not None:
            ansys.nsel(type='NONE')
            for x_loc in meas_locs:
                ansys.nsel('A', item='LOC', comp='x', vmin=x_loc, vmax=x_loc)
            ansys.cm(cname='meas_nodes', entity='NODE') # and group into component assembly
        else:
            ansys.nsel(type='ALL')
            
        ansys.cm(cname='meas_nodes', entity='NODE') # and group into component assembly
        
        
        
        ansys.finish()        

    def build_mdof(self, f_init=None, d_init=None, masses=None, meas_nodes=None,  **kwargs):
        ansys=self.ansys
        ansys.prep7()
        #Nodes
        #ansys.n(40,0,0,2)
        ansys.n(20,0,0,2)
        ansys.n(30,0,0,2)
        ansys.n(10,0,0,2)
        
        # Elements
         
        voigt_kelvins = kwargs.pop('voigt_kelvins', None)
        
        
            
        if voigt_kelvins is not None:
            for voigt_kelvin, nodes in zip(voigt_kelvins,((20,10),(20,30))): 
                ansys.type(voigt_kelvin[0])
                ansys.real(voigt_kelvin[1])
                ansys.e(*nodes)

        
        if masses is not None:
            for mass, node in zip(masses, (20,30)):
                ansys.type(mass[0])
                ansys.real(mass[1])
                ansys.e(node)
            #ansys.e(30)
            #ansys.e(40)
        
       
        
        
        
        #boundary conditions
        ansys.d(node=10,value=0,lab='UX',lab2='UY',lab3='UZ')
        ansys.d(node=20,value=0,lab='UX',lab2='UY')
        ansys.d(node=30,value=0,lab='UX',lab2='UY')
        #ansys.d(node=40,value=0,lab='UX',lab2='UY')
        if f_init is not None:
            ansys.f(node=30, lab='FZ', value=f_init)
        if d_init is not None:
            ansys.ic(node=30, lab='UZ', value=d_init)
        
        
        if meas_nodes is not None:
            ansys.nsel(type='NONE')
            for node in meas_nodes:
                ansys.nsel(type='A', item='NODE', vmin=node,vmax=node) # select only nodes of interest
            #ansys.nsel(type='A', item='NODE', vmin=2,vmax=2) # select only nodes of interest
            ansys.cm(cname='meas_nodes', entity='NODE') # and group into component assembly
        else:
            ansys.nsel(type='ALL')
        ansys.cm(cname='meas_nodes', entity='NODE') # and group into component assembly
            
        ansys.finish()    
    
    def build_sdof(self, f_init=None, d_init=None, mass=None, **kwargs):
        ansys=self.ansys
        ansys.prep7()
        #Nodes
        #ansys.n(40,0,0,2)
        #ansys.n(30,0,0,2)
        ansys.n(20,0,0,2)
        ansys.n(10,0,0,2)
        
        # Elements
        
        #mass
        
        if mass is not None:
            ansys.type(mass[0])
            ansys.real(mass[1])
            ansys.e(20)
            
        #ansys.e(30)
        #ansys.e(40)
        
        coulomb = kwargs.pop('coulomb', None)
        voigt_kelvin = kwargs.pop('voigt_kelvin', None)
        nonlinear = kwargs.pop('nonlinear', None)
        
        if coulomb is not None:
            ansys.type(coulomb[0])
            ansys.real(coulomb[1])
            ansys.e(10,20)
            
        if voigt_kelvin is not None:
            ansys.type(voigt_kelvin[0])
            ansys.real(voigt_kelvin[1])
            ansys.e(10,20)
            
        if nonlinear is not None:
            ansys.type(nonlinear[0])
            ansys.real(nonlinear[1])
            ansys.e(10,20)
        #ansys.e(20,30)
        #ansys.e(30,40)
        
        
        
        #boundary conditions
        ansys.d(node=10,value=0,lab='UX',lab2='UY',lab3='UZ')
        ansys.d(node=20,value=0,lab='UX',lab2='UY')
        if f_init is not None:
            ansys.f(node=20, lab='FZ', value=f_init)
        if d_init is not None:
            ansys.ic(node=20, lab='UZ', value=d_init)
        #ansys.d(node=30,value=0,lab='UX',lab2='UY')
        #ansys.d(node=40,value=0,lab='UX',lab2='UY')
        
        ansys.finish()
        
    def static(self, fz=1):
        ansys = self.ansys
        # Static Response
        ansys.run('/SOL')
        ansys.antype('STATIC')
        ansys.deltim(1)
        ansys.f(node=20,lab='FZ',value=fz)
        ansys.solve()
        ansys.finish()
            #ansys.finish()
        ansys.post1()
        ansys.get(par='uz', entity='NODE', entnum='20', item1='U', it1num='Z')
        ansys.load_parameters()
        print( fz,ansys.parameters['UZ'])
        ansys.finish()

        
        self.last_analysis = 'static'
    def modal(self, damped=True, coulombs=None):# Modal Analysis
        ansys = self.ansys        
        
        ansys.prep7()
        if coulombs is not None:
            real_constants = []
            for coulomb in coulombs: 
                nset=coulomb[1]
                #print(ansys.rlist(nset))
                ansys.get('k_coul', entity="rcon", entnum=nset, item1=1, ) #r5=f_sl_in, r6=k_2
                ansys.get('f_coul', entity="rcon", entnum=nset, item1=5, ) #r5=f_sl_in, r6=k_2
                ansys.load_parameters()
                real_constants.append((ansys.parameters["K_COUL"], ansys.parameters["F_COUL"]))
                #print(real_constants)
                ansys.rmodif(nset, 1, 0)
                ansys.rmodif(nset, 5, 0)
                #print(ansys.rlist(nset))
                #print(real_constants[-1])
        ansys.finish()
        
        ansys.run('/SOL')
        ansys.antype('MODAL')
                
        ansys.outres(item='ERASE')
        ansys.outres(item='ALL',freq='NONE')# Disable all output
        ansys.nsel(type='S', item='NODE', vmin=20,vmax=20) # select only nodes of interest
        #ansys.nsel(type='A', item='NODE', vmin=2,vmax=2) # select only nodes of interest
        ansys.cm(cname='meas_nodes', entity='NODE') # and group into component assembly
        ansys.nsel(type='ALL')
        ansys.outres(#item='A',
                     item='NSOL',
                     freq='ALL'
                    ,cname='meas_nodes'
                     )# Controls the solution data written to the database.
        if damped:
            ansys.modopt(method='QRDAMP',nmode=50,freqb=0,
                         freqe=1e8,
                         cpxmod='cplx',
                         nrmkey='on',
                         )
        else:
            ansys.modopt(method='LANB',nmode=50,freqb=0,
                         freqe=1e8,
                         nrmkey='on',
                         )
            
        ansys.mxpand(nmode='all',elcalc=1)
        ansys.solve()
        ansys.finish()
        
        ansys.prep7()
        if coulombs is not None:
            for coulomb, real_constant in zip(coulombs, real_constants):
                nset=coulomb[1]
                ansys.rmodif(nset, 1, real_constant[0])
                ansys.rmodif(nset, 5, real_constant[1])
        ansys.finish()
        
        self.last_analysis = 'modal'
    
    def transient(self, f=None, d=None, meth='NMK', parameter_set=None, **kwargs):
        ansys = self.ansys
        # Transient/Harmonic Response
        deltat=self.deltat
        timesteps=self.timesteps
        
        ansys.run('/SOL')
        ansys.antype('TRANS')
        ansys.trnopt(method='FULL',vaout=meth)# bug: vaout should be tintopt
        
        delta=""
        alpha=""
        gamma=""
        theta=""
        alphaf=""
        alpham=""
        
        if meth == 'NMK':
            if isinstance(parameter_set, tuple):
                assert len(parameter_set) == 2
                delta, alpha = parameter_set
            elif isinstance(parameter_set, str):
                if parameter_set == 'AAM':# Trapezoidal Rule -> Constant Acceleration
                    delta = 1/2
                    alpha = 1/4
                elif parameter_set == 'LAM': # Original Newmark 1/6 -> Linear Acceleration
                    delta=1/2
                    alpha=1/6
                elif parameter_set == 'CDM': # Explicit Central Difference Method
                    delta=1/2
                    alpha=0 
                else:
                    print("Using default parameters (linear acceleration) for NMK integration.")
                    delta=1/2
                    alpha = 1/6
            elif isinstance(parameter_set, float):
                rho_inf = parameter_set
                alphaf = 0
                alpham = 0
                delta = (3-rho_inf)/(2*rho_inf+2)
                alpha = 1/((rho_inf+1)**2)
            else:
                print("Using default parameters (linear acceleration) for NMK integration.")
                delta=1/2
                alpha = 1/6
            
            
        elif meth == 'HHT':# HHT-α Hilber-Hugh Taylor
            rho_inf = parameter_set
            alphaf = (1-rho_inf)/(rho_inf+1)
            alpham = 0
            delta = 1/2+alphaf
            alpha = (1+alphaf)**2/4
        elif meth == 'WBZ': # WBZ-α Wood-Bosak-Zienkiewicz 
            rho_inf = parameter_set
            alphaf = 0
            alpham = (rho_inf-1)/(rho_inf+1)
            delta = 1/2-alpham
            alpha = (1-alpham)**2/4
        elif meth == 'G-alpha':
            rho_inf = parameter_set
            alpham = (2*rho_inf-1)/(rho_inf+1)
            alphaf = rho_inf/(rho_inf+1)
            delta = 1/2-alpham+alphaf
            alpha = 1/4*(1-alpham+alphaf)**2
        else:
            assert len(parameter_set) == 4
            delta, alpha, alphaf, alpham = parameter_set
        
        ansys.tintp(gamma=gamma, alpha=alpha, delta=delta, theta=theta, oslm="", tol="", avsmooth="", alphaf=alphaf, alpham=alpham,)
        ansys.kbc(1) # Specifies ramped or stepped loading within a load step.
        ansys.timint('ON') #Turns on transient effects.
        ansys.outres(item='ERASE')
        ansys.outres(item='ALL',freq='NONE')# Disable all output
        
        ansys.nsel(type='ALL')
        ansys.outres(#item='A',
                     item='NSOL',
                     freq='LAST' #Writes the specified solution results item only for the last substep of each load step. This value is the default for a static (ANTYPE,STATIC) or transient (ANTYPE,TRANS) analysis.
                    ,cname='meas_nodes'
                     )# Controls the solution data written to the database.
        ansys.outres(item='A',
                     #item='NSOL',
                     freq='LAST'
                    ,cname='meas_nodes'
                     )# Controls the solution data written to the database.
        ansys.outres(item='V',
                    #item='NSOL',
                    freq='LAST'
                    ,cname='meas_nodes'
                    )# Controls the solution data written to the database.
        ansys.rescontrol(action='DEFINE',ldstep='NONE',frequency='NONE',maxfiles=-1)  # Controls file writing for multiframe restarts
        ansys.deltim(dtime=deltat, dtmin=deltat, dtmax=deltat, carry='OFF')
        
        
        
        t_end = deltat*(timesteps)
        t = np.linspace(deltat,stop=t_end,num=timesteps)
        
        
        
        printsteps = list(np.linspace(0,timesteps, 100, dtype=int))
        t_start = time.time()
        
        dts=[]
        #timesteps=10000
        for lsnum in range(timesteps):
            if not lsnum %1000 and lsnum:
                t_end=t_start
                t_start = time.time()
                dts.append(t_start-t_end)
                print(f'1000 timesteps in {np.mean(dts)} s')
            #print(lsnum)
            #continue
#             while lsnum in printsteps: 
# 
#                 del printsteps[0]
#                 print('.',end='', flush=True)
            #continue
            ansys.time((lsnum+1)*deltat)
            
            if f is not None:
                ansys.fdele(node='ALL', lab='ALL')
                ansys.f(node=20, lab='FZ', value=f[lsnum])
            
            if d is not None:
                ansys.ddele(node=20, lab='UZ')
                if d[lsnum]:
                    ansys.d(node=20, lab='UZ', value=d[lsnum])
                else:
                    ansys.timint(1)
            
            #ansys.lswrite()
            ansys.solve()
        #ansys.lssolve()

        print('.',end='\n', flush=True)
        ansys.finish()
        
        self.last_analysis = 'trans'
    
    def mode_superpos(self, f=None, d=None):
        ansys = self.ansys
        # Transient/Harmonic Response
        deltat=self.deltat
        timesteps=self.timesteps
        
        ansys.run('/SOL')
        ansys.antype('TRANS')
        ansys.trnopt(method='MSUP')# bug: vaout should be tintopt
        
        
        ansys.kbc(0) # Specifies ramped or stepped loading within a load step.
        ansys.timint(1) #Turns on transient effects.
        ansys.alphad(value=0)
        ansys.betad(value=0)
        ansys.dmprat(ratio=0)
        ansys.mdamp(stloc=1, v1=0, v2=0, v3=0, v4=0, v5=0, v6=0)
#         ansys.outres(item='ERASE')
        ansys.outres(item='ALL',freq='ALL')# Enable all output
#         ansys.nsel(type='S', item='NODE', vmin=30,vmax=40) # select only nodes of interest
#         #ansys.nsel(type='A', item='NODE', vmin=2,vmax=2) # select only nodes of interest
#         #ansys.cm(cname='meas_nodes', entity='NODE') # and group into component assembly
#         ansys.nsel(type='ALL')
#         ansys.outres(#item='A',
#                      item='NSOL',
#                      freq='ALL'
#                      #,cname='meas_nodes'
#                      )# Controls the solution data written to the database.
#         ansys.outres(item='A',
#                      #item='NSOL',
#                      freq='ALL'
#                      #,cname='meas_nodes'
#                      )# Controls the solution data written to the database.
#         ansys.outres(item='V',
#                      #item='NSOL',
#                      freq='ALL'
#                      #,cname='meas_nodes'
#                      )# Controls the solution data written to the database.
#         ansys.rescontrol(action='DEFINE',ldstep='NONE',frequency='NONE',maxfiles=-1)  # Controls file writing for multiframe restarts
        ansys.deltim(dtime=deltat, dtmin=deltat, dtmax=deltat, carry='OFF')
        
        #ansys.solve()
        
        t_end = deltat*(timesteps-1)
        t = np.linspace(deltat,stop=t_end,num=timesteps)
        
        printsteps = list(np.linspace(0,timesteps, 100, dtype=int))
        dts=[]
        timesteps=10000
        t_start = time.time()
        for lsnum in range(timesteps):
            if not lsnum %1000:
                t_end=t_start
                t_start = time.time()
                dts.append(t_start-t_end)
                print(lsnum, np.mean(dts))
                
            while lsnum in printsteps: 
                del printsteps[0]
                print('.',end='', flush=True)
            ansys.time((lsnum+1)*deltat)
            if f is not None:
                ansys.fdele(node='ALL', lab='ALL')
                ansys.f(node=20,lab='FZ', value=f[lsnum])
            if d is not None:
                ansys.ddele(node=20, lab='UZ')
                if d[lsnum]:
                    ansys.d(node=20, lab='UZ', value=d[lsnum])
            #continue
            ansys.solve()
        #np.savetxt('dts_amsupsolve',dts)
        #asd
        print('.',end='\n', flush=True)  
        ansys.finish()
        ansys.run('/SOL')
        ansys.expass(key='ON')
        ansys.numexp(num=timesteps, begrng=0, endrng=timesteps*deltat)
        ansys.solve()
        ansys.finish()
        self.last_analysis = 'trans'
        
    def harmonic_inc(self, deltat = 0.01, timesteps = 1024, cycl = 8, ampl= 100000):

        f = np.sin(np.linspace(0, 2*cycl*np.pi, timesteps,endpoint=True))
        #f*=np.linspace(0,ampl,timesteps)
        f*=ampl
            
        self.deltat = deltat
        self.timesteps = timesteps
        
        return f
    
    def sweep(self, deltat = 0.01, timesteps = 1024, f_start=0/2/np.pi, f_end=None, phi=0, ampl= 1000):
        
        nyq = 1/deltat/2.5        
        if f_end is None:
            f_end=nyq
        
        t_end = deltat*timesteps
        t = np.linspace(deltat,stop=t_end,num=timesteps)

        assert f_end<=nyq
        
        f = np.sin(2*np.pi*((f_start*t)+(f_end-f_start)/(2*t_end)*t**2+phi))*ampl

        self.deltat = deltat
        self.timesteps = timesteps
        
        return f    
    
    def impulse(self, deltat = 0.01, timesteps = 1024, imp_len = 10, ampl=10000):
        
        nyq = 1/deltat/2        
        
        t_end = deltat*timesteps
        t = np.linspace(deltat,stop=t_end,num=timesteps)
        # Impulse of length imp_len*deltat
        
        f=np.sin(np.linspace(0,np.pi,imp_len, endpoint=True))*ampl
        f = np.concatenate((f, np.zeros(timesteps-imp_len)))
            
        self.deltat = deltat
        self.timesteps = timesteps
        
        return f    
    
    def white_noise(self, deltat = 0.01, timesteps = 1024, ampl=1000):
        
        # white noise
        f = np.random.randn(timesteps)*ampl
            
        self.deltat = deltat
        self.timesteps = timesteps
        
        return f
    
def response_frequency(path):
    res=pyansys.read_binary(path)
    node_numbers=res.geometry['nnum']
    num_nodes=len(node_numbers)
    time_values=res.time_values
    dt = time_values[1]-time_values[0]
    time_values -= dt

        
    solution_data_info = res.solution_data_info(0)
    DOFS = solution_data_info['DOFS']
    
    uz = DOFS.index(3)
    nnum, all_disp = res.nodal_time_history('NSL')
    #print(nnum)

    ydata = all_disp[:,np.where(nnum==20)[0][0],uz]
    this_t = time_values
    
    popt, pcov = scipy.optimize.curve_fit(f=free_decay, xdata = this_t, ydata=ydata, p0=[0.5,0.05,2*np.pi,0])#,  bounds=[(-1,0,0,0),(1,1,np.pi/dt,2*np.pi)])
    perr = np.sqrt(np.diag(pcov))
    print('R: {:1.3f} m, zeta: {:1.3f} \%, f_d: {:1.4f} Hz, phi: {:1.4f}'.format(popt[0], popt[1]*100,  popt[2]/2/np.pi, popt[3]*180/np.pi))
    return popt, perr
    
def process(path, last_analysis = 'trans', f = None):
    
    
    '''
    nsol,2,2,U,Z,uz  
    store,merge 
    plvar,2
    
    nsol,3,2,ACC,Z,acz  
    store,merge 
    plvar,3 
    '''
    
    res=pyansys.read_binary(path)
    
    if last_analysis=='static': #static
        nodes, disp = res.nodal_solution(0)
        uz=disp[nodes==20, 2]#knoten 20, DOF 2 (UZ)
        print(uz)
        return uz
        
    elif last_analysis == 'modal': #modal
        num_modes = res.resultheader['nsets']
        if res.resultheader['cpxrst']: # real and imaginary parts are saved as separate sets
            num_modes //= 2
        nnodes = res.resultheader['nnod']
        ndof = res.resultheader['numdof']
        
        mode_shapes = np.full((nnodes,ndof,num_modes), (1+1j)*np.nan, dtype=complex)
        frequencies = np.full(num_modes, np.nan)
        damping = np.full(num_modes, np.nan)
        
        if res.resultheader['cpxrst']:
            for mode in range(num_modes):
                sigma = res.time_values[2*mode]
                omega = res.time_values[2*mode+1]
                if omega < 0 : continue # complex conjugate pair
                
                frequencies[mode] = omega
                damping[mode] = -sigma/np.sqrt(sigma**2+omega**2)

                mode_shapes[:,:,mode].real= res.nodal_solution(2*mode)[1]                
                mode_shapes[:,:,mode].imag= res.nodal_solution(2*mode+1)[1]
                
        else:
            frequencies[:] = res.time_values
            for mode in range(num_modes):
                nnum, modal_disp = res.nodal_solution(mode)
                mode_shapes[:,:,mode]= modal_disp
            
        return frequencies, damping, mode_shapes
        
    elif last_analysis == 'trans': #transient
        
        #meas_nodes=res.geometry['components']['MEAS_NODES']
        node_numbers=res.geometry['nnum']
        num_nodes=len(node_numbers)
        time_values=res.time_values
        #print(time_values)
        dt = time_values[1]-time_values[0]
        time_values -= dt
        #print(dt)
        #print(res.resultheader['neqv'])
            
        solution_data_info = res.solution_data_info(0)
        DOFS = solution_data_info['DOFS']
        
        uz = DOFS.index(3)
        nnum, all_disp = res.nodal_time_history('NSL')
        
        #print(nnum, all_disp.shape)
        if f is not None:
            #fix,axes = plot.subplots(nrows=2, ncols=2, sharex='col', sharey='row')
            fig = plot.figure()
            ax1 = fig.add_subplot(221)
            ax2 = fig.add_subplot(222, sharey=ax1)
            ax3 = fig.add_subplot(223, sharex=ax1)
            ax4 = fig.add_subplot(224)
            t = time_values#np.linspace(0,f.shape[0]*dt,f.shape[0])
            ax2.plot(t,f, marker='+')
        for node in range(1,num_nodes):
            if f is not None:
            
                ydata = all_disp[:,node,uz]
                #ydata = np.concatenate(([1],ydata))
                this_t = time_values
                #this_t = np.concatenate(([0],time_values))
                
                ax1.plot(all_disp[:,node,uz],f, label=str(nnum[node]), marker='+')
                
                ax3.plot(ydata,this_t, marker='+')
                popt, pcov = scipy.optimize.curve_fit(f=free_decay, xdata = this_t, ydata=ydata, p0=[0.5,0.05,2*np.pi,0])#,  bounds=[(-1,0,0,0),(1,1,np.pi/dt,2*np.pi)])
                perr = np.sqrt(np.diag(pcov))
                print('R: {:1.3f} m, zeta: {:1.3f} \%, f_d: {:1.4f} Hz, phi: {:1.4f}'.format(popt[0], popt[1]*100,  popt[2]/2/np.pi, popt[3]*180/np.pi))
                print(perr)
                print(popt)
                this_t = np.linspace(0,time_values[-1],len(time_values)*10)
                ax3.plot(free_decay(this_t,*popt),this_t)
                
                Sxx = np.abs(np.fft.rfft(all_disp[:,node,uz]))
                freq = np.fft.rfftfreq(all_disp.shape[0], dt)
                #freq, Sxx = scipy.signal.welch(all_disp[:,node,uz], fs=1/dt, nperseg = f.shape[0]//2)  
                ax4.plot(freq, Sxx, marker='+')
                ax4.axvline(20.0)
                print(freq[Sxx.argmax()])
            else:
                plot.plot(all_disp[:,node,uz], label=str(nnum[node]))
        
        ax = ax1
        #ax.grid(True)
        ax.legend()
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')
        ax.set_xlabel('y [m]')
        ax.xaxis.set_label_coords(1.05, 0.55)
        ax.set_ylabel('F [N]', rotation='horizontal')
        ax.yaxis.set_label_coords(0.45, 1.05)
        
        xmin,xmax = ax.get_xlim()
        xlim = max(-1*xmin,xmax)
        ax.set_xlim((-1*xlim,xlim))
        
        ymin,ymax = ax.get_ylim()
        ylim = max(-1*ymin,ymax)
        ax.set_ylim((-1*ylim,ylim))
        plot.show()
    return

    path='/dev/shm/test.csv'
    t,d=[],[]
    f=open(path,'rt')
    f.readline()
    for line in f:
        l=line.split()
        t.append(float(l[0]))
        d.append(float(l[1]))
        
    d=np.array(d)
    t=np.array(t)
    blocks=1
    block_length = int(np.floor(len(t)/blocks))
    H=np.zeros(block_length,dtype=complex)
    for i in range(blocks):
        Sx=np.fft.fft(d[i*block_length:(i+1)*block_length])
        F=np.sin(t[i*block_length:(i+1)*block_length]*2*np.pi*t[i*block_length:(i+1)*block_length]/1000)
        Sf=np.fft.fft(F)
        Sff=Sf**2
        Sfx=Sf*Sx
        H+=Sfx/Sff
    H/=blocks
    fftfreq =np.fft.fftfreq(block_length, t[1]-t[0])
    fig,axes=plot.subplots(2,1,sharex='col')
    axes[0].plot(fftfreq, np.abs(H))
    axes[1].plot(fftfreq, np.angle(H)/np.pi*180)
    plot.xlim(xmin=0)
    plot.show()
    
    plot.plot(t,d)
    plot.plot(t,np.sin(t*2*np.pi*t/1000))
    
    plot.show()

def free_decay(t, R, zeta, omega_d, phi=0):
    
    #return -2*R*omega_d**2*np.exp(-zeta*omega_d/(np.sqrt(1-zeta**2))*t)*np.cos(omega_d*t+phi)
    return 2*R*np.exp(-zeta*omega_d/(np.sqrt(1-zeta**2))*t)*np.cos(omega_d*t+phi)

def generate_student(ansys, omega, zeta, d0, deltat=None,dt_fact=None,timesteps=None, num_cycles=None, f_scale=None, **kwargs):
    print(jid)
    assert deltat is not None or dt_fact is not None
    assert timesteps is not None or num_cycles is not None
    
    m    = 1    # kg
    k_2  = omega**2*m/(1-zeta**2)# N/m
    d=zeta*(2*np.sqrt(k_2*m))
    
    f_max = np.sqrt((k_2)/m)/np.pi/2    
    
    if dt_fact is None:
        dt_fact =deltat*f_max # \varOmega /2/pi
        print("\\varOmega",dt_fact*2*np.pi)
    elif deltat is None:
        deltat = dt_fact/f_max
    
    if dt_fact*2*np.pi > 0.1: print("Warning \\varOmega > 0.1")
    
    if timesteps is None:
        timesteps = int(np.ceil(num_cycles/f_max/deltat))
    elif num_cycles is None:
        num_cycles = int(np.floor(timesteps*f_max*deltat))
    if d0 is not None:
        print("Simulating system with omega {:1.3f}, zeta {:1.3f}, d0 {:1.3f}. deltat {:1.5f}, dt_fact {:1.2f}, timesteps {}, num_cycles {}".format(omega, zeta, d0, deltat, dt_fact, timesteps, num_cycles))
    elif f_scale is not None:
        print("Simulating system with omega {:1.3f}, zeta {:1.3f}, random excitation variance {:1.3f}. deltat {:1.5f}, dt_fact {:1.2f}, timesteps {}, num_cycles {}".format(omega, zeta, f_scale, deltat, dt_fact, timesteps, num_cycles))
    mech = Mechanical(ansys)
    voigt_kelvin=mech.voigt_kelvin(k=k_2, d=d)
    mass = mech.mass(m)
    mech.build_sdof(
               mass=mass, 
               voigt_kelvin = voigt_kelvin,
               d_init=d0
               )
    
    mech.modal(True)
    
    f_max = process(f'{jid}.rst', last_analysis=mech.last_analysis)[0].max()
    f_analytical = np.sqrt((k_2)/m)/np.pi/2
    zeta_analytical = d/2/np.sqrt(k_2*m)*100 # in percent of critical damping
    #print(f_analytical,zeta_analytical)

    mech.timesteps=timesteps
    mech.deltat = deltat
    
    #print(timesteps,deltat)
    
    f=np.zeros(timesteps)
    if d0 is not None:
        f[0]=d0
        mech.transient(d=f,parameter_set= kwargs.pop('parameter_set',None))
    else:
        assert f_scale is not None
        f = np.random.normal(loc=0.0,scale=f_scale, size=(timesteps,))
        mech.transient(f=f,parameter_set= kwargs.pop('parameter_set',None))
        #mech.mode_superpos(f)
    
    
    res=pyansys.read_binary(f"{jid}.rst")
    node_numbers=res.geometry['nnum']
    num_nodes=len(node_numbers)
    time_values=res.time_values
    dt = time_values[1]-time_values[0]
    time_values -= dt

        
    solution_data_info = res.solution_data_info(0)
    DOFS = solution_data_info['DOFS']
    
    uz = DOFS.index(3)
    nnum, all_disp = res.nodal_time_history('NSL')
    #print(all_disp)

    ydata = all_disp[:,np.where(nnum==20)[0][0],uz]
#     print(ydata)
    this_t = time_values
    ty=np.vstack((this_t, ydata))
    plot.figure()
    
    plot.gca().plot(this_t, ydata, color='black', **kwargs)
    plot.plot(this_t, ydata, ls='none', marker='+')
    plot.show()

    np.savetxt("/vegas/users/staff/womo1998/data_hadidi/{}.csv".format(jid), ty.T)
    with open("/vegas/users/staff/womo1998/data_hadidi/description_new.txt", "at") as f:
        f.write("{},\t{:1.3f},\t{:1.4f},\t{:1.5f}\n".format(jid,k_2,d,deltat))
        
def generate_student_nl(ansys, omega, zeta, d0, deltat=None,dt_fact=None,timesteps=None, num_cycles=None, f_scale=None, **kwargs):
    
    global jid
    oldjid= jid
    jid=str(uuid.uuid4()).split('-')[-1]
    ansys.filname(fname=jid, key=1)
    for file in glob.glob(f'/dev/shm/womo1998/{oldjid}.*'): 
        print(f'removing {file}')
        os.remove(file)
    print(jid)

    total, used, free = shutil.disk_usage("/dev/shm/")
    if free/total < 0.1:
        raise RuntimeError(f'Disk "/dev/shm/ almost full {used} of {total}')
    
    assert deltat is not None or dt_fact is not None
    assert timesteps is not None or num_cycles is not None
    
    m    = 1    # kg
    k_2  = omega**2*m/(1-zeta**2)# N/m
    d=zeta*(2*np.sqrt(k_2*m))
    
    f_max = np.sqrt((k_2)/m)/np.pi/2
    
    if dt_fact is None:
        dt_fact =deltat*f_max # \varOmega /2/pi
        print("\\varOmega",dt_fact*2*np.pi)
    elif deltat is None:
        deltat = dt_fact/f_max
    
    if dt_fact*2*np.pi > 0.1: print("Warning \\varOmega > 0.1")
    
    nl_ity=kwargs.pop('nl_ity',-0.5)
    
    if timesteps is None:
        timesteps = int(np.ceil(num_cycles/f_max/deltat))
    elif num_cycles is None:
        num_cycles = int(np.floor(timesteps*f_max*deltat))
    if d0 is not None:
        print("Simulating system with omega {:1.3f}, zeta {:1.3f}, d0 {:1.3f}. deltat {:1.5f}, dt_fact {:1.2f}, timesteps {}, num_cycles {}, nonlinearity {}".format(omega, zeta, d0, deltat, dt_fact, timesteps, num_cycles, nl_ity))
    elif f_scale is not None:
        print("Simulating system with omega {:1.3f}, zeta {:1.3f}, random excitation variance {:1.3f}. deltat {:1.5f}, dt_fact {:1.2f}, timesteps {}, num_cycles {}, nonlinearity {}".format(omega, zeta, f_scale, deltat, dt_fact, timesteps, num_cycles,nl_ity))
    
    if d0 is None:
        sigma_scale = 1 # 68–95–99.7 rule
        empiric_factor=1/2.5
        d_max = sigma_scale*f_scale/2/k_2/zeta*empiric_factor
    else:
        d_max = d0
        
    mech = Mechanical(ansys)
    voigt_kelvin=mech.voigt_kelvin(k=0, d=d)
    nonlinear = mech.nonlinear(nl_ity=nl_ity, d_max=d_max, k_lin=k_2)
    mass = mech.mass(m)
    mech.build_sdof(
               mass=mass, 
               voigt_kelvin = voigt_kelvin,
               nonlinear = nonlinear,
               d_init=d0
               )
    
    mech.modal(True)
    
    f_max = process(f'{jid}.rst', last_analysis=mech.last_analysis)[0].max()
    
    freqs, damping, mode_shapes = process(f'{jid}.rst', last_analysis=mech.last_analysis)
    print(f"Numeric solution: f={freqs} zeta={damping}")
    f_analytical = np.sqrt((k_2)/m)/np.pi/2
    zeta_analytical = d/2/np.sqrt(k_2*m)*100 # in percent of critical damping
    print(f"Analytic solution: f={f_analytical},zeta={zeta_analytical}")


    mech.timesteps=timesteps
    mech.deltat = deltat
    
    #print(timesteps,deltat)
    
    f=np.zeros(timesteps)
    if d0 is not None:
        f[0]=d0
        mech.transient(d=f,parameter_set= kwargs.pop('parameter_set',None))
    else:
        assert f_scale is not None
        f = np.random.normal(loc=0.0,scale=f_scale, size=(timesteps,))
        mech.transient(f=f,parameter_set= kwargs.pop('parameter_set',None))
        #mech.mode_superpos(f)
    
    
    res=pyansys.read_binary(f"{jid}.rst")
    node_numbers=res.geometry['nnum']
    num_nodes=len(node_numbers)
    time_values=res.time_values
    dt = time_values[1]-time_values[0]
    time_values -= dt

        
    solution_data_info = res.solution_data_info(0)
    DOFS = solution_data_info['DOFS']
    
    uz = DOFS.index(3)
    nnum, all_disp = res.nodal_time_history('NSL')
#     print(all_disp)

    ydata = all_disp[:,np.where(nnum==20)[0][0],uz]
#     print(ydata)
    this_t = time_values
    ty=np.vstack((this_t, ydata))
    plot.figure()
    
    plot.gca().plot(this_t, ydata, color='black', **kwargs)
    plot.axhline(d_max)
    plot.axhline(-d_max)
#     plot.plot(this_t, ydata, ls='none', marker='+')
    #plot.show()
    #print("{},\t{:1.3f},\t{:1.4f},\t{:1.5f},\t{:1.5f},\t{:1.3f}\n".format(jid,omega,zeta,d0,deltat,nl_ity))
    source_folder = "/vegas/scratch/womo1998/data_hadidi/datasets_nonlinear_"
    if d0 is not None:
        source_folder+='decay/'
    else:
        source_folder +='ambient/'
        
    np.savetxt(f"{source_folder}{jid}.csv", ty.T)
    np.savetxt(f"{source_folder}inp{jid}.csv",f)
    with open(f"{source_folder}description.txt", "at") as f:
        f.write("{},\t{:1.3f},\t{:1.4f},\t{:1.5f},\t{:1.5f},\t{:1.3f}\n".format(jid,k_2,d,d_max,deltat,nl_ity))
    

def generate_damping(ansys, omega, zeta, d0, fric_visc_rat=0, deltat=None,dt_fact=None,timesteps=None, num_cycles=None):
    global jid
    
    assert deltat is not None or dt_fact is not None
    assert timesteps is not None or num_cycles is not None
    
    assert fric_visc_rat>=0 and fric_visc_rat<=1
    m    = 1    # kg
    k_2  = omega**2*m# N/m
    d=zeta*(2*np.sqrt(k_2*m))
    f_max = np.sqrt((k_2)/m)/np.pi/2    
    
    if dt_fact is None:
        dt_fact =deltat*f_max # \varOmega /2/pi
        print("\\varOmega",dt_fact*2*np.pi)
    elif deltat is None:
        deltat = dt_fact/f_max
    
    if dt_fact*2*np.pi > 0.1: print("Warning \\varOmega > 0.1")
    
    if timesteps is None:
        timesteps = int(np.ceil(num_cycles/f_max/deltat))
    elif num_cycles is None:
        num_cycles = int(np.floor(timesteps*f_max*deltat))
    
    print("Simulating system with omega {:1.3f}, zeta {:1.3f}, d0 {:1.3f}, fric_visc_rat {:1.3f}, deltat {:1.5f}, dt_fact {:1.2f}, timesteps {}, num_cycles {}".format(omega, zeta, d0, fric_visc_rat, deltat, dt_fact, timesteps, num_cycles))
    
    mech = Mechanical(ansys)

    voigt_kelvin=mech.voigt_kelvin(k=k_2*(1-fric_visc_rat), d=d*(1-fric_visc_rat))

    
    # Computation of equivalent friction damping
    d_fric = d*fric_visc_rat
#     f_sl_equiv = k_2*d0/2-np.sqrt(np.pi*omega**2*k_2**2*d0**4-omega*d_fric*k_2)/2/np.sqrt(np.pi)/omega/d0
    s = d0
    Omega = omega
    k_1 = k_2*fric_visc_rat*100
    f_sl_equiv = s/2*(k_1-np.sqrt(k_1*(-np.pi*Omega*d_fric+k_1)))
    print(f"equivalent slip force {f_sl_equiv} at omega {omega} with d {d_fric}")
    coulomb = mech.coulomb(k_2=k_2*fric_visc_rat, k_1=k_1, f_sl= f_sl_equiv)

        
#     coulomb = mech.coulomb(k_2=k_2, k_1=k_2*100, f_sl= 0.94)
    mass = mech.mass(m)
    mech.build_sdof(
               mass=mass, 
               voigt_kelvin = voigt_kelvin,
               coulomb = coulomb,
               d_init=d0
               )
#     if coulomb is not None:
#         mech.modal(False, [coulomb])
#     else:
#         mech.modal(False)
    
#     freqs, damping, mode_shapes = process('file.rst', last_analysis=mech.last_analysis)
#     print(f"Numeric solution: f={freqs} zeta={damping}")
#     f_analytical = np.sqrt((k_2)/m)/np.pi/2
#     zeta_analytical = d/2/np.sqrt(k_2*m)*100 # in percent of critical damping
#     print(f"Analytic solution: f={f_analytical},zeta={zeta_analytical}")

    mech.timesteps=timesteps
    mech.deltat = deltat
    
    f=np.zeros(timesteps)
    if d0 is not None:
        f[0]=d0
        mech.transient(d=f,parameter_set=None)

        #mech.mode_superpos(f)
    
    
    res=pyansys.read_binary(f"{jid}.rst")
    node_numbers=res.geometry['nnum']
    num_nodes=len(node_numbers)
    time_values=res.time_values
    dt = time_values[1]-time_values[0]
    time_values -= dt

        
    solution_data_info = res.solution_data_info(0)
    DOFS = solution_data_info['DOFS']
    
    uz = DOFS.index(3)
    nnum, all_disp = res.nodal_time_history('NSL')
    #print(nnum)

    ydata = all_disp[:,np.where(nnum==20)[0][0],uz]
    this_t = time_values
    ty=np.vstack((this_t, ydata))
    #plot.figure()
    
    plot.gca().plot(this_t, ydata, ls='solid')
    #plot.plot(this_t, ydata, ls='none', marker='+')
#     plot.show()

     
#     id=str(uuid.uuid4()).split('-')[-1]
#     np.savetxt("/vegas/users/staff/womo1998/data_yamini/{}.csv".format(id), ty.T)
#     with open("/vegas/users/staff/womo1998/data_yamini/description.txt", "at") as f:
#         f.write("{},\t{:1.3f},\t{:1.4f},\t{:1.2f},\t{:1.2f},\t{:1.5f}\n".format(id,omega,zeta,d0,fric_visc_rat,deltat))
    

def generate_mdof(ansys, deltat=None,timesteps=None, num_cycles=None, **kwargs):
    
    assert deltat is not None 
    assert timesteps is not None or num_cycles is not None
    try: os.remove('file.rst')
    except: pass
    
    A1 = 0.454e-3
    I  = 0.19e-6
    r2 = 0.00005
    A2=np.pi*r2**2
    E=210e9
    
    
    
    m1 = 7850*(2*A1+np.sqrt(8)*A2)# kg
    m2 = 7850*A1+50
    
    k1= E*(3/2*I+1/2*A2)
    k2 =E*3/2*I 
    
    d = 0
    
    d0 = 0.1
    
    mech = Mechanical(ansys)
    voigt_kelvins=[]
    voigt_kelvins.append(mech.voigt_kelvin(k=k1, d=d))
#     voigt_kelvins.append(mech.voigt_kelvin(k=k_2, d=d))
    voigt_kelvins.append(mech.voigt_kelvin(k=k2, d=d))
    masses = []
    masses.append(mech.mass(m1))
#     masses.append(mech.mass(m))
    masses.append(mech.mass(m2))
    mech.build_mdof(
               masses=masses, 
               voigt_kelvins = voigt_kelvins,
               d_init=d0
               )
    
    mech.modal(True)
    
#     ansys.open_gui()
    frequencies, damping, mode_shapes = process('file.rst', last_analysis=mech.last_analysis)
    
    
    if timesteps is None:
        timesteps = int(np.ceil(num_cycles/min(frequencies)/deltat))
    
    num_cycles_min = int(np.floor(timesteps*min(frequencies)*deltat))
    num_cycles_max = int(np.floor(timesteps*max(frequencies)*deltat))
    
    print("Generating {} cycles at {:1.3f} Hz ({:1.3f} \% damping) and {} cycles at {:1.3f} Hz ({:1.3f} \% damping)@ deltat {:1.5f} ".format(num_cycles_min, min(frequencies),damping[frequencies.argmin()]*100, num_cycles_max, max(frequencies), damping[frequencies.argmax()]*100,deltat))
    
    
    mech.timesteps= timesteps
    mech.deltat = deltat
    
    f=np.zeros(timesteps)
    f[0]=d0
    mech.transient(d=f,parameter_set='AAM')
#     mech.transient(d=f,meth='G-alpha',parameter_set=0.5)
    
    res=pyansys.read_binary("file.rst")
    node_numbers=res.geometry['nnum']
    num_nodes=len(node_numbers)
    time_values=res.time_values
    dt = time_values[1]-time_values[0]
    time_values -= dt

    
    
    
    #print(DOFS)
    nnum, all_disp = res.nodal_time_history('NSL')
#     print(nnum, all_disp[:,:,2])
    nnum, all_vel = res.nodal_time_history('VEL')
#     print(nnum, all_vel[:,:,2])
    nnum, all_acc = res.nodal_time_history('ACC')
#     print(nnum, all_acc[:,:,2])
    
    solution_data_info = res.solution_data_info(0)
    DOFS = solution_data_info['DOFS']
    uz = DOFS.index(3)
    
    #ydata = all_disp[:,np.where(nnum==20)[0][0],uz]
    this_t = time_values
    #ty=np.vstack((this_t, ydata))
    
    deltaf = 0.1
    axes = kwargs.pop('axes')#, sharey='col')
    if axes is None:
        axes = plot.subplots(nrows = 3, ncols=2, sharex='col')[1]
#     for node in range(2,3):
#         axes[0,0].plot(this_t, all_disp[:,node,2], ls='solid', label='{:1.4f}: {}'.format(deltat, (node+1)*10))
#         axes[0,1].psd(all_disp[:,node,2],Fs=1/deltat, NFFT= int(1/2/deltat/deltaf))
#         axes[1,0].plot(this_t, all_vel[:,node,2], ls='solid', label='{:1.4f}: {}'.format(deltat, (node+1)*10))
#         axes[1,1].psd(all_vel[:,node,2],Fs=1/deltat, NFFT= int(1/2/deltat/deltaf))
#         axes[2,0].plot(this_t, all_acc[:,node,2], ls='solid', label='{:1.4f}: {}'.format(deltat, (node+1)*10))
#         axes[2,1].psd(all_acc[:,node,2],Fs=1/deltat, NFFT= int(1/2/deltat/deltaf))
    node = 2
    axes[0].plot(this_t, all_acc[:,node,2], ls='solid')
    axes[1].psd(all_acc[:,node,2],Fs=1/deltat, NFFT= int(1/2/deltat/deltaf), label='$f_s = {:1.0f}$ \\si{{\\hertz}}'.format(1/deltat))
    
    for freq in frequencies:
#         axes[0,1].axvline(freq, color='grey')
#         axes[1,1].axvline(freq, color='grey')
#         axes[2,1].axvline(freq, color='grey')
        axes[1].axvline(freq, color='grey')

        
    axes[0].set_xlim((0, 1/min(frequencies)*4))

#     axes[0,0].set_ylim((-1.1,1.1))
#     axes[0,1].grid(False)
#     axes[1,1].grid(False)
#     axes[2,1].grid(False)
    
    #plot.plot(this_t, ydata, ls='none', marker='+')
#     plot.show()
    
#     import uuid     
#     id=str(uuid.uuid4()).split('-')[-1]
#     np.savez("/vegas/scratch/womo1998/vib_data/{}.npz".format(id), this_t = this_t, all_disp = all_disp, all_vel=all_vel, all_acc=all_acc)
    

def generate_conti(ansys, deltat=None,timesteps=None, num_cycles=None, **kwargs):
    
    assert deltat is not None 
    assert timesteps is not None or num_cycles is not None
    try: os.remove('file.rst')
    except: pass
    
    
    parameters = {
                'L'         : 40,

                'E'         : 210e9,
                'A'         : 4.97e-3,
                'rho'       : 7850,
                'Iy'        : 15.64e-6,
                'Iz'        : 15.64e-6,
                
                'ky_nl'     : 0,
                'kz_nl'     : 0,
                'x_knl'     : 15,

                'm_tmd'     : 0,
                'ky_tmd'    : 0,
                'kz_tmd'    : 0,
                'dy_tmd'    : 0,
                'dz_tmd'    : 0,
                'x_tmd '    : 40,

            }
    initial = {'d0y'       : 0.1,
               'd0z'       : 0.1,
               'x_d0'      : 40}
    mech = Mechanical(ansys)
    

    mech.build_conti(parameters, Ldiv = 10, initial=initial
               )
    
    mech.modal(True)
    
#     ansys.open_gui()
    frequencies, damping, mode_shapes = process('file.rst', last_analysis=mech.last_analysis)
    
    
    if timesteps is None:
        timesteps = int(np.ceil(num_cycles/min(frequencies)/deltat))
    
    num_cycles_min = int(np.floor(timesteps*min(frequencies)*deltat))
    num_cycles_max = int(np.floor(timesteps*max(frequencies)*deltat))
    
    print("Generating {} cycles at {:1.3f} Hz ({:1.3f} \% damping) and {} cycles at {:1.3f} Hz ({:1.3f} \% damping)@ deltat {:1.5f} ".format(num_cycles_min, min(frequencies),damping[frequencies.argmin()]*100, num_cycles_max, max(frequencies), damping[frequencies.argmax()]*100,deltat))
    
    
    mech.timesteps= timesteps
    mech.deltat = deltat
    
    f=np.zeros(timesteps)
    f[0]=d0
    mech.transient(d=f,parameter_set='AAM')
#     mech.transient(d=f,meth='G-alpha',parameter_set=0.5)
    
    res=pyansys.read_binary("file.rst")
    node_numbers=res.geometry['nnum']
    num_nodes=len(node_numbers)
    time_values=res.time_values
    dt = time_values[1]-time_values[0]
    time_values -= dt

    
    
    
    #print(DOFS)
    nnum, all_disp = res.nodal_time_history('NSL')
#     print(nnum, all_disp[:,:,2])
    nnum, all_vel = res.nodal_time_history('VEL')
#     print(nnum, all_vel[:,:,2])
    nnum, all_acc = res.nodal_time_history('ACC')
#     print(nnum, all_acc[:,:,2])
    
    solution_data_info = res.solution_data_info(0)
    DOFS = solution_data_info['DOFS']
    uz = DOFS.index(3)
    
    #ydata = all_disp[:,np.where(nnum==20)[0][0],uz]
    this_t = time_values
    #ty=np.vstack((this_t, ydata))
    
    deltaf = 0.1
    axes = kwargs.pop('axes')#, sharey='col')
    if axes is None:
        axes = plot.subplots(nrows = 3, ncols=2, sharex='col')[1]
#     for node in range(2,3):
#         axes[0,0].plot(this_t, all_disp[:,node,2], ls='solid', label='{:1.4f}: {}'.format(deltat, (node+1)*10))
#         axes[0,1].psd(all_disp[:,node,2],Fs=1/deltat, NFFT= int(1/2/deltat/deltaf))
#         axes[1,0].plot(this_t, all_vel[:,node,2], ls='solid', label='{:1.4f}: {}'.format(deltat, (node+1)*10))
#         axes[1,1].psd(all_vel[:,node,2],Fs=1/deltat, NFFT= int(1/2/deltat/deltaf))
#         axes[2,0].plot(this_t, all_acc[:,node,2], ls='solid', label='{:1.4f}: {}'.format(deltat, (node+1)*10))
#         axes[2,1].psd(all_acc[:,node,2],Fs=1/deltat, NFFT= int(1/2/deltat/deltaf))
    node = 2
    axes[0].plot(this_t, all_acc[:,node,2], ls='solid')
    axes[1].psd(all_acc[:,node,2],Fs=1/deltat, NFFT= int(1/2/deltat/deltaf), label='$f_s = {:1.0f}$ \\si{{\\hertz}}'.format(1/deltat))
    
    for freq in frequencies:
#         axes[0,1].axvline(freq, color='grey')
#         axes[1,1].axvline(freq, color='grey')
#         axes[2,1].axvline(freq, color='grey')
        axes[1].axvline(freq, color='grey')

        
    axes[0].set_xlim((0, 1/min(frequencies)*4))

#     axes[0,0].set_ylim((-1.1,1.1))
#     axes[0,1].grid(False)
#     axes[1,1].grid(False)
#     axes[2,1].grid(False)
    
    #plot.plot(this_t, ydata, ls='none', marker='+')
#     plot.show()
    
#     import uuid     
#     id=str(uuid.uuid4()).split('-')[-1]
#     np.savez("/vegas/scratch/womo1998/vib_data/{}.npz".format(id), this_t = this_t, all_disp = all_disp, all_vel=all_vel, all_acc=all_acc)
    



def accuracy_study(ansys):
        
    timesteps=1000
    results = np.zeros((40,8))
    
    if True:
        for i in range(40):
            dt_fact = (i+1)*0.005
            
            mech = Mechanical(ansys)
    
            fact=1e5
            m    = fact/1    # kg
            k_2  = 1*fact*4*np.pi**2 # N/m
            ampl = fact/10   # N
            d=0.00*2*np.sqrt(k_2*m)
            
            voigt_kelvin=mech.voigt_kelvin(k=k_2, d=d)
            mass = mech.mass(m)
            mech.build_sdof(
                       mass=mass, 
                       voigt_kelvin = voigt_kelvin,
                       d_init=1
                       )
            
            mech.modal()
            
            f_max = process('file.rst', last_analysis=mech.last_analysis).max()
            f_analytical = np.sqrt((k_2)/m)/np.pi/2
            zeta_analytical = d/2/np.sqrt(k_2*m)*100 # in percent of critical damping
            print(f_analytical,zeta_analytical)

            deltat = dt_fact/f_max
            dt_fact =deltat/f_max            
            num_cycles = 16
            timesteps = int(np.ceil(num_cycles/dt_fact))
            
            mech.timesteps=timesteps
            mech.deltat = deltat
            
            f=np.zeros(timesteps)
            f[0]=1
            mech.transient(d=f)
            
            popt,perr = response_frequency('file.rst')
            results[i,:4]=popt
            results[i,4:]=perr
            
    
        ansys.exit()
        np.save('newmark_errors.npz', results)
        
    results = np.load('newmark_errors.npz.npy')
    dt_fact = np.linspace(0.005,0.2,40)
    #print('R: {:1.3f} m, zeta: {:1.3f} \%, f_d: {:1.4f} Hz, phi: {:1.4f}'.format(popt[0], popt[1]*100,  popt[2]/2/np.pi, popt[3]*180/np.pi))
    
    #confidence intervals
    sqrt_ts = np.sqrt(np.ceil(16/dt_fact))
    tppf = [scipy.stats.t.ppf(0.95,int(timesteps)) for timsteps in np.ceil(16/dt_fact)]
    #scaling
    results[results[:,0]<0,0]*=-1
    results[:,0]+=-0.5
    results[:,0]/=0.5
    
    results[:,4]/=0.5**2
    results[:,4]/=sqrt_ts
    results[:,4]*=tppf
    
    
    results[:,1]*=100
    
    results[:,5]*=100**2
    results[:,5]/=sqrt_ts
    results[:,5]*=tppf
    
    
    results[:,2]/=2*np.pi
    
    results[:,6]/=(2*np.pi)**2
    results[:,6]/=sqrt_ts
    results[:,6]*=tppf
    
    
    results[:,3]=np.arcsin(np.sin(results[:,3]))
    results[:,3]*=180/np.pi    
    results[results[:,3]<=0,3]*=-1
    
    results[:,7]*=(180/np.pi)**2
    results[:,7]/=sqrt_ts
    results[:,7]*=tppf
    
    with matplotlib.rc_context(rc=print_context_dict):
        fig,axes = plot.subplots(2,2, sharex=True)
        print(fig.get_size_inches())
        axes = axes.flatten()
        ylabels = ['$\Delta R$ [\si{\percent}]', '$\Delta \zeta [\si{\percent}]$', '$\Delta f [\si{\percent}]$', '$\Delta \phi [\si{\degree}]$']
        
        for j in range(4):
            
            axes[j].errorbar(dt_fact, results[:,j], yerr=results[:,j+4], errorevery=2)
            axes[j].set_ylabel(ylabels[j])
            #axes[j].plot(dt_fact[:], results[:,j], marker='+')
        axes[2].set_xlabel('Frequency ratio $\\nicefrac{f_{\\text{max}}}{f_s} [-]$')
        axes[3].set_xlabel('Frequency ratio $\\nicefrac{f_{\\text{max}}}{f_s} [-]$')        
        plot.subplots_adjust(left=0.085, right=0.985, top=0.985, bottom=0.085, hspace=0.1, wspace=0.18)
        plot.show()
        
def verify_friction(ansys):
    ########
    # Verification code for friction / generate_damping
    ########
    plot.subplot()
    zeta = 10/100
    num_cycles = 8
    d0=1
    omega=2
    generate_damping(ansys, omega=omega, zeta=zeta, fric_visc_rat=0, d0=d0, deltat=0.03, num_cycles=8)
    generate_damping(ansys, omega=omega, zeta=zeta, fric_visc_rat=0.5, d0=d0, deltat=0.03, num_cycles=8)    
    generate_damping(ansys, omega=omega, zeta=zeta, fric_visc_rat=1, d0=d0, deltat=0.03, num_cycles=8)
    plot.show()
    
    
def stepsize_example(ansys): 
#########
# Example showing effect of stepsizes
#########
    with matplotlib.rc_context(rc=print_context_dict):
        plot.figure(tight_layout=True)
        zeta_=0/100
        omega_=1
        for method, gamma_, beta_, linestyle in [('Average Acceleration',0.5,0.25,'solid'),
                                                     ('Linear Acceleration',0.5,1/6, 'dashed'),
    #                                                  ('Central Difference',0.5,0,'dotted'),
                                                     ('$\\gamma=0.6, \\beta=0.3025$',0.6,0.3025,'dashdot'),
                                                     ('$\\gamma=0.9142, \\beta=0.5$',np.sqrt(2)-1/2,1/2, (0, (3, 1, 1, 1, 1, 1))),]:
            generate_student(ansys, omega=omega_, zeta=zeta_, d0=1, dt_fact=0.1, num_cycles=4, parameter_set=(gamma_,beta_), label=method, linestyle=linestyle)
            
            
    #     generate_student(ansys, omega=1, zeta=0, d0=1, dt_fact=0.001, num_cycles=1, parameter_set='LAM', color='grey', label='Analytic')
        t= np.linspace(0, 20, 3000)
        ydata = free_decay(t, R=1/2, zeta=zeta_, omega_d=omega_, phi=0)
        plot.plot(t,ydata, color='grey', label='Analytical')
#         plot.plot(t,2*1/2*np.exp(-zeta_*omega_/(np.sqrt(1-zeta_**2))*t), color='grey', label='Envelope', ls='dotted')
#         plot.plot(t,-2*1/2*np.exp(-zeta_*omega_/(np.sqrt(1-zeta_**2))*t), color='grey', label='Envelope', ls='dotted')
        plot.xlim((0,20))
        plot.ylim((-1.1,1.1))
        datacursor(formatter='{label}'.format, bbox=None, draggable=True, display='multiple')
    #     plot.legend()
        plot.xlabel('Time [\\si{\\second}]')
        plot.ylabel('Displacement [\\si{\\metre}]')
        plot.show()

#     plot.figure()
#     plot.subplot()
#     import sympy
#     sympy.init_printing()
#     h,beta,gamma,zeta,omega,Omega,eta=sympy.symbols('h \\beta \gamma \zeta \omega \Omega \eta', real=True, positive=True)
#       
#     A1=sympy.Matrix([[1,0, -h**2*beta],[0, 1, -h*gamma],[omega**2,2*zeta*omega, 1]])
#     #with this one, Eq. 3.17 gives an oscillating response
#     A2=sympy.Matrix([[1,h,h**2*(0.5-beta)],[0,1,h*(1-gamma)],[0,0,0]])
#     A=A1.solve(A2)
#     dt = 2*np.pi*0.1
#     timesteps =30
#     t = np.linspace(0,stop=dt*timesteps,num=timesteps)
#     for beta_ in [1/6,1/4,0]:
#         Anp = np.array(A.subs(omega,1).subs(zeta,0).subs(h,dt).subs(gamma,0.5).subs(beta,beta_)).astype(np.float64)
#         res = np.zeros((3,timesteps+1))
#         if False:#not beta_:
#             res[0,1]=1 #u0
#             res[0,0]=res[0,1]-dt*res[1,1]+dt**2/2*res[2,1] #u-1
#             res[2,1]=-1 #dot{u}_0
#             k=1
#             m=1
#             khat = m/dt**2
#             a=m/dt**2
#             b=k-2*m/dt**2
#             for i in range(1,timesteps+1):
#                 phat=-a*res[0,i-2]-b*res[0,i-1]
#                 res[0,i] = phat/khat
#             res/=np.pi/2
#         else:
#             res[0,0]=1
#             for i in range(1,timesteps+1):
#   
#                 res[:,i]= Anp.dot(res[:,i-1])
#         if False:#not beta_:
#             res = res[:,1:]
#         else:
#             res = res[:,:-1]
#         plot.plot(t,res[0,:], label=f'{beta_}')
#           
#     dt = 2*np.pi*0.001
#       
#     timesteps =3000
#     t=np.linspace(0,stop=dt*timesteps,num=timesteps)
#     Anp = np.array(A.subs(omega,1).subs(zeta,0).subs(h,dt).subs(gamma,0.25).subs(beta,1/6)).astype(np.float64)
#     res = np.zeros((3,timesteps+1))
#     res[0,0]=1
#     for i in range(1,timesteps+1):
#         res[:,i]= Anp.dot(res[:,i-1])
#     plot.plot(t,res[0,:-1])
#     plot.xlim((0,20))
#     plot.ylim((-1.1,1.1))
#     plot.legend()
#     plot.show()
    
def spectr_shift_example(ansys):
    with matplotlib.rc_context(rc=print_context_dict):
        fig, axes = plot.subplots(nrows = 2, ncols=1)
        generate_mdof(ansys, deltat=1/200, num_cycles=8,axes=axes)
        generate_mdof(ansys, deltat=1/100, num_cycles=8, axes=axes)
        generate_mdof(ansys, deltat=1/50, num_cycles=8, axes=axes)
        
        
        axes[1].axvline(-1, color='grey', label="Theoretical")
            
        axes[0].set_ylabel('Acceleration [\\si{\\metre\\per\\square\\second}]')
        axes[0].set_xlabel('Time [\\si{\\second}]')
        axes[1].set_ylabel('PSD [\\si{\\decibel\\per\\hertz}]')
        axes[1].set_xlabel('Frequency [\\si{\\hertz}]')
        axes[1].grid(0)
        axes[1].set_xlim((0,100))
        axes[1].legend()
        fig.subplots_adjust(left=0.110, bottom=0.125, right=0.970, top=0.960, hspace=0.340)
        plot.show()
        
        
def student_data(ansys):
###
# Damping Data Yamini
###
#     
#     for i in range(30):
#         omega = np.random.random()*14+1
#         zeta = (np.random.random()*(10-0.1)+0.1)/100
#         dt_fact = np.random.random()*(0.015-0.001)+0.001
#         num_cycles = np.random.randint(3,20)
#         if i < 10:
#             fric_visc_rat = 0
#         elif i < 20:
#             fric_visc_rat = 1
#         else:
#             fric_visc_rat = np.random.random()
#         d0 = np.random.random()*10
#           
#         generate_damping(ansys, omega, zeta, d0, fric_visc_rat, dt_fact=dt_fact, num_cycles=num_cycles)
#          
#     plot.show()

###
# Damping Data Hadidi
###    
    #generate_damping(ansys, omega=1, zeta=0.01, m=1, fric_visc_rat=0, nl_ity=0, d0=1, f_scale=None, dt_fact=0.001, savefolder=None)
    generate_damping(ansys, 1, 1/100, 1, 0, dt_fact=0.001, num_cycles=5)
    #generate_student_nl(ansys, 1, 1/100, 1, dt_fact=0.001, num_cycles=5, nl_ity = -0.5)
    plot.show()
    
    for i in range(10):
        omega = np.random.random()*14+1
        zeta = np.random.random()*(10-0.1)+0.1
        dt_fact = np.random.random()*(0.015-0.001)+0.001
        num_cycles = np.random.randint(300,2000) #ambient
        #num_cycles = np.random.randint(3,20) # free decay
        d0=np.random.random()*100
        f_scale = np.random.random()*10
        nl_ity = np.random.random()-0.5
        
        #generate_student(ansys, omega, zeta/100, 1, dt_fact=dt_fact, num_cycles=num_cycles, f_scale=f_scale)
        #generate_student(ansys, omega, zeta/100, None, dt_fact=dt_fact, num_cycles=num_cycles, f_scale=f_scale)
        
        #generate_student_nl(ansys, omega, zeta/100, d0, dt_fact=dt_fact, num_cycles=num_cycles, f_scale=f_scale, nl_ity = nl_ity)
        generate_student_nl(ansys, omega, zeta/100, None, dt_fact=dt_fact, num_cycles=num_cycles, f_scale=f_scale, nl_ity = nl_ity)
        #if not i%10:
    plot.show()

def identify_student():
    source_num = 0
    
    source_folder = ['/vegas/scratch/womo1998/data_hadidi/datasets/',
                     '/vegas/scratch/womo1998/data_hadidi/datasets_ambient/',
                     '/vegas/scratch/womo1998/data_hadidi/datasets_nonlinear_decay/',
                     '/vegas/scratch/womo1998/data_hadidi/datasets_nonlinear_ambient/'
                     ][source_num]
    
    snr = 100
    decimate=0
    single=False
    
    id_res = []
    id_inp = []
    fig, axes = plot.subplots(nrows=5, ncols=8)
    axes = axes.flatten()
    #with open('/vegas/users/staff/womo1998/data_hadidi/description_new.txt','tr') as descr:
    with open(f'{source_folder}description.txt','tr') as descr:#, open(f'{source_folder}description_new.txt','tw') as descr_new:
        descr.readline()
        for i,line in enumerate(descr):
            print(line)
            m=1
            if source_num == 0:
                id,omega,zeta,R,deltat = [float(s.strip()) if j>0 else s.strip() for j,s in enumerate(line.split(',')) ]
                k=omega**2*m
                d=zeta*(2*np.sqrt(k*m))
                #descr_new.write("{},\t{:1.3f},\t{:1.4f},\t{:1.5f}\n".format(id,k,d,R,deltat))
            elif source_num == 1:
                id,omega,zeta,deltat = [float(s.strip()) if j>0 else s.strip() for j,s in enumerate(line.split(',')) ]
                k=omega**2*m
                d=zeta*(2*np.sqrt(k*m))
                #descr_new.write("{},\t{:1.3f},\t{:1.4f},\t{:1.5f}\n".format(id,k,d,deltat))
            elif source_num == 2:
                id,omega,zeta,R,deltat,nl_ity = [float(s.strip()) if j>0 else s.strip() for j,s in enumerate(line.split(',')) ]
                k=omega**2*m
                d=zeta*(2*np.sqrt(k*m))
                #descr_new.write("{},\t{:1.3f},\t{:1.4f},\t{:1.5f},\t{:1.5f},\t{:1.3f}\n".format(id,k,d,R,deltat,nl_ity))
            elif source_num == 3:
                id,omega,zeta,d_max,deltat,nl_ity= [float(s.strip()) if j>0 else s.strip() for j,s in enumerate(line.split(',')) ]
                k=omega**2*m
                d=zeta*(2*np.sqrt(k*m))
                #descr_new.write("{},\t{:1.3f},\t{:1.4f},\t{:1.5f},\t{:1.5f},\t{:1.3f}\n".format(id,k,d,d_max,deltat,nl_ity))
                
        #continue

            #print(f'zeta: {zeta*100} \%, omega: {omega} Hz')
            ty=np.loadtxt(f'{source_folder}{id}.csv')
            #print(deltat, ty[2,0]-ty[1,0])
            if snr:
                # SNR=u_eff,sig^2/u_eff,noise^2 (wikipedia: Signal-Rausch-Verhältnis: Rauschspannungsverhältnis)
                # u_eff,noise^2 = u_eff,sig^2/SNR
                ty[:,1] += np.random.randn(ty.shape[0])*np.sqrt(ty[:,1].var()*snr) # variance equals rms**2 here because it is a zero-mean process
                
            if decimate:
                ty=ty[::decimate,:]
            if 'ambient' in source_folder:
                if source_num==3:
                    d=max(np.abs(ty[:,1]))
                    k*=(1+nl_ity*((d/d_max)**2-1))
                ydata = scipy.signal.correlate(ty[:,1], ty[:,1], mode='full', method='direct')
                ydata = ydata[ydata.shape[0]//2:,][:1000]
                xdata = ty[:1000,0]
            else:
                ydata = ty[:,1]
                xdata = ty[:,0]
            #popt, pcov = scipy.optimize.curve_fit(f=free_decay, xdata = ty[:1000,0], ydata=corr[corr.shape[0]//2:,][:1000], p0=[0.5,0.05,2*np.pi,0])#,  bounds=[(-1,0,0,0),(1,1,np.pi/dt,2*np.pi)])
            try:
                popt, pcov = scipy.optimize.curve_fit(f=free_decay, xdata = xdata, ydata=ydata, p0=[0.5,0.05,2*np.pi,0])#,  bounds=[(-1,0,0,0),(1,1,np.pi/deltat,2*np.pi)])
            except Exception as e:
                print('ID failed', e)
                popt = [0,0,0,0]
                pcov=[np.inf,np.inf,np.inf,np.inf]
            
            perr = np.sqrt(np.diag(pcov))
            #print(' zeta: {:1.3f}+-{:1.3f} \%, omega: {:1.4f}+-{:1.3f} Hz\n\n'.format( popt[1]*100, perr[1]*100, popt[2], perr[2]))
            id_res.append((popt[2],popt[1],xdata[2]-xdata[1]))
            
            t_synth= np.linspace(0,xdata[-1],1000)
            #synth=free_decay(ty[:1000,0], *popt)
            synth=free_decay(t_synth, *popt)
            
            zeta=zeta
            omega = np.sqrt(k/m)*np.sqrt(1-zeta**2)
            
            deltat = deltat
            id_inp.append((omega,zeta,deltat))
            print(omega,zeta,deltat)
            
            if single:
                plot.figure()
                plot.plot(xdata,ydata, ls='none', marker=',')
                plot.plot(t_synth, synth)
                plot.show()
                continue
            #axes[i].plot(ty[:1000,0],corr[corr.shape[0]//2:,][:1000], ls='none', marker='+')
            axes[i].plot(xdata,ydata, ls='none', marker=',')
            axes[i].plot(t_synth, synth)
#             if len(id_res)>2:
#                 break
            #if i >4:break
    id_res = np.array(id_res)
    id_inp = np.array(id_inp)
    labels = ['Omega','Zeta','Delta_t']
    for i in range(2):
        plot.figure()
        plot.plot(id_inp[:,i],id_res[:,i], ls='none',marker='.')
        plot.xlabel(labels[i]+'_simulated')
        plot.ylabel(labels[i]+'_identified')
    
    plot.show()

if __name__ == '__main__':
    
    '''
    Static response
        Nonlinear Stiffness
    Quasi-Static
        Nonlinear Stiffness
        Friction Response
    Modal Analysis
    Dynamic Response (Nyquist Check, Amplitude Check)
       Initial Displacement -> Free Vibration
           Numerical Damping
           Prescribed Damping
       Harmonic Loading
           Nonlinear Stiffness
           Damping Hysteresis
       Impulse
           IRF
       Sweep / Random Loading
           FRF
    
    MDOF
    Pass all verifications
    Random Field Loading
    Mesh Size
    Meas Nodes Definition
    Full Parametrization
    '''
    os.makedirs('/dev/shm/womo1998/', exist_ok=True)
    os.chdir('/dev/shm/womo1998/')

#     import glob
#     filelist = glob.glob('/dev/shm/womo1998/*')
#     for file in filelist:
#         os.remove(file)

    #identify_student()
    
    global jid
    jid=str(uuid.uuid4()).split('-')[-1]
    
    ansys = pyansys.Mapdl(exec_file='/vegas/apps/ansys/v190/ansys/bin/ansys190', 
                          run_location='/dev/shm/womo1998/', jobname=str(jid), override=True, loglevel='ERROR',nproc=1,
                          log_apdl=False, log_broadcast=False,
                          prefer_pexpect=True)

    student_data(ansys)
    #plot.interactive(True)
    #accuracy_study(ansys)
    #stepsize_example(ansys)
    

