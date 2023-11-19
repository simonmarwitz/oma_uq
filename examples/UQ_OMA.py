
import sys
import os
sys.path.append("/usr/wrk/people9/sima9999/code/")
sys.path.append("/vegas/users/staff/womo1998/Projects/2019_OMA_UQ/code/")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from model.turbulent_wind import *
    
# Spatial domain grid
x_grid = np.arange(1,201,1)


if False:
    # 200 x 2**17 samples take 1min 42 (CPSD assembly) + 2min 20s (CPSD decomposition) + 1min 59s(Fourier coefficients assembly) + 1.4s (IFFT) = 6min 2s; 
    # peak at 270 GB RAM
    
    # Frequency domain grid
    
    # wind does not have to be generated up to higher frequencies, 
    # as the energy in these bands is negligible
    fs_w = 10 # Hz, Wind sample rate
    # sample rate is adjusted by zero padding the fft
    fs_m = 100 # Hz, Model sample rate
    
    duration = 360 # seconds
    
    if duration is None:
        N = 2**17
        duration = N / fs_w
    else:
        N = int(duration * fs_w)
    
    N_m =  int(duration * fs_m)
    
    
    f_w = np.fft.rfftfreq(N, 1/fs_w)[np.newaxis,:]
    
    # Geländekategorie I - IV
    category=3
    z_min = [None,2,4,8,16][category]
    z_0 = [None, 0.01,0.05,0.3,1.05][category] # Rauigkeitslänge
    alpha = [None,0.12,0.16,0.22, 0.3][category]  # Profilexponent
    
    # Windzone 1 - 4
    # v_b = [None, 22.5, 25.0, 27.5, 30][zone] # m/s Basiswindgeschwindigkeit (v_b = v_b0 in DE vgl. NA S. 5)
    v_b = scipy.stats.weibull_min.rvs(2.28,scale=5.64, size=1)
    
    # Compute basic wind parameters according to DIN EN 1994-1-4 and corresponding NA    
    # Werte nach Tab. NA.B.2
    c_r = 0.19*(z_0/0.05)**0.07*np.log(10/z_0)*(x_grid/10)**alpha # Rauigkeitsbeiwert nach NA.1
    c_o = 1 # Topographiebeiwert nach Anhang C (?)
    v_m = c_r*c_o*v_b # mittlere Windgeschwindigkeit
    
    # v_m = 1*v_b*(x_grid/10)**0.16 # mittlere Windgeschwindigkeit (obere Formel ergaebe Faktor 1.0066803, daher die geringen Abweichungen)
    Iv_fact = [None, 0.14,0.19,0.28,0.43][category]
    I_v = Iv_fact*(x_grid/10)**(-alpha) # Turbulenzintensität [% der mittleren Windgeschwindigkeit] (? siehe unten)
    Iv_min = [None, 0.17, 0.22, 0.29, 0.37][category]
    I_v[x_grid<=z_min] = Iv_min
    
    sigma_v = I_v * v_m # Standardabweichung der Turbulenz nach Gl. 4.7
    
    eps = [None, 0.13, 0.26, 0.37, 0.46][category] # Exponent \epsilon nach Tabelle NA.C.1
    L = (x_grid/300)**eps * 300 # Integrallängenmaß nach Gl. NA.C.2
    L[x_grid<=z_min] = (z_min/300)**eps * 300
    
    c_uj, c_vj = spectral_wind_field(x_grid, f_w, 
                                     L, v_m, sigma_v, C_uz=10, C_vz=7, C_wz=4, 
                                     seed=None)
    
    u_j, v_j = temporal_wind_field(c_uj, c_vj, N_m)
    
    
    # In[35]:
    
    
    # with mean wind
    # F_uj, F_vj = force_wind_field(u_j + v_m[:,np.newaxis], v_j, delta_x=x_grid[1]-x_grid[0], 
    #                               b=1.85, cscd=1.0, cf=2.36124, rho=1.25)
    # without mean wind
    F_uj, F_vj = force_wind_field(u_j, v_j, delta_x=x_grid[1]-x_grid[0], 
                                  b=1.85, cscd=1.0, cf=2.36124, rho=1.25)
    
    
    # In[36]:
    
    
    plot_windfield(u_j, v_j, F_uj, duration=duration)
    plot_windfield(F_vj=F_vj, duration=duration)
    
    
    # # Structural Model generation
    
# 

# In[69]:


from model.mechanical import Mechanical, MechanicalDummy

add_mass = 58.477412
zeta = 0.008117
dD = 284.903024
mD = 800
kD = 1025.48
num_nodes = 201
num_modes = 14

E = 2.1e11

A = 0.034251
Iy = 0.013353
Iz = 0.013716
Iyz = 0

A_wire = 0.000785
rho = 7850
N_wire = 67978.166088

meas_locs=x_grid

# approximately similar second moments of area are achieved with a truss 
# consisting of 4 L profiles 0.226x0.226x0.03 m in the corners at a distance of 1.416 m 
# additionally, the diagonal and horizontal bracings have to be considered for the weight
# they add a weight of 0.016005*rho per meter, 
# so we need a cross-section area of 0.034634 m² to achieve the same weight per meter
# use a 200x200x20 L Profile at a 1.85 m distance slightly lower area, but similar moments of area
# (also see photos of Sender Aholming)
# use only the parts due to Steiner's theorem and reduce to 80 % due to truss flexibility
# A_L = (2*a*t - t*t)*4 = 0.0304
# Iy_L = 0.8*A_L*(L/2)**2 = 0.020809
# Iz_L = 0.8*A_L*(L/2)**2 = 0.020809
# also see excel workbook 'pipe_to_truss_conversion.xlsx'

# rotate cross section about angle alpha
alpha=0
alpha = alpha * 2 * np.pi / 360
rIy = 0.5 * (Iy + Iz) + 0.5 * (Iy - Iz) * np.cos(2 * alpha) + Iyz * np.sin(2 * alpha)
rIz = 0.5 * (Iy + Iz) - 0.5 * (Iy - Iz) * np.cos(2 * alpha) - Iyz * np.sin(2 * alpha)
rIyz = -0.5 * (Iy - Iz) * np.sin(2 * alpha) + Iyz * np.cos(2 * alpha)

Aeq = A_wire/(1 + (A_wire*rho*9.819*70/N_wire)**2*E*A_wire/12/N_wire)

keq = (E * Aeq * (70**2 / (70**2 + 160**2)) + N_wire)/np.sqrt(70**2 + 160**2)

rho_pipe = rho + add_mass / A

struct_parms = {
        'L'         : 200,

        'E'         : E,
        'A'         : A,
        'rho'       : rho_pipe,
        'Iy'        : rIy,
        'Iz'        : rIz,
        'Iyz'       : rIyz,

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


working_dir='/dev/shm/womo1998/'
global ansys
if 'ansys' not in globals():
    ansys = Mechanical.start_ansys(working_dir=working_dir)
mech = Mechanical(ansys=ansys, wdir=working_dir)

mech.build_conti(struct_parms, Ldiv=num_nodes, damping=zeta, 
                 num_modes=num_modes, meas_locs=meas_locs)
ansys.open_gui()
mech.modal()



zeros = np.zeros((1,N_m))
this_N = 10000#N_m
t, [d,v,a] = mech.transient(fy=np.vstack((F_uj, zeros, zeros))[:,:this_N].T, fz=np.vstack((F_vj, zeros, zeros))[:,:this_N].T, deltat=1/fs_m, timesteps=this_N)


# In[50]:


plt.matshow(d[:,:,2].T, origin='lower', aspect='auto', extent=(0, duration, x_grid[0], x_grid[-1]),)


# In[49]:


plt.matshow(F_uj[:,:this_N], origin='lower', aspect='auto', extent=(0, duration, x_grid[0], x_grid[-1]),)


