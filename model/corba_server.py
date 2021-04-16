import sys
import time
import os
from ansys_corba import CORBA
import subprocess

#os.chdir('/vegas/scratch/womo1998/modal_uq/')
os.chdir('/dev/shm/womo1998/')

# edit this to match your ansys exe
ansys_loc = '/vegas/apps/ansys/v190/ansys/bin/ansys190'

# ansys apdl logging here:
logfile = 'mapdl_broadcasts.txt'
if os.path.isfile(logfile):
    os.remove(logfile)

# make temporary input file to stop ansys from prompting the user
with open('tmp.inp', 'w') as f:
    f.write('FINISH')

# start ANSYS
command = '%s -aas -i tmp.inp -o out.txt -b' % ansys_loc
subprocess.Popen(command.split(), stdout=subprocess.PIPE)