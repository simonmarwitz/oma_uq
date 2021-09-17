# trainer.py

# import scipy.linalg, scipy.integrate, scipy.optimize, scipy.signal
# import PyQt5.QtCore, PyQt5.QtGui
# import pyansys
from collections import Counter
import os
import time
import ray
num_cpus = 32#int(sys.argv[1])

ray.init(address='auto', _redis_password='5241590000000000')
#ray.init(address=os.environ["ip_head"])

print("Nodes in the Ray cluster:")
print(ray.nodes())

@ray.remote
def f():
    time.sleep(1)
    return os.uname()[1]

@ray.remote
def import_tester_fun():
    # The following imports must be on top of the file before import ray
    import scipy.linalg, scipy.integrate, scipy.optimize, scipy.signal
    import PyQt5.QtCore, PyQt5.QtGui
    from ansys.mapdl.core import launch_mapdl
    from ansys.dpf.core import Model
    import ansys.mapdl.core._version
    #import pyansys
    
    print(scipy.__version__, PyQt5.QtCore.QT_VERSION_STR, ansys.mapdl.core._version.version_info)
    import seaborn
    
    
    
    return os.uname()[1]
#print(fun_kwargs)
futures= []
for i in range(60):
    worker_ref = import_tester_fun.remote()
    
    futures.append(worker_ref)

futures = set(futures)

while True:
    ready, wait = ray.wait(
        list(futures), num_returns=min(len(futures), 25))

    ret_sets = ray.get(ready)
    if not ret_sets:
        print("All jobs already computed!")
        break
    
    for name in ret_sets:
        print(name)
        
    futures.difference_update(ready)


    