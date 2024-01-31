#!/bin/bash
#BSUB -q highmem
#BSUB -R "span[hosts=1]"
#BSUB -oo /usr/scratch4/sima9999/work/modal_uq/uq_oma_a/logs/job_%J.log
#BSUB -eo /usr/scratch4/sima9999/work/modal_uq/uq_oma_a/logs/job_%J.err
#BSUB -J frf_multi_memmap
#BSUB -L /usr/bin/bash
#BSUB -n 32
#BSUB -R 'rusage[mem=819200]' # in MB


# start with argument, passing the ip adress of the head node
source /home/sima9999/.bashrc
# zconda activate /usr/wrk/people9/sima9999/my-python
# conda activate py311
PYTHONPATH=/home/sima9999/code/:/home/sima9999/git/pyOMA/
export PYTHONPATH
export PYTHONUNBUFFERED=1
export MKL_NUM_THREADS=2
export OMP_NUM_THREADS=2
export NUMBA_NUM_THREADS=2
python /home/sima9999/code/examples/UQ_OMA.py 9