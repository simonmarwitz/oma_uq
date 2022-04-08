#!/bin/bash
#SBATCH --mail-user=simon.jakob.marwitz@uni-weimar.de
#SBATCH --mail-type=ALL
#SBATCH --partition=short                              	# partitions: vegas (default), small, short, medium, long
#SBATCH --cpus-per-task=1 				# ansys transient is single threaded anyway
#SBATCH --threads-per-core=1	                        # number of threads per core
#SBACTH --ntasks=1
#SBATCH --mem=3875					# 3875 (16 jobs per node = 62 GB)
#SBATCH --array=24-50					# with task limit %N to avoid overusing licenses
#SBATCH --output=/vegas/scratch/womo1998/modal_uq/log/conv_polyUQ_%A_%a.out # %A jobid, %a taskid
#SBATCH --job-name=conv_polyUQ

module load python/intelpython3.7 > /dev/zero  2>&1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK # limit MKL to the number of cpus allocated to avoid hyperthreading
PYTHONPATH=/vegas/users/staff/womo1998/Projects/2019_OMA_UQ/code/
export PYTHONPATH
export PYTHONUNBUFFERED=1

python /vegas/users/staff/womo1998/Projects/2019_OMA_UQ/code/uncertainty/convergence_PolyUQ.py 4 ${SLURM_ARRAY_TASK_ID} 