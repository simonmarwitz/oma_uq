#!/bin/bash
#SBATCH --mail-user=simon.jakob.marwitz@uni-weimar.de
#SBATCH --mail-type=ALL
#SBATCH --partition=short                              	# partitions: vegas (default), small, short, medium, long
#SBATCH --cpus-per-task=1 				# ansys transient is single threaded anyway
#SBATCH --threads-per-core=1	                        # number of threads per core
#SBACTH --ntasks=2
#SBATCH --mem=7000 					# 3875 (16 jobs per node = 62 GB)
#SBATCH --array=1-40					# with task limit %N to avoid overusing licenses
#SBATCH --output=/vegas/scratch/womo1998/modal_uq/log/gen_sig_%A_%a.out # %A jobid, %a taskid
#SBATCH --job-name=gen_sig

hostname

# define the handler function
# note that this is not executed here, but rather
# when the associated signal is sent
term_handler()
{
        echo "function term_handler called.  Exiting"
	rm -rf /dev/shm/womo1998/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
        exit 0
}
# associate the function "term_handler" with the EXIT signal
trap 'term_handler' EXIT

mkdir -p /dev/shm/womo1998/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
cd /dev/shm/womo1998/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

module load python/intelpython3.7 > /dev/zero  2>&1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK # limit MKL to the number of cpus allocated to avoid hyperthreading
PYTHONPATH=/vegas/users/staff/womo1998/Projects/2019_OMA_UQ/code/
export PYTHONPATH
export PYTHONUNBUFFERED=1

python /vegas/users/staff/womo1998/Projects/2019_OMA_UQ/code/uncertainty/data_manager.py