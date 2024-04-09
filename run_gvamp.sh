# Set data directory
DATA_DIR=/home/mnt
OUT_DIR=/home/mnt/output

# Create output folder in data directory.
mkdir ${OUT_DIR}

# Set parallelization according to available CPU resources
NUM_MPI_WORKERS=...
NUM_OMP_THREADS=...

export OMP_NUM_THREADS=${NUM_OMP_THREADS}
mpirun -np ${NUM_MPI_WORKERS} --allow-run-as-root main_real.exe \
	--model linear \
	--run-mode infere \
	--bim-file ${DATA_DIR}/... \
	--bed-file ${DATA_DIR}/... \
	--phen-file ${DATA_DIR}/... \
	--N ... \
	--Mt ... \
	--out-dir ${OUT_DIR}/ \
	--out-name ... \
	--iterations ... \
	--num-mix-comp ... \
	--probs ... \
	--vars ... \
	--rho ... \
> ${OUT_DIR}/gvamp.log
