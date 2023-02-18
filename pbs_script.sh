#!/bin/bash
#PBS -l select=1:ncpus=1:mem=7G
#PBS -N flexibility_simulation
#PBS -j oe

echo "working directory: " $PBS_O_WORKDIR
echo "omp thread num: " $OMP_NUM_THREADS
echo "ncpus: " $NCPUS
echo "cuda visible devices: " $CUDA_VISIBLE_DEVICES

date +"%Y/%m/%d %p %I:%M:%S"
start_time=`date +%s`
cd $PBS_O_WORKDIR

/home/taka/.pyenv/shims/python main.py

date +"%Y/%m/%d %p %I:%M:%S"
end_time=`date +%s`
run_time=$((end_time - start_time))

echo "start time: " $start_time
echo "end time: " $end_time
echo "run time: " $run_time
