#!/bin/bash
#SBATCH --job-name=lab2_tests              # name shown in the queue
#SBATCH --output=lab2_tests_%j.out         # stdout + stderr; %j becomes the job id
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rajas@ufl.edu
#SBATCH --account=eel4930                  # whose allocation pays
#SBATCH --qos=eel4930                      # which pool of it
#SBATCH --gpus=1                           # how many GPUs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4                  # >= 1 core per GPU
#SBATCH --mem=32gb                         # host memory, not GPU memory
#SBATCH --time=00:10:00                    # hh:mm:ss, then it is killed

# Run from the folder where the job was submitted
cd "$SLURM_SUBMIT_DIR"

module load cuda 

echo "---- Environment ----"
date
hostname
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
nvcc --version | tail -n 2

echo
echo "---- Build ----"
echo "nvcc -O2 -lineinfo -std=c++17 lab2.cu -o lab2"
nvcc -O2 -lineinfo -std=c++17 lab2.cu -o lab2 || { echo "BUILD FAILED"; exit 1; }

echo
echo "---- Part A Public Tests ----"
for dims in "1 1 1" "3 5 2" "16 16 16" "31 17 50" "65 33 129"; do
    read -r M K N <<< "$dims"
    echo "--- M=$M K=$K N=$N ---"
    ./lab2 --part mm --M "$M" --K "$K" --N "$N"
done

echo 
echo "---- Compute Sanitizer ----"
compute-sanitizer --tool memcheck ./lab2 --part mm --M 31 --K 17 --N 50
