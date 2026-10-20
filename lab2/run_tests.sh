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

echo 
echo "---- Timing Study (M=2048, K = 1024, N = 1536) ----"
echo "--- Chose M, K, and N values that are multiples of 16 or 32 ---"
for dim2s in "16 16" "32 8"; do
    read -r BX BY <<< "$dim2s"
    echo "--- block ${BX}x${BY} ---"
    ./lab2 --part mm --M 2048 --K 1024 --N 1536 --bx "$BX" --by "$BY"
done

echo
echo "---- Part B Public Tests ----"
./lab2 --part blur --input public_test_images/test_1x1.pgm   --R 1
./lab2 --part blur --input public_test_images/test_3x5.pgm   --R 1
./lab2 --part blur --input public_test_images/test_17x19.pgm --R 2
./lab2 --part blur --input images/irr_grayscale_test_image.pgm --R 2

echo
echo "---- Part B Compute Sanitizer (irregular image, R=2) ----"
compute-sanitizer --tool memcheck ./lab2 --part blur --input images/irr_grayscale_test_image.pgm --R 2

echo
echo "---- Part B Timing Study (irr_grayscale_test_image.pgm, 899x719, R=2) ----"
for shape in "16 16" "32 8"; do
    read -r BX BY <<< "$shape"
    echo "--- block ${BX}x${BY} ---"
    ./lab2 --part blur --input images/irr_grayscale_test_image.pgm --R 2 --bx "$BX" --by "$BY"
done
