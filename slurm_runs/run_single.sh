#!/bin/bash
#SBATCH -p GPU #requesting one node
#SBATCH -N1 #requesting 12 cpus
#SBATCH -n12 #requesting 1 V100 GPU
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-user=alicja.polanska.22@ucl.ac.uk
#SBATCH --mail-type=ALL

source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate al4pde

echo "Python is "
which python
echo "Visible devices"
echo $CUDA_VISIBLE_DEVICES

cd /home/alicjaap/al4pde

echo "Running acquisition=$ACQ seed=$SEED"

python -m scripts.run_active_learning task=burgers acquisition=$ACQ seed=$SEED prob_model=PREModel prob_model.model_wrapper.training_type=autoregressive

