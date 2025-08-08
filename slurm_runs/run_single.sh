#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1         # 1 tasks per node
#SBATCH --account=FUPA1_UKAEA_ML      # account name
#SBATCH --partition=gpu # partition name
#SBATCH --gres=gpu:1
#SBATCH --mail-user=alicja.polanska.22@ucl.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --time=1-00:00:00  # 1 day and 12 hours

echo "Activating environment"
source ~/venvs/al4pde/bin/activate
echo "Python is "
which python

cd /pitagora_work/FUPA1_UKAEA_ML/apolansk/al4pde

echo "Running acquisition=$ACQ seed=$SEED"

python -m scripts.run_active_learning task=burgers acquisition=$ACQ seed=$SEED prob_model=PREModel prob_model.model_wrapper.training_type=autoregressive

