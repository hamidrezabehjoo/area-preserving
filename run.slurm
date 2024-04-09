#!/bin/bash
#SBATCH --output=results.out
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --mem-per-cpu=5gb
#SBATCH --time=99:00:00   
#SBATCH --gres=gpu:1
#SBATCH --account=chertkov
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_chertkov



module load python/3.11/3.11.4
source  /home/u14/hbehjoo/env11/bin/activate

### change to your script’s directory
cd ~/diffusion-bridge
### Run your work
python3 train.py

