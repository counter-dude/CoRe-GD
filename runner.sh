#!/bin/bash
#SBATCH --mail-type=NONE # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/itet-stor/jangus/net_scratch/Thesis_J/CoRe-GD/slurm_log/NeuralDrawer/runs%j.out # where to store the output (%j is the JOBID), subdirectory "jobs" must exist
#SBATCH --error=/itet-stor/jangus/net_scratch/Thesis_J/CoRe-GD/slurm_log/NeuralDrawer/%j.err # where to store error messages
#SBATCH --mem=50G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#CommentSBATCH --exclude=tikgpu10,tikgpu[06-07]
#CommentSBATCH --nodelist=tikgpu08 # Specify that it should run on this particular node
#CommentSBATCH --account=tik-internal
#CommentSBATCH --constraint='geforce_rtx_2080_ti|titan_xp|titan_rtx'

set -x  # Enable debugging output
echo "Current Python Path: $(which python)"


ETH_USERNAME=jangus
PROJECT_NAME=CoRe-GD
DIRECTORY=/itet-stor/jangus/net_scratch/Thesis_J/CoRe-GD
CONDA_ENVIRONMENT=CoRe-GD

# debugging...
if [[ ! -d "${DIRECTORY}" ]]; then
    echo "Error: Directory ${DIRECTORY} does not exist."
    exit 1
fi


# Exit on errors
set -o errexit

# Send some noteworthy information to the output log

echo "Running on node: $(hostname)"
echo "In directory: $(pwd)"
echo "Starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"


[[ -f /itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda ]] && eval "$(/itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda shell.bash hook)"
conda activate ${CONDA_ENVIRONMENT}
echo "Conda activated"
cd ${DIRECTORY}


# Execute your code
/itet-stor/jangus/net_scratch/conda_envs/CoRe-GD/bin/python run_experiment.py --config configs/config_rome.json

# Send more noteworthy information to the output log
echo "Finished at: $(date)"

# End the script with exit code 0
exit 0