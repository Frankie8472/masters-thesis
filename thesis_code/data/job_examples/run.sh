#!/bin/bash
# Author:       Franz Knobel
# Email:        knobelf@student.ethz.ch
# Date:         2018-04-09
# Usage:        bsub -oo /cluster/home/ma/run_script.txt < /cluster/home/run.sh
# Description:
# Run batch jobs over here to keep commandline clean!
# https://scicomp.ethz.ch/wiki/Python
# https://tstesco.github.io/euler-hpc-cluster/
# Max time: 120h
# Max cores: 48
# Max mem: 131072


# Job details
TIME=48:00  # HH:MM (default: 04:00, max: 240:00)
NUM_GPUS=1  # GPUs per node
GPU_MODEL=TeslaV100_SXM2_32GB  # Choices: GeForceGTX1080,GeForceGTX1080Ti,GeForceRTX2080Ti,TeslaV100_SXM2_32GB
NUM_CPUS=8  # Number of cores (default: 1)
CPU_RAM=8000  # RAM for each core (default: 1024)
SCRIPT_="python /cluster/work/cotterell/knobelf/generate_corpora.py 3"
SCRIPT_="python /cluster/work/cotterell/knobelf/train_lda.py 6 100"
SCRIPT_="python /cluster/work/cotterell/knobelf/score_lda.py"
SCRIPT_="python /cluster/work/cotterell/knobelf/test.py"
SCRIPT="python /cluster/work/cotterell/knobelf/generate_corpora.py /cluster/work/cotterell/knobelf/data/ trafo_xl multinomial 1 20000"

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!! Switch to new infrastructure with command "env2lmod"!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Load python module (from wiki)
module load gcc/8.2.0 eth_proxy python_gpu/3.9.9

# Run python batch jobs
#bsub -N -B -W 4:00 -n 48 $SCRIPT
#bsub -N -B -W 24:00 -n 48 $SCRIPT                                   
#bsub -N -B -W 120:00 -n 48 $SCRIPT                                 

#bsub -N -W 4:00 -n 48 -R "rusage[mem=2666]" $SCRIPT
#bsub -N -W 24:00 -n 96 -R "rusage[mem=1024]" $SCRIPT
#bsub -N -W 120:00 -n 48 -R "rusage[mem=2666]" $SCRIPT

#bsub -N -W $TIME -n $NUM_CPUS -R "rusage[mem=${CPU_RAM},ngpus_excl_p=${NUM_GPUS}]" -R "select[gpu_model0==${GPU_MODEL}]" "python ma/generate_corpora.py 3"
#bsub -N -W $TIME -n $NUM_CPUS -R "rusage[mem=${CPU_RAM},ngpus_excl_p=${NUM_GPUS}]" -R "select[gpu_model0==${GPU_MODEL}]" "python ma/generate_corpora.py 4"
#bsub -N -W $TIME -n $NUM_CPUS -R "rusage[mem=${CPU_RAM},ngpus_excl_p=${NUM_GPUS}]" -R "select[gpu_model0==${GPU_MODEL}]" $SCRIPT

bsub -N -W 24:00 -n 4 -R "rusage[mem=5120]" -oo logs/10000/log-classic_lda-gpt2_nt-wiki_nt-is-10-first.txt "python /cluster/work/cotterell/knobelf/train_lda.py /cluster/work/cotterell/knobelf/data/ classic_lda gpt2_nt wiki_nt first multinomial 10 intersection 10000"
bsub -N -W 24:00 -n 4 -R "rusage[mem=5120]" -oo logs/10000/log-classic_lda-gpt2_nt-wiki_nt-un-10-first.txt "python /cluster/work/cotterell/knobelf/train_lda.py /cluster/work/cotterell/knobelf/data/ classic_lda gpt2_nt wiki_nt first multinomial 10 union 10000"
bsub -N -W 24:00 -n 4 -R "rusage[mem=5120]" -oo logs/10000/log-classic_lda-gpt2_nt-wiki_nt-is-10-second.txt "python /cluster/work/cotterell/knobelf/train_lda.py /cluster/work/cotterell/knobelf/data/ classic_lda gpt2_nt wiki_nt second multinomial 10 intersection 10000"
bsub -N -W 24:00 -n 4 -R "rusage[mem=5120]" -oo logs/10000/log-classic_lda-gpt2_nt-wiki_nt-un-10-second.txt "python /cluster/work/cotterell/knobelf/train_lda.py /cluster/work/cotterell/knobelf/data/ classic_lda gpt2_nt wiki_nt second multinomial 10 union 10000"

bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-img-gpt2_nt-wiki_nt.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ img 10000 gpt2_nt-wiki_nt"


