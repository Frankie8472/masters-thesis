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
NUM_GPUS=2  # GPUs per node
GPU_MODEL=TeslaV100_SXM2_32GB  # Choices: GeForceGTX1080,GeForceGTX1080Ti,GeForceRTX2080Ti,TeslaV100_SXM2_32GB
NUM_CPUS=8  # Number of cores (default: 1)
CPU_RAM=8000  # RAM for each core (default: 1024)
SCRIPT1="python /cluster/work/cotterell/knobelf/run_clm.py --model_type gpt2 --tokenizer_name gpt2 --dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 --output_dir /cluster/work/cotterell/knobelf/model-gpt2-wiki-integrated --do_train --do_eval --block_size 512 --overwrite_output_dir --seed 42"
SCRIPT2="python /cluster/work/cotterell/knobelf/run_clm.py --model_type gpt2 --tokenizer_name gpt2 --output_dir /cluster/work/cotterell/knobelf/model-gpt2-wiki_nt --do_train --do_eval --block_size 512 --overwrite_output_dir --train_file /cluster/work/cotterell/knobelf/data/data_wikitext-103-raw/wiki.train.raw_no_titles.txt --validation_file /cluster/work/cotterell/knobelf/data/data_wikitext-103-raw/wiki.valid.raw_no_titles.txt --seed 42"
SCRIPT3="python /cluster/work/cotterell/knobelf/run_clm.py --model_type gpt2 --tokenizer_name gpt2 --output_dir /cluster/work/cotterell/knobelf/model-gpt2-wiki --do_train --do_eval --block_size 512 --overwrite_output_dir --train_file /cluster/work/cotterell/knobelf/data/data_wikitext-103-raw/wiki.train.raw.txt --validation_file /cluster/work/cotterell/knobelf/data/data_wikitext-103-raw/wiki.valid.raw.txt --seed 42"

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!! Switch to new infrastructure with command "env2lmod"!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Load python module (from wiki)
module load gcc/8.2.0 eth_proxy python_gpu/3.9.9

# Run python batch jobs
bsub -N -W $TIME -n $NUM_CPUS -R "rusage[mem=${CPU_RAM},ngpus_excl_p=${NUM_GPUS}]" -R "select[gpu_model0==${GPU_MODEL}]" $SCRIPT1
bsub -N -W $TIME -n $NUM_CPUS -R "rusage[mem=${CPU_RAM},ngpus_excl_p=${NUM_GPUS}]" -R "select[gpu_model0==${GPU_MODEL}]" $SCRIPT2
bsub -N -W $TIME -n $NUM_CPUS -R "rusage[mem=${CPU_RAM},ngpus_excl_p=${NUM_GPUS}]" -R "select[gpu_model0==${GPU_MODEL}]" $SCRIPT3

