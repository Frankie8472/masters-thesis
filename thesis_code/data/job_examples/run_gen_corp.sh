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
TIME=4:00  # HH:MM (default: 04:00, max: 240:00)
NUM_GPUS=1  # GPUs per node
GPU_MODEL=NVIDIAGeForceRTX3090   
GPU_MEM=20000 
# Choices: NVIDIAGeForceGTX1080, NVIDIAGeForceGTX1080Ti, NVIDIAGeForceRTX2080Ti, NVIDIATITANRTX, QuadroRTX6000, TeslaV100_SXM2_32GB, NVIDIAA100_PCIE_40GB
NUM_CPUS=8  # Number of cores (default: 1)
CPU_RAM=8192  # RAM for each core (default: 1024)
SCRIPT1="python /cluster/work/cotterell/knobelf/generate_corpora.py /cluster/work/cotterell/knobelf/data/ gpt2 multinomial 0 100000"
SCRIPT2="python /cluster/work/cotterell/knobelf/generate_corpora.py /cluster/work/cotterell/knobelf/data/ gpt2 multinomial 1 100000"
SCRIPT3="python /cluster/work/cotterell/knobelf/generate_corpora.py /cluster/work/cotterell/knobelf/data/ gpt2-wiki multinomial 0 100000"
SCRIPT4="python /cluster/work/cotterell/knobelf/generate_corpora.py /cluster/work/cotterell/knobelf/data/ gpt2-wiki-integrated multinomial 0 100000"
SCRIPT5="python /cluster/work/cotterell/knobelf/generate_corpora.py /cluster/work/cotterell/knobelf/data/ gpt2-wiki_nt multinomial 0 100000"
SCRIPT6="python /cluster/work/cotterell/knobelf/generate_corpora.py /cluster/work/cotterell/knobelf/data/ gpt2-wiki_nt multinomial 1 100000"
SCRIPT7="python /cluster/work/cotterell/knobelf/generate_corpora.py /cluster/work/cotterell/knobelf/data/ gpt2-wiki_nt top_p 0 100000"
SCRIPT8="python /cluster/work/cotterell/knobelf/generate_corpora.py /cluster/work/cotterell/knobelf/data/ gpt2-wiki_nt top_p 1 100000"
SCRIPT9="python /cluster/work/cotterell/knobelf/generate_corpora.py /cluster/work/cotterell/knobelf/data/ gpt2-wiki_nt typ_p 0 100000"
SCRIPT10="python /cluster/work/cotterell/knobelf/generate_corpora.py /cluster/work/cotterell/knobelf/data/ gpt2-wiki_nt typ_p 1 100000"

SCRIPT11="python /cluster/work/cotterell/knobelf/generate_corpora.py /cluster/work/cotterell/knobelf/data/ trafo_xl multinomial 0 10000"
SCRIPT12="python /cluster/work/cotterell/knobelf/generate_corpora.py /cluster/work/cotterell/knobelf/data/ trafo_xl multinomial 1 10000"
SCRIPT13="python /cluster/work/cotterell/knobelf/generate_corpora.py /cluster/work/cotterell/knobelf/data/ trafo_xl-wiki multinomial 0 10000"
SCRIPT14="python /cluster/work/cotterell/knobelf/generate_corpora.py /cluster/work/cotterell/knobelf/data/ trafo_xl-wiki-integrated multinomial 0 10000"
SCRIPT15="python /cluster/work/cotterell/knobelf/generate_corpora.py /cluster/work/cotterell/knobelf/data/ trafo_xl-wiki_nt multinomial 0 10000"
SCRIPT16="python /cluster/work/cotterell/knobelf/generate_corpora.py /cluster/work/cotterell/knobelf/data/ trafo_xl-wiki_nt multinomial 1 2500"
SCRIPT17="python /cluster/work/cotterell/knobelf/generate_corpora.py /cluster/work/cotterell/knobelf/data/ trafo_xl-wiki_nt top_p 0 10000"
SCRIPT18="python /cluster/work/cotterell/knobelf/generate_corpora.py /cluster/work/cotterell/knobelf/data/ trafo_xl-wiki_nt top_p 1 10000"
SCRIPT19="python /cluster/work/cotterell/knobelf/generate_corpora.py /cluster/work/cotterell/knobelf/data/ trafo_xl-wiki_nt typ_p 0 10000"
SCRIPT20="python /cluster/work/cotterell/knobelf/generate_corpora.py /cluster/work/cotterell/knobelf/data/ trafo_xl-wiki_nt typ_p 1 10000"

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!! Switch to new infrastructure with command "env2lmod"!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Load python module (from wiki)
module load gcc/8.2.0 eth_proxy python_gpu/3.9.9

# Run python batch jobs
#bsub -N -W $TIME -n $NUM_CPUS -R "rusage[mem=${CPU_RAM},ngpus_excl_p=${NUM_GPUS}]" -R "select[gpu_model0==${GPU_MODEL}]" $SCRIPT1
#bsub -N -W $TIME -n $NUM_CPUS -R "rusage[mem=${CPU_RAM},ngpus_excl_p=${NUM_GPUS}]" -R "select[gpu_model0==${GPU_MODEL}]" $SCRIPT2
#bsub -N -W $TIME -n $NUM_CPUS -R "rusage[mem=${CPU_RAM},ngpus_excl_p=${NUM_GPUS}]" -R "select[gpu_model0==${GPU_MODEL}]" $SCRIPT3
#bsub -N -W $TIME -n $NUM_CPUS -R "rusage[mem=${CPU_RAM},ngpus_excl_p=${NUM_GPUS}]" -R "select[gpu_model0==${GPU_MODEL}]" $SCRIPT4
#bsub -N -W $TIME -n $NUM_CPUS -R "rusage[mem=${CPU_RAM},ngpus_excl_p=${NUM_GPUS}]" -R "select[gpu_model0==${GPU_MODEL}]" $SCRIPT5
#bsub -N -W $TIME -n $NUM_CPUS -R "rusage[mem=${CPU_RAM},ngpus_excl_p=${NUM_GPUS}]" -R "select[gpu_model0==${GPU_MODEL}]" $SCRIPT6
#bsub -N -W $TIME -n $NUM_CPUS -R "rusage[mem=${CPU_RAM},ngpus_excl_p=${NUM_GPUS}]" -R "select[gpu_model0==${GPU_MODEL}]" $SCRIPT7
#bsub -N -W $TIME -n $NUM_CPUS -R "rusage[mem=${CPU_RAM},ngpus_excl_p=${NUM_GPUS}]" -R "select[gpu_model0==${GPU_MODEL}]" $SCRIPT8
#bsub -N -W $TIME -n $NUM_CPUS -R "rusage[mem=${CPU_RAM},ngpus_excl_p=${NUM_GPUS}]" -R "select[gpu_model0==${GPU_MODEL}]" $SCRIPT9
#bsub -N -W $TIME -n $NUM_CPUS -R "rusage[mem=${CPU_RAM},ngpus_excl_p=${NUM_GPUS}]" -R "select[gpu_model0==${GPU_MODEL}]" $SCRIPT10

#bsub -N -W $TIME -n $NUM_CPUS -R "rusage[mem=${CPU_RAM},ngpus_excl_p=${NUM_GPUS}]" -R "select[gpu_mtotal0>=${GPU_MEM}]" $SCRIPT11
#bsub -N -W $TIME -n $NUM_CPUS -R "rusage[mem=${CPU_RAM},ngpus_excl_p=${NUM_GPUS}]" -R "select[gpu_mtotal0>=${GPU_MEM}]" $SCRIPT12
#bsub -N -W $TIME -n $NUM_CPUS -R "rusage[mem=${CPU_RAM},ngpus_excl_p=${NUM_GPUS}]" -R "select[gpu_mtotal0>=${GPU_MEM}]" $SCRIPT13
#bsub -N -W $TIME -n $NUM_CPUS -R "rusage[mem=${CPU_RAM},ngpus_excl_p=${NUM_GPUS}]" -R "select[gpu_mtotal0>=${GPU_MEM}]" $SCRIPT14
#bsub -N -W $TIME -n $NUM_CPUS -R "rusage[mem=${CPU_RAM},ngpus_excl_p=${NUM_GPUS}]" -R "select[gpu_mtotal0>=${GPU_MEM}]" $SCRIPT15
bsub -N -W $TIME -n $NUM_CPUS -R "rusage[mem=${CPU_RAM},ngpus_excl_p=${NUM_GPUS}]" -R "select[gpu_model0==${GPU_MODEL}]" $SCRIPT16
#bsub -N -W $TIME -n $NUM_CPUS -R "rusage[mem=${CPU_RAM},ngpus_excl_p=${NUM_GPUS}]" -R "select[gpu_mtotal0>=${GPU_MEM}]" $SCRIPT17
#bsub -N -W $TIME -n $NUM_CPUS -R "rusage[mem=${CPU_RAM},ngpus_excl_p=${NUM_GPUS}]" -R "select[gpu_mtotal0>=${GPU_MEM}]" $SCRIPT18
#bsub -N -W $TIME -n $NUM_CPUS -R "rusage[mem=${CPU_RAM},ngpus_excl_p=${NUM_GPUS}]" -R "select[gpu_mtotal0>=${GPU_MEM}]" $SCRIPT19
#bsub -N -W $TIME -n $NUM_CPUS -R "rusage[mem=${CPU_RAM},ngpus_excl_p=${NUM_GPUS}]" -R "select[gpu_mtotal0>=${GPU_MEM}]" $SCRIPT20

#bsub -N -W 48:00 -n 8 -R "rusage[mem=8192,ngpus_excl_p=1]" -R "select[gpu_model0==A100_PCIE_40GB]" "python /cluster/work/cotterell/knobelf/generate_corpora.py /cluster/work/cotterell/knobelf/data/ trafo_xl-wiki_nt multinomial 0 10000"