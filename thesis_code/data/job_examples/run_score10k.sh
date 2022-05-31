#!/bin/bash
# Author:       Franz Knobel
# Email:        knobelf@student.ethz.ch
# Date:         2022-02-01
# Usage:        ./run.sh
# Description:
# Run batch jobs over here to keep commandline clean!
# https://scicomp.ethz.ch/wiki/Python
# https://tstesco.github.io/euler-hpc-cluster/
# Max time: 120h
# Max cores: 48
# Max mem: 131072 

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!! Switch to new infrastructure with command "env2lmod"!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Load python module (from wiki)
module load gcc/8.2.0 eth_proxy python_gpu/3.9.9

# Run python batch jobs
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-gpt2_nt-gpt2_nt.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 gpt2_nt-gpt2_nt"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-gpt2_nt-gpt2.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 gpt2_nt-gpt2"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-gpt2_nt-trafo_xl_nt.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 gpt2_nt-trafo_xl_nt"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-gpt2_nt-trafo_xl.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 gpt2_nt-trafo_xl"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-gpt2_nt-wiki_nt.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 gpt2_nt-wiki_nt"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-gpt2_nt-arxiv.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 gpt2_nt-arxiv"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-gpt2-gpt2.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 gpt2-gpt2"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-gpt2-trafo_xl_nt.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 gpt2-trafo_xl_nt"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-gpt2-trafo_xl.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 gpt2-trafo_xl"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-gpt2-wiki_nt.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 gpt2-wiki_nt"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-gpt2-arxiv.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 gpt2-arxiv"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-trafo_xl_nt-trafo_xl_nt.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 trafo_xl_nt-trafo_xl_nt"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-trafo_xl_nt-trafo_xl.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 trafo_xl_nt-trafo_xl"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-trafo_xl_nt-wiki_nt.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 trafo_xl_nt-wiki_nt"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-trafo_xl_nt-arxiv.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 trafo_xl_nt-arxiv"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-trafo_xl-trafo_xl.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 trafo_xl-trafo_xl"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-trafo_xl-wiki_nt.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 trafo_xl-wiki_nt"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-trafo_xl-arxiv.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 trafo_xl-arxiv"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-wiki_nt-wiki_nt.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 wiki_nt-wiki_nt"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-wiki_nt-arxiv.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 wiki_nt-arxiv"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-arxiv-arxiv.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 arxiv-arxiv"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-gpt2_nt-gpt2_nt-top_p.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 gpt2_nt-gpt2_nt-top_p"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-gpt2_nt-gpt2-top_p.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 gpt2_nt-gpt2-top_p"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-gpt2_nt-trafo_xl_nt-top_p.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 gpt2_nt-trafo_xl_nt-top_p"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-gpt2_nt-trafo_xl-top_p.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 gpt2_nt-trafo_xl-top_p"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-gpt2_nt-wiki_nt-top_p.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 gpt2_nt-wiki_nt-top_p"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-gpt2_nt-arxiv-top_p.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 gpt2_nt-arxiv-top_p"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-gpt2-trafo_xl_nt-top_p.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 gpt2-trafo_xl_nt-top_p"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-trafo_xl_nt-trafo_xl_nt-top_p.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 trafo_xl_nt-trafo_xl_nt-top_p"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-trafo_xl_nt-trafo_xl-top_p.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 trafo_xl_nt-trafo_xl-top_p"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-trafo_xl_nt-wiki_nt-top_p.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 trafo_xl_nt-wiki_nt-top_p"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-trafo_xl_nt-arxiv-top_p.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 trafo_xl_nt-arxiv-top_p"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-gpt2_nt-gpt2_nt-typ_p.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 gpt2_nt-gpt2_nt-typ_p"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-gpt2_nt-gpt2-typ_p.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 gpt2_nt-gpt2-typ_p"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-gpt2_nt-trafo_xl_nt-typ_p.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 gpt2_nt-trafo_xl_nt-typ_p"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-gpt2_nt-trafo_xl-typ_p.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 gpt2_nt-trafo_xl-typ_p"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-gpt2_nt-wiki_nt-typ_p.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 gpt2_nt-wiki_nt-typ_p"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-gpt2_nt-arxiv-typ_p.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 gpt2_nt-arxiv-typ_p"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-gpt2-trafo_xl_nt-typ_p.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 gpt2-trafo_xl_nt-typ_p"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-trafo_xl_nt-trafo_xl_nt-typ_p.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 trafo_xl_nt-trafo_xl_nt-typ_p"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-trafo_xl_nt-trafo_xl-typ_p.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 trafo_xl_nt-trafo_xl-typ_p"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-trafo_xl_nt-wiki_nt-typ_p.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 trafo_xl_nt-wiki_nt-typ_p"
bsub -W 24:00 -n 48 -R "rusage[mem=2666,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10000]" -oo logs/10000/log-tt-trafo_xl_nt-arxiv-typ_p.txt "python /cluster/work/cotterell/knobelf/score_lda.py /cluster/work/cotterell/knobelf/data/ tt 10000 trafo_xl_nt-arxiv-typ_p"