# Probing Language Models With Topic Models
**Master's thesis of:** Franz Knobel

**Supervised by:** Clara Meister and Prof. Dr. Ryan Cotterell

## Introduction
In this thesis, we try to gain a better understanding of the inductive bias encoded in the language modeling algorithm.  
We then train topic models to understand the distribution over topics that the language model captures 
and compare those distributions across different corpora to better comprehend this inductive.
For more details please have a look at the thesis itself.

## General Procedure
Therefore, we train two language models (GPT-2  Transformer-XL) on the Wikitext103 corpus. 
We do some preprocessing on the corpus in before training, namely we remove all headers.
We then gather different corpora to perform our analysis on:
1. We generate two corpora of size 100k (with seed 42 and 1337) from *gpt2_nt* with
   1. Ancestral sampling
   2. Top-P sampling
   3. Typical sampling
2. We create a subset of 10k of all corpora generated in **1.** by taking the first 10k documents
3. We generate two corpora of size 100k (with seed 42 and 1337) from the original language model *gpt2* with ancestral sampling
4. We create a subset of 10k of all corpora generated in **3.** by taking the first 10k documents
5. We generate two corpora of size 10k (with seed 42 and 1337) from *trafo_xl_nt* with
   1. Ancestral sampling
   2. Top-P sampling
   3. Typical sampling
6. We generate two corpora of size 10k (with seed 42 and 1337) from the original language model *trafo_xl* with ancestral sampling
7. We choose four subsets of existing corpora *wikitext* and *arxiv* (from *arxiv* we only take the abstracts) by sampling two times 100k and 10k documents with different seeds (42 and 1337) with replacement

In the thesis the marker *_ours* refers to the marker *_nt*

All in all we get the following corpora (dataset1 and dataset2 are different by their seed (42 and 1337)):
    
    dataset1-Wikitext103-10000
    dataset1-Wikitext103-100000
    dataset1-ArXiv-10000
    dataset1-ArXiv-100000
    dataset2-Wikitext103-10000
    dataset2-Wikitext103-100000
    dataset2-ArXiv-10000
    dataset2-ArXiv-100000

    dataset1-gpt2-10000
    dataset1-gpt2-100000
    dataset1-trafo_xl-10000
    dataset2-gpt2-10000
    dataset2-gpt2-100000
    dataset2-trafo_xl-10000

    dataset1-gpt2_nt-10000
    dataset1-gpt2_nt-100000
    dataset1-gpt2_nt-top_p-10000
    dataset1-gpt2_nt-top_p-100000
    dataset1-gpt2_nt-typ_p-10000
    dataset1-gpt2_nt-typ_p-100000
    dataset1-trafo_xl_nt-10000
    dataset1-trafo_xl_nt-top_p-10000
    dataset1-trafo_xl_nt-typ_p-10000

    dataset2-gpt2_nt-10000
    dataset2-gpt2_nt-100000
    dataset2-gpt2_nt-top_p-10000
    dataset2-gpt2_nt-top_p-100000
    dataset2-gpt2_nt-typ_p-10000
    dataset2-gpt2_nt-typ_p-100000
    dataset2-trafo_xl_nt-10000
    dataset2-trafo_xl_nt-top_p-10000
    dataset2-trafo_xl_nt-typ_p-10000

We compare each *dataset1-X* with its counterpart *dataset2-X* and all *dataset1* models with each other.

We evaluate the quality of our topic models with the C_v score and measure the comparison between two topic models with
our own designed metric. In the code we refer to our own metric as the top topic score (tt-score).

Our computations were executed on the Euler supercomputer with the help of gpus. 

## Replication
To replicate our results you need access to the euler super computer, or you have to make some changes in the code.
### Folder Structure
The folder structure is important as our code is somewhat dependent on it. You should make a "data" folder 
at the same level as your execution files.
### How to Bjobs
To run jobs on the euler, you simply can use my bash script examples in the "data/job_examples" folder. 
Have first a look at "run.sh" to get aquainted with the structure.
### Python Packages
You need to install the HuggingFace library from source. The rest can be installed with the "requirements.txt".
```
pip install git+https://github.com/huggingface/transformers
```
### Train Language Models
```
python /cluster/work/cotterell/knobelf/run_clm.py --model_type gpt2 --tokenizer_name gpt2 --output_dir /cluster/work/cotterell/knobelf/model-gpt2-wiki_nt --do_train --do_eval --block_size 512 --overwrite_output_dir --train_file /cluster/work/cotterell/knobelf/data/data_wikitext-103-raw/wiki.train.raw_no_titles.txt --validation_file /cluster/work/cotterell/knobelf/data/data_wikitext-103-raw/wiki.valid.raw_no_titles.txt --seed 42
python /cluster/work/cotterell/knobelf/run_clm.py --model_type transfo-xl --tokenizer_name transfo-xl-wt103 --config_name transfo-xl-wt103 --output_dir /cluster/work/cotterell/knobelf/model-trafo_xl-wiki_nt --do_train --do_eval --block_size 256 --overwrite_output_dir --train_file /cluster/work/cotterell/knobelf/data/data_wikitext-103-raw/wiki.train.raw_no_titles.txt --validation_file /cluster/work/cotterell/knobelf/data/data_wikitext-103-raw/wiki.valid.raw_no_titles.txt --seed 42
```
### Sample Text From Language Models
```
python /cluster/work/cotterell/knobelf/generate_corpora.py /cluster/work/cotterell/knobelf/data/ gpt2 multinomial 0 100000
python /cluster/work/cotterell/knobelf/generate_corpora.py /cluster/work/cotterell/knobelf/data/ gpt2 multinomial 1 100000
python /cluster/work/cotterell/knobelf/generate_corpora.py /cluster/work/cotterell/knobelf/data/ gpt2-wiki_nt multinomial 0 100000
python /cluster/work/cotterell/knobelf/generate_corpora.py /cluster/work/cotterell/knobelf/data/ gpt2-wiki_nt multinomial 1 100000
python /cluster/work/cotterell/knobelf/generate_corpora.py /cluster/work/cotterell/knobelf/data/ gpt2-wiki_nt top_p 0 100000
python /cluster/work/cotterell/knobelf/generate_corpora.py /cluster/work/cotterell/knobelf/data/ gpt2-wiki_nt top_p 1 100000
python /cluster/work/cotterell/knobelf/generate_corpora.py /cluster/work/cotterell/knobelf/data/ gpt2-wiki_nt typ_p 0 100000
python /cluster/work/cotterell/knobelf/generate_corpora.py /cluster/work/cotterell/knobelf/data/ gpt2-wiki_nt typ_p 1 100000

python /cluster/work/cotterell/knobelf/generate_corpora.py /cluster/work/cotterell/knobelf/data/ trafo_xl multinomial 0 10000
python /cluster/work/cotterell/knobelf/generate_corpora.py /cluster/work/cotterell/knobelf/data/ trafo_xl multinomial 1 10000
python /cluster/work/cotterell/knobelf/generate_corpora.py /cluster/work/cotterell/knobelf/data/ trafo_xl-wiki_nt multinomial 0 10000
python /cluster/work/cotterell/knobelf/generate_corpora.py /cluster/work/cotterell/knobelf/data/ trafo_xl-wiki_nt multinomial 1 10000
python /cluster/work/cotterell/knobelf/generate_corpora.py /cluster/work/cotterell/knobelf/data/ trafo_xl-wiki_nt top_p 0 10000
python /cluster/work/cotterell/knobelf/generate_corpora.py /cluster/work/cotterell/knobelf/data/ trafo_xl-wiki_nt top_p 1 10000
python /cluster/work/cotterell/knobelf/generate_corpora.py /cluster/work/cotterell/knobelf/data/ trafo_xl-wiki_nt typ_p 0 10000
python /cluster/work/cotterell/knobelf/generate_corpora.py /cluster/work/cotterell/knobelf/data/ trafo_xl-wiki_nt typ_p 1 10000
``` 
### Generate Topic Models for Pairs
The easiest way to do that is to open the "bjobs_gen.py" file, adjust the path parameter 
and run each function line in main function seperately. 
That way you generate stepwise the bjobs you then can submit to euler.

1. generate_tokenize_bjobs(path, 100000)
2. generate_tokenize_bjobs(path, 10000)
3. generate_classic_lda_bjobs(path, 100000)
4. generate_classic_lda_bjobs(path, 10000)
5. generate_classic_lda_variation_bjobs(path, 100000)
6. generate_classic_lda_variation_bjobs(path, 10000)
7. generate_neural_lda_bjobs(path, 10000)
8. generate_score_bjobs(path, 10000)
9. generate_score_bjobs(path, 100000)

After each step add the generated jobs manually to the bash script 
(like in the different job_examples) 
and wait until every job has finished completely without errors.

How the individual functions for topic model creation and scoring work can be found well 
documented in the main functions of the respective python files. 

### Plotting
After all those jobs run successfully, you can plot everything:
```
python /cluster/work/cotterell/knobelf/plot_scores.py /cluster/work/cotterell/knobelf/data/
```

## Questions
If you are interested in my research or have questions, please feel free to contact me :)