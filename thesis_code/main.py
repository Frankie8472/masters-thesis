import gc
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator
from gensim.models.ldamulticore import LdaMulticore
from gensim.utils import SaveLoad
import torch
from tqdm.auto import tqdm
import itertools
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel, GPTNeoForCausalLM, \
    set_seed


def generate_bjobs():
    topics = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]
    models = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for model in models:
        for topic in topics:
            for union in [0, 1]:
                combi = "un" if union else "is"
                if model == 1:
                    name = f"wiki_nt-gpt2_nt-{combi}-{topic}.txt"
                elif model == 2:
                    name = f"gpt2_nt-wiki_nt-{combi}-{topic}.txt"
                elif model == 3:
                    name = f"gpt2_nt_1-gpt2_nt_2-{combi}-{topic}.txt"
                elif model == 4:
                    name = f"gpt2_nt_2-gpt2_nt_1-{combi}-{topic}.txt"
                elif model == 5:
                    name = f"gpt2_1-gpt2_2-{combi}-{topic}.txt"
                elif model == 6:
                    name = f"gpt2_2-gpt2_1-{combi}-{topic}.txt"
                elif model == 7:
                    name = f"wiki_nt-arxiv-{combi}-{topic}.txt"
                elif model == 8:
                    name = f"arxiv-wiki_nt-{combi}-{topic}.txt"
                elif model == 9:
                    name = f"gpt2_nt-arxiv-{combi}-{topic}.txt"
                else:
                    name = f"arxiv-gpt2_nt-{combi}-{topic}.txt"
                print(
                    f"bsub -N -W 24:00 -n 48 -R \"rusage[mem=2666]\" -o ma/log-{name} \"python ma/train_lda.py {union} {model} {topic}\"")


def score_by_topic_probability(ldamodel_1, ldamodel_2, corpus_1, corpus_2, distance='jensen_shannon'):
    mdiff1, annotation1 = ldamodel_1.diff(ldamodel_2, distance=distance, num_words=1000000000)
    mdiff2, annotation2 = ldamodel_2.diff(ldamodel_1, distance=distance, num_words=1000000000)
    min1 = np.amin(mdiff1, axis=1)
    min2 = np.amin(mdiff2, axis=1)
    topic_corpus_prob_1 = np.zeros(ldamodel_1.num_topics)
    topic_corpus_prob_2 = np.zeros(ldamodel_2.num_topics)
    probas_1 = ldamodel_1.get_document_topics(list(itertools.chain.from_iterable(corpus_1)), minimum_probability=0.0)
    probas_2 = ldamodel_2.get_document_topics(list(itertools.chain.from_iterable(corpus_2)), minimum_probability=0.0)
    for key, val in probas_1:
        topic_corpus_prob_1[key] = val
    for key, val in probas_2:
        topic_corpus_prob_2[key] = val
    return (np.sum(topic_corpus_prob_1 * min1) + np.sum(topic_corpus_prob_2 * min2))/2


def score_by_top_topic(ldamodel_1, ldamodel_2, corpus_1, corpus_2, distance='jensen_shannon'):
    mdiff1, annotation1 = ldamodel_1.diff(ldamodel_2, distance=distance, num_words=1000000000)
    mdiff2, annotation2 = ldamodel_2.diff(ldamodel_1, distance=distance, num_words=1000000000)
    min1 = np.amin(mdiff1, axis=1)
    min2 = np.amin(mdiff2, axis=1)

    cnt1 = np.zeros(ldamodel_1.num_topics)
    for doc in corpus_1:
        prob = ldamodel_1.get_document_topics(doc, minimum_probability=0.0)
        topic = max(prob, prob.get)
        cnt1[topic] += 1
    cnt2 = np.zeros(ldamodel_1.num_topics)
    for doc in corpus_2:
        prob = ldamodel_2.get_document_topics(doc, minimum_probability=0.0)
        topic = max(prob, prob.get)
        cnt2[topic] += 1

    return (np.sum(cnt1 * min1)/np.sum(cnt1) + np.sum(cnt2 * min2)/np.sum(cnt2)) / 2


def calc_score():
    topics = np.asarray([2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100])
    modes = ["intersection", "union"]
    model_pairs = [
        (
            "lda-wiki_nt-gpt2_nt/wiki_nt",
            "lda-wiki_nt-gpt2_nt/gpt2_nt"
        ),
        (
            "lda-gpt2_nt-gpt2_nt/gpt2_1_nt",
            "lda-gpt2_nt-gpt2_nt/gpt2_2_nt"
        ),
        (
            "lda-gpt2_nt-arxiv/gpt2_nt",
            "lda-gpt2_nt-arxiv/arxiv"
        ),
        (
            "lda-wiki_nt-arxiv/wiki_nt",
            "lda-wiki_nt-arxiv/arxiv"
        ),
        (
            "lda-gpt2-gpt2/gpt2_1",
            "lda-gpt2-gpt2/gpt2_2",
        )
    ]
    length = len(topics) * len(modes) * len(model_pairs)
    with tqdm(total=length) as pbar:
        fig, ax = plt.subplots(figsize=(18, 14))
        for model_pair in model_pairs:
            for mode in modes:
                for idx, topic in enumerate(topics):
                    path1 = model_pair[0]
                    path2 = model_pair[1]
                    path_ldamodel_1 = f"data/{path1}/{mode}/{topic}/ldamodel_{topic}"
                    path_ldamodel_2 = f"data/{path2}/{mode}/{topic}/ldamodel_{topic}"
                    path_dictionary_1 = f"data/{path1}/{mode}/{topic}/dictionary_{topic}"
                    path_dictionary_2 = f"data/{path2}/{mode}/{topic}/dictionary_{topic}"
                    path_corpus_1 = f"data/{path1}/{mode}/{topic}/corpus_{topic}"
                    path_corpus_2 = f"data/{path2}/{mode}/{topic}/corpus_{topic}"

                    # Load pretrained models from disk.
                    with open(path_corpus_1, 'r') as file:
                        corpus_1 = json.load(file)
                    with open(path_corpus_2, 'r') as file:
                        corpus_2 = json.load(file)
                    dictionary_1 = SaveLoad.load(path_dictionary_1)
                    dictionary_2 = SaveLoad.load(path_dictionary_2)
                    ldamodel_1 = LdaMulticore.load(path_ldamodel_1)
                    ldamodel_2 = LdaMulticore.load(path_ldamodel_2)

                    distance = 'jensen_shannon'
                    words = 100000000

                    # Compare models with scores_by_topic_probability and save
                    diff_score = score_by_topic_probability(ldamodel_1, ldamodel_2, corpus_1, corpus_2)

                    score_path = "data/score_by_topic_probability_values.json"

                    if os.path.isfile(score_path):
                        with open(score_path, 'r') as file:
                            score_values = json.load(file)
                    else:
                        score_values = dict()

                    short_mode = "is" if mode == "intersection" else "un"
                    key = f"{path1.split('/')[0]}-{short_mode}"
                    if key not in score_values.keys():
                        score_values[key] = np.ones(topics.shape).tolist()

                    score_values[key][idx] = diff_score

                    with open(score_path, 'w') as file:
                        json.dump(score_values, file)

                    # Compare models with score_by_top_topic and save
                    diff_score = score_by_top_topic(ldamodel_1, ldamodel_2, corpus_1, corpus_2)

                    score_path = "data/score_by_top_topic.json"

                    if os.path.isfile(score_path):
                        with open(score_path, 'r') as file:
                            score_values = json.load(file)
                    else:
                        score_values = dict()

                    short_mode = "is" if mode == "intersection" else "un"
                    key = f"{path1.split('/')[0]}-{short_mode}"
                    if key not in score_values.keys():
                        score_values[key] = np.ones(topics.shape).tolist()

                    score_values[key][idx] = diff_score

                    with open(score_path, 'w') as file:
                        json.dump(score_values, file)

                    # Calculate Difference Graph and save it
                    mdiff, annotation = ldamodel_1.diff(ldamodel_2, distance=distance, num_words=words)

                    plt.cla()
                    data = ax.imshow(mdiff, cmap='RdBu_r', vmin=0.0, vmax=1.0, origin='lower')
                    for axis in [ax.xaxis, ax.yaxis]:
                        axis.set_major_locator(MaxNLocator(integer=True))
                    plt.title(
                        f"Topic difference ({path1.split('/')[1]} - {path2.split('/')[1]} - {mode})[{distance} distance] for {topic} topics")
                    plt.colorbar(data)
                    plt.savefig(f"./data/{path1.split('/')[0]}/diff_{short_mode}_{topic}.png", dpi=300)
                    pbar.update(1)


def generate_plot():
    for case in [1,2,3,4]:
        if case == 1:
            score_file_path = "data/score_by_top_topic.json"
            title = "'Score by Top Topic'-Topic Graph for LDA Models (intersected dicts)"
            y_label = "Score by Top Topic (lower is better)"
            mode = 'is'
        elif case == 2:
            score_file_path = "data/score_by_top_topic.json"
            title = "'Score by Top Topic'-Topic Graph for LDA Models (unionized dicts)"
            y_label = "Score by Top Topic (lower is better)"
            mode = 'un'
        elif case == 3:
            score_file_path = "data/score_by_topic_probability_values.json"
            title = "'Score by Topic Probability'-Topic Graph for LDA Models (intersected dicts)"
            y_label = "Score by Topic Probability (lower is better)"
            mode = 'is'
        else:
            score_file_path = "data/score_by_topic_probability_values.json"
            title = "'Score by Topic Probability'-Topic Graph for LDA Models (unionized dicts)"
            y_label = "Score by Topic Probability (lower is better)"
            mode = 'un'
        if os.path.isfile(score_file_path):
            with open(score_file_path, 'r') as file:
                score_values = json.load(file)

        names = list(score_values.keys())
        values = list(score_values.values())
        topics = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]

        plt.clf()
        fig, axes = plt.subplots()
        for idx, name in enumerate(names):
            if mode not in name:
                continue
            axes.plot(topics, values[idx], label=name)
        axes.legend(bbox_to_anchor=(1, 1))
        plt.ylim(0, 1)
        axes.set_title(title)
        axes.set_xscale('log')
        axes.set_xlabel('Number of Topics')
        axes.set_ylabel(y_label)
        axes.set_xticks(topics)
        axes.get_xaxis().set_major_formatter(ScalarFormatter())
        plt.savefig(f"{score_file_path[:-5]}_{mode}.png", dpi=300)
        plt.close('all')


def main():
    generate_plot()


main()
