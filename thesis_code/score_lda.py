import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from gensim.models.ldamulticore import LdaMulticore
from gensim.utils import SaveLoad
from tqdm.auto import tqdm
from operator import itemgetter
import itertools


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
    return (np.sum(topic_corpus_prob_1 * min1) + np.sum(topic_corpus_prob_2 * min2)) / 2


def score_by_top_topic(ldamodel_1, ldamodel_2, corpus_1, corpus_2, distance='jensen_shannon'):
    mdiff1, annotation1 = ldamodel_1.diff(ldamodel_2, distance=distance, num_words=1000000000)
    mdiff2, annotation2 = ldamodel_2.diff(ldamodel_1, distance=distance, num_words=1000000000)
    min1 = np.amin(mdiff1, axis=1)
    min2 = np.amin(mdiff2, axis=1)

    cnt1 = np.zeros(ldamodel_1.num_topics)
    for doc in corpus_1:
        topic_prob_list = ldamodel_1.get_document_topics(doc, minimum_probability=0.0)
        topic_prob_tupel = max(topic_prob_list, key=itemgetter(1))
        cnt1[topic_prob_tupel[0]] += 1
    cnt2 = np.zeros(ldamodel_1.num_topics)
    for doc in corpus_2:
        topic_prob_list = ldamodel_2.get_document_topics(doc, minimum_probability=0.0)
        topic_prob_tupel = max(topic_prob_list, key=itemgetter(1))
        cnt2[topic_prob_tupel[0]] += 1

    return (np.sum(cnt1 * min1) / np.sum(cnt1) + np.sum(cnt2 * min2) / np.sum(cnt2)) / 2


def calc_score():
    topics = np.asarray([2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100])
    modes = ["intersection", "union"]
    model_pairs = [
        (
            "lda-wiki_nt-gpt2_nt/wiki_nt",
            "lda-wiki_nt-gpt2_nt/gpt2_nt"
        ),
        (
            "lda-gpt2_nt-gpt2_nt/gpt2_nt_1",
            "lda-gpt2_nt-gpt2_nt/gpt2_nt_2"
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
        for model_pair in model_pairs:
            for mode in modes:
                for idx, topic in enumerate(topics):
                    path1 = model_pair[0]
                    path2 = model_pair[1]
                    path_ldamodel_1 = f"/cluster/scratch/knobelf/{path1}/{mode}/{topic}/ldamodel_{topic}"
                    path_ldamodel_2 = f"/cluster/scratch/knobelf/{path2}/{mode}/{topic}/ldamodel_{topic}"
                    path_dictionary_1 = f"/cluster/scratch/knobelf/{path1}/{mode}/{topic}/dictionary_{topic}"
                    path_dictionary_2 = f"/cluster/scratch/knobelf/{path2}/{mode}/{topic}/dictionary_{topic}"
                    path_corpus_1 = f"/cluster/scratch/knobelf/{path1}/{mode}/{topic}/corpus_{topic}"
                    path_corpus_2 = f"/cluster/scratch/knobelf/{path2}/{mode}/{topic}/corpus_{topic}"

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

                    score_path = "/cluster/scratch/knobelf/score_by_topic_probability_values.json"

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

                    score_path = "/cluster/scratch/knobelf/score_by_top_topic.json"

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

                    fig, ax = plt.subplots(figsize=(18, 14))
                    data = ax.imshow(mdiff, cmap='RdBu_r', vmin=0.0, vmax=1.0, origin='lower')
                    for axis in [ax.xaxis, ax.yaxis]:
                        axis.set_major_locator(MaxNLocator(integer=True))
                    plt.title(
                        f"Topic difference ({path1.split('/')[1]} - {path2.split('/')[1]} - {mode})[{distance} distance] for {topic} topics")
                    plt.colorbar(data)
                    plt.savefig(f"/cluster/scratch/knobelf/{path1.split('/')[0]}/diff_{short_mode}_{topic}.png", dpi=300)
                    plt.close('all')
                    pbar.update(1)


def main():
    calc_score()


main()
