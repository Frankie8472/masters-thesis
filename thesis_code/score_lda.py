import json
import os
import random
import re
import sys
from multiprocessing import freeze_support
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from gensim.models import CoherenceModel, Phrases
from matplotlib.ticker import MaxNLocator
from gensim.models.ldamulticore import LdaMulticore
from gensim.utils import SaveLoad
from nltk import WordNetLemmatizer, RegexpTokenizer
from tqdm.auto import tqdm
from operator import itemgetter
import itertools

from transformers import set_seed


def load_json(filename):
    with open(filename, 'r') as file:
        train_articles = json.load(file)
    return train_articles


def load_wikitext(samples=100000):
    heading_pattern = '( \n [=\s].*[=\s] \n)'
    train_data = Path('/cluster/work/cotterell/knobelf/data/data_wikitext-103-raw/wiki.train.raw').read_text(encoding='utf-8')
    train_split = re.split(heading_pattern, train_data)
    train_headings = [x[7:-7] for x in train_split[1::2]]
    train_articles = [x for x in train_split[2::2]]
    return random.choices(train_articles, k=samples)


def load_arxiv(samples=100000):
    def get_metadata():
        with open('/cluster/work/cotterell/knobelf/data/data_arxiv-metadata-oai-snapshot.json', 'r') as f:
            for line in f:
                yield line

    metadata = get_metadata()
    size = 0
    for paper in metadata:
        size += 1
    choices = random.choices(list(np.arange(size)), k=samples)
    choices.sort()
    metadata = get_metadata()
    step = 0
    idx = 0
    corpus = []
    for paper in metadata:
        if idx >= samples:
            break
        if step == choices[idx]:
            if step != choices[idx+1]:
                step += 1
            corpus.append(json.loads(paper)['abstract'])
            idx += 1
        else:
            step += 1
    return corpus


def tokenize(docs):
    # Split the documents into tokens.
    tokenizer = RegexpTokenizer(r'\w+')

    for idx in range(len(docs)):
        docs[idx] = docs[idx].lower()  # Convert to lowercase.
        docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

    # Remove numbers, but not words that contain numbers.
    docs = [[token for token in doc if not token.isnumeric()] for doc in docs]

    # Remove words that are only one character.
    docs = [[token for token in doc if len(token) > 1] for doc in docs]

    # Lemmatize the documents. Better than stemmer as is easier to read
    lemmatizer = WordNetLemmatizer()
    docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]

    # Add bigrams to docs (only ones that appear 20 times or more).
    bigram = Phrases(docs, min_count=20)
    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                docs[idx].append(token)

    return docs


def score_by_topic_coherence(lda_model, text, dictionary, coherence='c_nmpi'):
    coherence_model_lda = CoherenceModel(model=lda_model, texts=text, dictionary=dictionary, coherence=coherence)
    return coherence_model_lda.get_coherence()


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
            "lda-gpt2_nt-wiki_nt/wiki_nt",
            "lda-gpt2_nt-wiki_nt/gpt2_nt"
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
        ),
        (
            "lda-gpt2_nt-wiki_nt-top_p/wiki_nt",
            "lda-gpt2_nt-wiki_nt-top_p/gpt2_nt"
        ),
        (
            "lda-gpt2_nt-wiki_nt-typ_p/wiki_nt",
            "lda-gpt2_nt-wiki_nt-typ_p/gpt2_nt"
        ),
        (
            "lda-gpt2_nt-gpt2_nt-top_p/gpt2_nt_1",
            "lda-gpt2_nt-gpt2_nt-top_p/gpt2_nt_2"
        ),
        (
            "lda-gpt2_nt-gpt2_nt-typ_p/gpt2_nt_1",
            "lda-gpt2_nt-gpt2_nt-typ_p/gpt2_nt_2"
        ),
        (
            "lda-gpt2_nt-arxiv-top_p/gpt2_nt",
            "lda-gpt2_nt-arxiv-top_p/arxiv"
        ),
        (
            "lda-gpt2_nt-arxiv-typ_p/gpt2_nt",
            "lda-gpt2_nt-arxiv-typ_p/arxiv"
        )
    ]
    length = len(topics) * len(modes) * len(model_pairs)
    with tqdm(total=length) as pbar:
        for model_pair in model_pairs:
            for mode in modes:
                for idx, topic in enumerate(topics):
                    path1 = model_pair[0]
                    path2 = model_pair[1]
                    path_ldamodel_1 = f"/cluster/work/cotterell/knobelf/data/{path1}/{mode}/{topic}/ldamodel_{topic}"
                    path_ldamodel_2 = f"/cluster/work/cotterell/knobelf/data/{path2}/{mode}/{topic}/ldamodel_{topic}"
                    # path_dictionary_1 = f"/cluster/work/cotterell/knobelf/data/{path1}/{mode}/{topic}/dictionary_{topic}"
                    # path_dictionary_2 = f"/cluster/work/cotterell/knobelf/data/{path2}/{mode}/{topic}/dictionary_{topic}"
                    path_corpus_1 = f"/cluster/work/cotterell/knobelf/data/{path1}/{mode}/{topic}/corpus_{topic}"
                    path_corpus_2 = f"/cluster/work/cotterell/knobelf/data/{path2}/{mode}/{topic}/corpus_{topic}"

                    # Load pretrained models from disk.
                    with open(path_corpus_1, 'r') as file:
                        corpus_1 = json.load(file)
                    with open(path_corpus_2, 'r') as file:
                        corpus_2 = json.load(file)
                    # dictionary_1 = SaveLoad.load(path_dictionary_1)
                    # dictionary_2 = SaveLoad.load(path_dictionary_2)
                    ldamodel_1 = LdaMulticore.load(path_ldamodel_1)
                    ldamodel_2 = LdaMulticore.load(path_ldamodel_2)

                    distance = 'jensen_shannon'
                    words = 100000000

                    # Compare models with scores_by_topic_probability and save
                    diff_score = score_by_topic_probability(ldamodel_1, ldamodel_2, corpus_1, corpus_2)

                    score_path = "/cluster/work/cotterell/knobelf/data/score_by_topic_probability_values.json"

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

                    score_path = "/cluster/work/cotterell/knobelf/data/score_by_top_topic.json"

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
                    plt.savefig(f"/cluster/work/cotterell/knobelf/data/{path1.split('/')[0]}/diff_{short_mode}_{topic}.png", dpi=300)
                    plt.close('all')
                    pbar.update(1)


def calc_score_var_same(state=0):
    topics = np.asarray([2, 5, 10])
    modes = ["intersection", "union"]
    model_pairs = [
        (
            "lda-gpt2_nt-wiki_nt/wiki_nt",
            "lda-gpt2_nt-wiki_nt/gpt2_nt"
        ),
        (
            "lda-wiki_nt-arxiv/wiki_nt",
            "lda-wiki_nt-arxiv/arxiv"
        )
    ]
    length = len(topics) * len(modes) * len(model_pairs) * 9
    with tqdm(total=length) as pbar:
        for model_pair in model_pairs:
            path1 = model_pair[0]
            path2 = model_pair[1]

            for mode in modes:
                for idx, topic in enumerate(topics):

                    for i in [6, 7, 8]:
                        path_ldamodel_1 = f"/cluster/work/cotterell/knobelf/data/{path1}/{i}/{mode}/{topic}/ldamodel_{topic}"
                        path_corpus_1 = f"/cluster/work/cotterell/knobelf/data/{path1}/{i}/{mode}/{topic}/corpus_{topic}"
                        # Load pretrained models from disk.
                        with open(path_corpus_1, 'r') as file:
                            corpus_1 = json.load(file)
                        ldamodel_1 = LdaMulticore.load(path_ldamodel_1)

                        b = 6
                        for j in [6, 7, 8]:
                            path_ldamodel_2 = f"/cluster/work/cotterell/knobelf/data/{path2}/{j}/{mode}/{topic}/ldamodel_{topic}"
                            path_corpus_2 = f"/cluster/work/cotterell/knobelf/data/{path2}/{j}/{mode}/{topic}/corpus_{topic}"

                            # Load pretrained models from disk.
                            with open(path_corpus_2, 'r') as file:
                                corpus_2 = json.load(file)
                            ldamodel_2 = LdaMulticore.load(path_ldamodel_2)

                            if state == 0:
                                # Compare models with scores_by_topic_probability and save
                                diff_score = score_by_topic_probability(ldamodel_1, ldamodel_2, corpus_1, corpus_2)

                                score_path = "/cluster/work/cotterell/knobelf/data/score_by_topic_probability_values_var_same.json"

                                if os.path.isfile(score_path):
                                    with open(score_path, 'r') as file:
                                        score_values = json.load(file)
                                else:
                                    score_values = dict()

                                short_mode = "is" if mode == "intersection" else "un"
                                key = f"{path1.split('/')[0]}-{short_mode}"
                                if key not in score_values.keys():
                                    score_values[key] = np.ones((9, topics.shape[0])).tolist()

                                score_values[key][(j-b)+(i-b)*3][idx] = diff_score

                                with open(score_path, 'w') as file:
                                    json.dump(score_values, file)
                            elif state == 1:
                                # Compare models with score_by_top_topic and save
                                diff_score = score_by_top_topic(ldamodel_1, ldamodel_2, corpus_1, corpus_2)

                                score_path = "/cluster/work/cotterell/knobelf/data/score_by_top_topic_var_same.json"

                                if os.path.isfile(score_path):
                                    with open(score_path, 'r') as file:
                                        score_values = json.load(file)
                                else:
                                    score_values = dict()

                                short_mode = "is" if mode == "intersection" else "un"
                                key = f"{path1.split('/')[0]}-{short_mode}"
                                if key not in score_values.keys():
                                    score_values[key] = np.ones((9, topics.shape[0])).tolist()

                                score_values[key][(j-b)+(i-b)*3][idx] = diff_score

                                with open(score_path, 'w') as file:
                                    json.dump(score_values, file)

                            pbar.update(1)


def calc_score_var_diff(state=0):
    topics = np.asarray([2, 5, 10])
    modes = ["intersection", "union"]
    model_pairs = [
        (
            "lda-gpt2_nt-wiki_nt/wiki_nt",
            "lda-gpt2_nt-wiki_nt/gpt2_nt"
        ),
        (
            "lda-wiki_nt-arxiv/wiki_nt",
            "lda-wiki_nt-arxiv/arxiv"
        )
    ]
    length = len(topics) * len(modes) * len(model_pairs) * 5
    with tqdm(total=length) as pbar:
        for model_pair in model_pairs:
            path1 = model_pair[0]
            path2 = model_pair[1]

            for mode in modes:
                for idx, topic in enumerate(topics):

                    for i in [1, 2, 3, 4, 5]:
                        path_ldamodel_1 = f"/cluster/work/cotterell/knobelf/data/{path1}/{i}/{mode}/{topic}/ldamodel_{topic}"
                        path_corpus_1 = f"/cluster/work/cotterell/knobelf/data/{path1}/{i}/{mode}/{topic}/corpus_{topic}"
                        # Load pretrained models from disk.
                        with open(path_corpus_1, 'r') as file:
                            corpus_1 = json.load(file)
                        ldamodel_1 = LdaMulticore.load(path_ldamodel_1)

                        for j in [i]:
                            path_ldamodel_2 = f"/cluster/work/cotterell/knobelf/data/{path2}/{j}/{mode}/{topic}/ldamodel_{topic}"
                            path_corpus_2 = f"/cluster/work/cotterell/knobelf/data/{path2}/{j}/{mode}/{topic}/corpus_{topic}"

                            # Load pretrained models from disk.
                            with open(path_corpus_2, 'r') as file:
                                corpus_2 = json.load(file)
                            ldamodel_2 = LdaMulticore.load(path_ldamodel_2)

                            if state == 0:
                                # Compare models with scores_by_topic_probability and save
                                diff_score = score_by_topic_probability(ldamodel_1, ldamodel_2, corpus_1, corpus_2)

                                score_path = "/cluster/work/cotterell/knobelf/data/score_by_topic_probability_values_var_diff.json"

                                if os.path.isfile(score_path):
                                    with open(score_path, 'r') as file:
                                        score_values = json.load(file)
                                else:
                                    score_values = dict()

                                short_mode = "is" if mode == "intersection" else "un"
                                key = f"{path1.split('/')[0]}-{short_mode}"
                                if key not in score_values.keys():
                                    score_values[key] = np.ones((5, topics.shape[0])).tolist()

                                score_values[key][(j-1)][idx] = diff_score

                                with open(score_path, 'w') as file:
                                    json.dump(score_values, file)
                            elif state == 1:
                                # Compare models with score_by_top_topic and save
                                diff_score = score_by_top_topic(ldamodel_1, ldamodel_2, corpus_1, corpus_2)

                                score_path = "/cluster/work/cotterell/knobelf/data/score_by_top_topic_var_diff.json"

                                if os.path.isfile(score_path):
                                    with open(score_path, 'r') as file:
                                        score_values = json.load(file)
                                else:
                                    score_values = dict()

                                short_mode = "is" if mode == "intersection" else "un"
                                key = f"{path1.split('/')[0]}-{short_mode}"
                                if key not in score_values.keys():
                                    score_values[key] = np.ones((5, topics.shape[0])).tolist()

                                score_values[key][(j-1)][idx] = diff_score

                                with open(score_path, 'w') as file:
                                    json.dump(score_values, file)

                            pbar.update(1)


def calc_score_new(state=0):
    topics = np.asarray([2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100])
    modes = ["intersection", "union"]
    model_pairs = [
        (
            "lda-gpt2-wiki_nt/wiki_nt",
            "lda-gpt2-wiki_nt/gpt2"
        ),

        (
            "lda-gpt2_nt-gpt2/gpt2_nt",
            "lda-gpt2_nt-gpt2/gpt2"
        )
    ]
    length = len(topics) * len(modes) * len(model_pairs)
    with tqdm(total=length) as pbar:
        for model_pair in model_pairs:
            for mode in modes:
                for idx, topic in enumerate(topics):
                    path1 = model_pair[0]
                    path2 = model_pair[1]
                    path_ldamodel_1 = f"/cluster/work/cotterell/knobelf/data/{path1}/{mode}/{topic}/ldamodel_{topic}"
                    path_ldamodel_2 = f"/cluster/work/cotterell/knobelf/data/{path2}/{mode}/{topic}/ldamodel_{topic}"
                    path_corpus_1 = f"/cluster/work/cotterell/knobelf/data/{path1}/{mode}/{topic}/corpus_{topic}"
                    path_corpus_2 = f"/cluster/work/cotterell/knobelf/data/{path2}/{mode}/{topic}/corpus_{topic}"

                    # Load pretrained models from disk.
                    with open(path_corpus_1, 'r') as file:
                        corpus_1 = json.load(file)
                    with open(path_corpus_2, 'r') as file:
                        corpus_2 = json.load(file)
                    ldamodel_1 = LdaMulticore.load(path_ldamodel_1)
                    ldamodel_2 = LdaMulticore.load(path_ldamodel_2)

                    distance = 'jensen_shannon'
                    words = 100000000
                    short_mode = "is" if mode == "intersection" else "un"

                    if state == 0:
                        # Compare models with scores_by_topic_probability and save
                        diff_score = score_by_topic_probability(ldamodel_1, ldamodel_2, corpus_1, corpus_2)

                        score_path = "/cluster/work/cotterell/knobelf/data/score_by_topic_probability_values.json"

                        if os.path.isfile(score_path):
                            with open(score_path, 'r') as file:
                                score_values = json.load(file)
                        else:
                            score_values = dict()

                        key = f"{path1.split('/')[0]}-{short_mode}"
                        if key not in score_values.keys():
                            score_values[key] = np.ones(topics.shape).tolist()

                        score_values[key][idx] = diff_score

                        with open(score_path, 'w') as file:
                            json.dump(score_values, file)
                    elif state == 1:
                        # Compare models with score_by_top_topic and save
                        diff_score = score_by_top_topic(ldamodel_1, ldamodel_2, corpus_1, corpus_2)

                        score_path = "/cluster/work/cotterell/knobelf/data/score_by_top_topic.json"

                        if os.path.isfile(score_path):
                            with open(score_path, 'r') as file:
                                score_values = json.load(file)
                        else:
                            score_values = dict()

                        key = f"{path1.split('/')[0]}-{short_mode}"
                        if key not in score_values.keys():
                            score_values[key] = np.ones(topics.shape).tolist()

                        score_values[key][idx] = diff_score

                        with open(score_path, 'w') as file:
                            json.dump(score_values, file)
                    elif state == 2:
                        # Calculate Difference Graph and save it
                        mdiff, annotation = ldamodel_1.diff(ldamodel_2, distance=distance, num_words=words)

                        fig, ax = plt.subplots(figsize=(18, 14))
                        data = ax.imshow(mdiff, cmap='RdBu_r', vmin=0.0, vmax=1.0, origin='lower')
                        for axis in [ax.xaxis, ax.yaxis]:
                            axis.set_major_locator(MaxNLocator(integer=True))
                        plt.title(
                            f"Topic difference ({path1.split('/')[1]} - {path2.split('/')[1]} - {mode})[{distance} distance] for {topic} topics")
                        plt.colorbar(data)
                        plt.savefig(f"/cluster/work/cotterell/knobelf/data/{path1.split('/')[0]}/diff_{short_mode}_{topic}.png", dpi=300)
                        plt.close('all')
                    pbar.update(1)


def get_all_coherences():
    set_seed(42)

    # Load all texts
    with tqdm(total=5, desc='Load texts') as pbar:
        text_wiki = load_wikitext()
        pbar.update(1)
        text_arxiv = load_arxiv()
        pbar.update(1)
        text_gpt2_nt = load_json(f"/cluster/work/cotterell/knobelf/data/dataset1-gpt2-wiki_nt.json")
        pbar.update(1)
        text_gpt2_nt_top_p = load_json(f"/cluster/work/cotterell/knobelf/data/dataset1-gpt2-wiki_nt-top_p.json")
        pbar.update(1)
        text_gpt2_nt_typ_p = load_json(f"/cluster/work/cotterell/knobelf/data/dataset1-gpt2-wiki_nt-typ_p.json")
        pbar.update(1)

    # Tokenize all texts
    with tqdm(total=5, desc='Tokenize texts') as pbar:
        text_wiki = tokenize(text_wiki)
        pbar.update(1)
        text_arxiv = tokenize(text_arxiv)
        pbar.update(1)
        text_gpt2_nt = tokenize(text_gpt2_nt)
        pbar.update(1)
        text_gpt2_nt_top_p = tokenize(text_gpt2_nt_top_p)
        pbar.update(1)
        text_gpt2_nt_typ_p = tokenize(text_gpt2_nt_typ_p)
        pbar.update(1)

    # Iterate through all relevant models and save in json file
    topics = np.asarray([2, 5, 10, 20, 50, 100])
    modes = ["intersection", "union"]
    models = [
        ('lda-gpt2_nt-wiki_nt/gpt2_nt', 'gpt2_nt', text_gpt2_nt),
        ('lda-gpt2_nt-wiki_nt/wiki_nt', 'wiki_nt', text_wiki),
        ('lda-gpt2_nt-wiki_nt-top_p/gpt2_nt', 'gpt2_nt-top_p', text_gpt2_nt_top_p),
        ('lda-gpt2_nt-wiki_nt-typ_p/gpt2_nt', 'gpt2_nt-typ_p', text_gpt2_nt_typ_p),
        ('lda-gpt2_nt-arxiv/arxiv', 'arxiv', text_arxiv)
    ]

    length = len(topics) * len(modes) * len(models)
    with tqdm(total=length) as pbar:
        for path, name, text in models:
            for mode in modes:
                for idx, topic in enumerate(topics):
                    path_ldamodel = f"/cluster/work/cotterell/knobelf/data/{path}/{mode}/{topic}/ldamodel_{topic}"
                    path_dictionary = f"/cluster/work/cotterell/knobelf/data/{path}/{mode}/{topic}/dictionary_{topic}"
                    ldamodel = LdaMulticore.load(path_ldamodel)
                    dictionary = SaveLoad.load(path_dictionary)

                    score = score_by_topic_coherence(ldamodel, text, dictionary, 'c_uci')

                    score_path = "/cluster/work/cotterell/knobelf/data/score_by_topic_coherence_uci.json"

                    if os.path.isfile(score_path):
                        with open(score_path, 'r') as file:
                            score_values = json.load(file)
                    else:
                        score_values = dict()

                    short_mode = "is" if mode == "intersection" else "un"

                    key = f"{name}-{short_mode}"

                    if key not in score_values.keys():
                        score_values[key] = np.ones(topics.shape).tolist()

                    score_values[key][idx] = score

                    with open(score_path, 'w') as file:
                        json.dump(score_values, file)

                    pbar.update(1)


def main():
    if len(sys.argv) > 1:
        calc_score_new(int(sys.argv[1]))


if __name__ == '__main__':
    freeze_support()
    main()
