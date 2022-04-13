import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from gensim.matutils import kullback_leibler, jensen_shannon
from gensim.models import CoherenceModel
from matplotlib.ticker import MaxNLocator
from gensim.models.ldamulticore import LdaMulticore
from gensim.utils import SaveLoad
from tqdm.auto import tqdm
from operator import itemgetter
import itertools
from transformers import set_seed
from octis.models.NeuralLDA import NeuralLDA

import train_lda


def diff(topic_model_1, topic_model_2, distance="kullback_leibler", normed=True):
    """Calculate the difference in topic distributions between two models.

    Parameters
    ----------
    topic_model_1 : numpy.ndarray
        The probability for each word in each topic, shape (`num_topics`, `vocabulary_size`).
    topic_model_2 : numpy.ndarray
        The probability for each word in each topic, shape (`num_topics`, `vocabulary_size`).
    distance : {'kullback_leibler', 'jensen_shannon'}
        The distance metric to calculate the difference with.
    normed : bool, optional
        Whether the matrix should be normalized or not.

    Returns
    -------
    numpy.ndarray
        A difference matrix. Each element corresponds to the difference between the two topics,
        shape (`self.num_topics`, `other.num_topics`)
    """

    distances = {
        "kullback_leibler": kullback_leibler,
        "jensen_shannon": jensen_shannon
    }

    if distance not in distances:
        valid_keys = ", ".join("`{}`".format(x) for x in distances.keys())
        raise ValueError("Incorrect distance, valid only {}".format(valid_keys))

    distance_func = distances[distance]

    if isinstance(topic_model_1, LdaMulticore) and isinstance(topic_model_2, LdaMulticore):
        d1, d2 = topic_model_1.get_topics(), topic_model_2.get_topics()
    elif isinstance(topic_model_1, NeuralLDA) and isinstance(topic_model_2, NeuralLDA):
        d1, d2 = topic_model_1.model.get_topic_word_mat(), topic_model_2.model.get_topic_word_mat()
    else:
        raise ValueError(">> Error: topic models are not the same instance")

    t1_size, t2_size = d1.shape[0], d2.shape[0]

    # initialize z and annotation matrix
    z = np.zeros((t1_size, t2_size))

    # iterate over each cell in the initialized z and annotation
    for topic in np.ndindex(z.shape):
        topic1 = topic[0]
        topic2 = topic[1]
        z[topic] = distance_func(d1[topic1], d2[topic2])
    if normed:
        if np.abs(np.max(z)) > 1e-8:
            z /= np.max(z)
    return z


def score_by_topic_coherence(model, texts, dictionary, topn=20):
    """
    Calculates c_v coherence score

    :param model: LdaModel or NeuralLDA model
    :param texts: corpus
    :param dictionary: dictionary
    :param topn: int, optional
        Integer corresponding to the number of top words to be extracted from each topic.

    :return:
    """
    if isinstance(model, LdaMulticore):
        score = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v', topn=topn).get_coherence()
    elif isinstance(model, NeuralLDA):
        score = CoherenceModel(topics=model.model.get_topics(k=topn), texts=texts, dictionary=dictionary, coherence='c_v', topn=topn).get_coherence()
    else:
        raise ValueError(">> Error: topic models are not the same instance")
    return score


def score_by_topic_corpus_probability(topic_model_1, topic_model_2, corpus_1=None, corpus_2=None, documents_1=None, documents_2=None, distance='jensen_shannon'):
    """
    Calculates the score by 'distance' and weights the strongest similarities by their topic probability over the whole corpus.

    :param topic_model_1:
    :param topic_model_2:
    :param corpus_1:
    :param corpus_2:
    :param documents_1:
    :param documents_2:
    :param distance:
    :return:
    """
    mdiff1 = diff(topic_model_1, topic_model_2, distance=distance)
    mdiff2 = diff(topic_model_2, topic_model_1, distance=distance)
    min1 = np.amin(mdiff1, axis=1)  # smaller ~ more similar, take the most similar score for each topic
    min2 = np.amin(mdiff2, axis=1)  # smaller ~ more similar, take the most similar score for each topic

    if isinstance(topic_model_1, LdaMulticore) and isinstance(topic_model_2, LdaMulticore):
        assert topic_model_1.num_topics == topic_model_2.num_topics, ">> ERROR: Not the same amount of topics"
        assert corpus_1 is not None and corpus_2 is not None, ">> ERROR: At least one 'corpus' is None"
        num_topics = topic_model_1.num_topics

        topic_corpus_prob_1 = np.zeros(num_topics)
        topic_corpus_prob_2 = np.zeros(num_topics)

        # Get topic probability distribution for the whole corpus
        probas_1 = topic_model_1.get_document_topics(list(itertools.chain.from_iterable(corpus_1)), minimum_probability=0.0)
        probas_2 = topic_model_2.get_document_topics(list(itertools.chain.from_iterable(corpus_2)), minimum_probability=0.0)
        for key, val in probas_1:
            topic_corpus_prob_1[key] = val
        for key, val in probas_2:
            topic_corpus_prob_2[key] = val

    elif isinstance(topic_model_1, NeuralLDA) and isinstance(topic_model_2, NeuralLDA):
        assert topic_model_1.model.num_topics == topic_model_2.model.num_topics, ">> ERROR: Not the same amount of topics"
        assert documents_1 is not None and documents_2 is not None, ">> ERROR: At least one 'documents' is None"

        # Concatenate all documents and calculate topic probability distribution for the whole corpus
        data_corpus = [' '.join(list(itertools.chain.from_iterable(documents_1)))]
        x_train, input_size = topic_model_1.preprocess(topic_model_1.vocab, train=data_corpus)
        topic_corpus_prob_1 = topic_model_1.model.get_thetas(x_train).T

        data_corpus = [' '.join(list(itertools.chain.from_iterable(documents_2)))]
        x_train, input_size = topic_model_2.preprocess(topic_model_2.vocab, train=data_corpus)
        topic_corpus_prob_2 = topic_model_2.model.get_thetas(x_train).T

    else:
        raise ValueError(">> Error: topic models are not the same instance")

    return (np.sum(topic_corpus_prob_1 * min1) + np.sum(topic_corpus_prob_2 * min2)) / 2


def score_by_top_topic_corpus_probability(topic_model_1, topic_model_2, corpus_1, corpus_2, distance='jensen_shannon'):
    """
    Calculates the score by 'distance' and weights the strongest similarities
    by the respective normed sum of the most probable topic for each document over the whole corpus

    :param topic_model_1:
    :param topic_model_2:
    :param corpus_1:
    :param corpus_2:
    :param distance:
    :return:
    """
    mdiff1 = diff(topic_model_1, topic_model_2, distance=distance)
    mdiff2 = diff(topic_model_2, topic_model_1, distance=distance)
    min1 = np.amin(mdiff1, axis=1)
    min2 = np.amin(mdiff2, axis=1)

    if isinstance(topic_model_1, LdaMulticore) and isinstance(topic_model_2, LdaMulticore):
        num_topics = topic_model_1.num_topics

        cnt1 = np.zeros(num_topics)
        cnt2 = np.zeros(num_topics)

        for doc in corpus_1:
            topic_prob_list = topic_model_1.get_document_topics(doc, minimum_probability=0.0)
            topic_prob_tupel = max(topic_prob_list, key=itemgetter(1))
            cnt1[topic_prob_tupel[0]] += 1

        for doc in corpus_2:
            topic_prob_list = topic_model_2.get_document_topics(doc, minimum_probability=0.0)
            topic_prob_tupel = max(topic_prob_list, key=itemgetter(1))
            cnt2[topic_prob_tupel[0]] += 1
    elif isinstance(topic_model_1, NeuralLDA) and isinstance(topic_model_2, NeuralLDA):
        num_topics = topic_model_1.model.num_topics

        cnt1 = np.zeros(num_topics)
        cnt2 = np.zeros(num_topics)

        topic_prob_list = topic_model_1.model.get_thetas(topic_model_1.model.train_data)
        for i in topic_prob_list.argmax(axis=1):
            cnt1[i] += 1

        topic_prob_list = topic_model_2.model.get_thetas(topic_model_2.model.train_data)
        for i in topic_prob_list.argmax(axis=1):
            cnt2[i] += 1

    else:
        raise ValueError(">> Error: topic models are not the same instance")

    return (np.sum(cnt1 * min1) / np.sum(cnt1) + np.sum(cnt2 * min2) / np.sum(cnt2)) / 2


def save_score(score_path, score, key, idx, array_length):
    if os.path.exists(score_path):
        with open(score_path, "r") as file:
            scores = json.load(file)
    else:
        scores = dict()
    if key not in scores.keys():
        scores[key] = np.ones(len(array_length))
    scores[key][idx] = score
    with open(score_path, "w") as file:
        json.dump(scores, file)
    return


def main():
    """
    Command:
        python score_lda.py [data_folder_path] [score_mode] [10000] [models]

        score_mode: str
            cv
            tt (top_topic
            tp
        samples: int
            10000, 100000
        models:
            trafo_xl_nt-trafo_xl_nt, gpt2_nt-gpt2_nt-typ_p, gpt2_nt-trafo_xl_nt-typ_p, gpt2_nt-wiki_nt-typ_p, gpt2_nt-arxiv-typ_p
            gpt2_nt-gpt2-typ_p, gpt2_nt-trafo_xl-typ_p, gpt2_nt-wiki_nt-top_p, gpt2_nt-arxiv-top_p, gpt2_nt-trafo_xl-top_p, gpt2-wiki_nt
            trafo_xl-wiki_nt, gpt2_nt-trafo_xl_nt-top_p, gpt2_nt-arxiv, gpt2_nt-gpt2_nt-top_p, gpt2_nt-gpt2-top_p, gpt2_nt-wiki_nt, arxiv-arxiv
            trafo_xl_nt-wiki_nt, wiki_nt-arxiv, wiki_nt-wiki_nt, trafo_xl_nt-trafo_xl, trafo_xl-arxiv, gpt2-trafo_xl_nt, trafo_xl-trafo_xl
            trafo_xl_nt-arxiv, gpt2-gpt2, gpt2-arxiv, gpt2_nt-trafo_xl, gpt2-trafo_xl, gpt2_nt-trafo_xl_nt, gpt2_nt-gpt2_nt, gpt2_nt-gpt2
    """
    if len(sys.argv) < 5:
        raise ValueError(f">> ERROR: Wrong number of arguments")

    data_folder_path = sys.argv[1]

    if data_folder_path[-1] != '/':
        data_folder_path += '/'

    score_mode = sys.argv[2]
    samples = int(sys.argv[3])
    models = sys.argv[4]
    topic_models = ["classic_lda", "neural_lda"],
    num_topics = [2, 3, 5, 10, 20, 50, 100],

    merge_types = ["intersection", "union"],

    length = len(merge_types) * len(topic_models) * len(num_topics)
    with tqdm(total=length) as pbar:
        model1_name = models.split("-")[0]
        model2_name = models.split("-")[1]
        model1_name_ = model1_name
        model2_name_ = model2_name
        if model1_name_ == model2_name_:
            model1_name_ += "_1"
            model2_name_ += "_2"

        sampling_method = "" if models.split("-")[2] == "multinomial" else "-" + models.split("-")[2]
        root_path = f"{data_folder_path}{samples}/{model1_name}-{model2_name}{sampling_method}/"
        for merge_type in merge_types:
            # Load doc, cor, dic
            subroot_path1 = f"{root_path}{model1_name_}/{merge_type}/"
            subroot_path2 = f"{root_path}{model2_name_}/{merge_type}/"
            documents1, dictionary1, corpus1 = train_lda.load_data(
                docs_path=f"{subroot_path1}documents",
                dic_path=f"{subroot_path1}dictionary",
                cor_path=f"{subroot_path1}corpus"
            )
            documents2, dictionary2, corpus2 = train_lda.load_data(
                docs_path=f"{subroot_path2}documents",
                dic_path=f"{subroot_path2}dictionary",
                cor_path=f"{subroot_path2}corpus"
            )

            for topic_model in topic_models:
                for idx, topic in enumerate(num_topics):
                    # Load models
                    model1_path = f"{subroot_path1}{topic_model}/{topic}/"
                    model2_path = f"{subroot_path1}{topic_model}/{topic}/"

                    if topic_model == "classic_lda":
                        model1 = LdaMulticore.load(f"{model1_path}model")
                        model2 = LdaMulticore.load(f"{model2_path}model")
                    elif topic_model == "neural_lda":
                        model1 = train_lda.decompress_pickle(f"{model1_path}model")
                        model2 = train_lda.decompress_pickle(f"{model2_path}model")
                    else:
                        raise ValueError(">> ERROR: undefined score_mode")

                    # Score models
                    if score_mode == "cv":
                        score1 = score_by_topic_coherence(model1_name, documents1, dictionary1)
                        score2 = score_by_topic_coherence(model2_name, documents2, dictionary2)

                        key1 = f"{topic_model}-{models}-{model1_name_}-{merge_type}"
                        key2 = f"{topic_model}-{models}-{model2_name_}-{merge_type}"

                        score_path = f"{root_path}cv_score.json"
                        save_score(score_path, score1, key1, idx, len(num_topics))
                        save_score(score_path, score2, key2, idx, len(num_topics))

                    elif score_mode == "tt":
                        score = score_by_top_topic_corpus_probability(model1, model2, corpus1, corpus2)
                        score_path = f"{root_path}tt_score.json"
                        key = f"{topic_model}-{models}-{merge_type}"
                        save_score(score_path, score, key, idx, len(num_topics))

                    elif score_mode == "tp":
                        score = score_by_topic_corpus_probability(model1, model2, corpus1, corpus2, documents1, documents2)
                        score_path = f"{root_path}tp_score.json"
                        key = f"{topic_model}-{models}-{merge_type}"
                        save_score(score_path, score, key, idx, len(num_topics))

                    else:
                        raise ValueError(">> ERROR: undefined score_mode")

                    pbar.update(1)
    return


def calc_diff_score(data_path, models, mode=0):
    if mode < 0 or mode > 2:
        print(f">> ERROR: invalid mode")
        return

    if data_path[-1] != '/':
        data_path += '/'

    topics = np.asarray([2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100])
    combinations = ["intersection", "union"]

    length = len(models) * len(combinations) * len(topics)
    with tqdm(total=length) as pbar:
        for model in models:
            split = model.split("-")
            first = split[1]
            second = split[2]
            if first == second:
                first += "_1"
                second += "_2"
            sampling = "multinomial"
            if len(split) == 4:
                sampling = split[3]
            # model_title = f"LDA: {first} vs. {second} with {sampling} sampling"

            for combination in combinations:
                for idx, topic in enumerate(topics):
                    pbar.update(1)
                    base_path_1 = f"{data_path}{model}/{first}/{combination}/{topic}/"
                    base_path_2 = f"{data_path}{model}/{second}/{combination}/{topic}/"
                    path_ldamodel_1 = f"{base_path_1}ldamodel_{topic}"
                    path_ldamodel_2 = f"{base_path_2}ldamodel_{topic}"
                    # path_dictionary_1 = f"{base_path_1}dictionary_{topic}"
                    # path_dictionary_2 = f"{base_path_2}dictionary_{topic}"
                    path_corpus_1 = f"{base_path_1}corpus_{topic}"
                    path_corpus_2 = f"{base_path_2}corpus_{topic}"

                    # Load pretrained models from disk.
                    with open(path_corpus_1, 'r') as file:
                        corpus_1 = json.load(file)
                    with open(path_corpus_2, 'r') as file:
                        corpus_2 = json.load(file)
                    # dictionary_1 = SaveLoad.load(path_dictionary_1)
                    # dictionary_2 = SaveLoad.load(path_dictionary_2)
                    ldamodel_1 = LdaMulticore.load(path_ldamodel_1)
                    ldamodel_2 = LdaMulticore.load(path_ldamodel_2)

                    short_mode = "is" if mode == "intersection" else "un"

                    if mode == 0:
                        # Calculate the Difference Graph and save it
                        distance = 'jensen_shannon'
                        words = 100000000

                        mdiff, annotation = ldamodel_1.diff(ldamodel_2, distance=distance, num_words=words)

                        fig, axes = plt.subplots(figsize=(18, 14))
                        data = axes.imshow(mdiff, cmap='RdBu_r', vmin=0.0, vmax=1.0, origin='lower')
                        for axis in [axes.xaxis, axes.yaxis]:
                            axis.set_major_locator(MaxNLocator(integer=True))

                        title = f"Topic Model difference {first} vs. {second} (LDA)"
                        subtitle = f"({sampling} sampling, {topic} topics, {combination} dictionaries, {distance} distance)"
                        plt.suptitle(title, fontsize=15)
                        axes.set_title(subtitle, fontsize=8, x=0.6)

                        plt.colorbar(data)
                        plt.savefig(f"{data_path}{model}/diff_{short_mode}_{topic}.png", dpi=300)
                        plt.close('all')
                        continue

                    elif mode == 1:
                        # Compare models with scores_by_topic_probability and save
                        diff_score = score_by_topic_probability(ldamodel_1, ldamodel_2, corpus_1, corpus_2)

                        score_path = f"{data_path}score_by_topic_probability.json"

                    elif mode == 2:
                        # Compare models with score_by_top_topic and save
                        diff_score = score_by_top_topic(ldamodel_1, ldamodel_2, corpus_1, corpus_2)

                        score_path = f"{data_path}score_by_top_topic.json"

                    if os.path.isfile(score_path):
                        with open(score_path, 'r') as file:
                            score_values = json.load(file)
                    else:
                        score_values = dict()

                    short_mode = "is" if mode == "intersection" else "un"
                    key = f"{model}-{short_mode}"
                    if key not in score_values.keys():
                        score_values[key] = np.ones(topics.shape).tolist()

                    score_values[key][idx] = diff_score

                    with open(score_path, 'w') as file:
                        json.dump(score_values, file, indent=2)


def calc_diff_score_all(data_path, state=0):
    models = list()
    first = ["gpt2_nt", "gpt2", "wiki_nt", "arxiv"]
    second = first.copy()
    for i in first:
        for j in second:
            models.append(f"lda-{i}-{j}")
            if i == "gpt2_nt" and j in ["gpt2_nt", "wiki_nt", "arxiv"]:
                for k in ["typ_p", "top_p"]:
                    models.append(f"lda-{i}-{j}-{k}")
        second.remove(i)

    calc_diff_score(data_path, models, state)


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
                                    json.dump(score_values, file, indent=2)
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
                                    json.dump(score_values, file, indent=2)

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
                                    json.dump(score_values, file, indent=2)
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
                                    json.dump(score_values, file, indent=2)

                            pbar.update(1)


def calc_coherence_score(data_path):
    set_seed(42)

    # Load all texts
    with tqdm(total=5, desc='Load texts') as pbar:
        text_wiki = train_lda.load_wikitext(data_path)
        pbar.update(1)
        text_arxiv = train_lda.load_arxiv(data_path)
        pbar.update(1)
        text_gpt2_nt = train_lda.load_json(f"{data_path}dataset1-gpt2-wiki_nt.json")
        pbar.update(1)
        text_gpt2_nt_top_p = train_lda.load_json(f"{data_path}dataset1-gpt2-wiki_nt-top_p.json")
        pbar.update(1)
        text_gpt2_nt_typ_p = train_lda.load_json(f"{data_path}dataset1-gpt2-wiki_nt-typ_p.json")
        pbar.update(1)

    # Tokenize all texts
    with tqdm(total=5, desc='Tokenize texts') as pbar:
        text_wiki = train_lda.tokenize_text(text_wiki, add_bigrams=False, add_trigrams=False)
        pbar.update(1)
        text_arxiv = train_lda.tokenize_text(text_arxiv, add_bigrams=False, add_trigrams=False)
        pbar.update(1)
        text_gpt2_nt = train_lda.tokenize_text(text_gpt2_nt, add_bigrams=False, add_trigrams=False)
        pbar.update(1)
        text_gpt2_nt_top_p = train_lda.tokenize_text(text_gpt2_nt_top_p, add_bigrams=False, add_trigrams=False)
        pbar.update(1)
        text_gpt2_nt_typ_p = train_lda.tokenize_text(text_gpt2_nt_typ_p, add_bigrams=False, add_trigrams=False)
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
                    path_ldamodel = f"{data_path}{path}/{mode}/{topic}/ldamodel_{topic}"
                    path_dictionary = f"{data_path}{path}/{mode}/{topic}/dictionary_{topic}"
                    ldamodel = LdaMulticore.load(path_ldamodel)
                    dictionary = SaveLoad.load(path_dictionary)

                    score_type = 'c_v'
                    score = score_by_topic_coherence(ldamodel, text, dictionary, score_type)
                    score_path = f"{data_path}score_by_topic_coherence_{score_type}.json"

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
                        json.dump(score_values, file, indent=2)

                    pbar.update(1)


if __name__ == '__main__':
    main()
