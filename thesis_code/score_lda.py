import json
import logging
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from gensim import matutils
from gensim.matutils import kullback_leibler, jensen_shannon
from gensim.models import CoherenceModel
from matplotlib.ticker import MaxNLocator
from gensim.models.ldamulticore import LdaMulticore
from tqdm.auto import tqdm
from operator import itemgetter
import itertools
from octis.models.NeuralLDA import NeuralLDA

import train_lda


def diff(topic_model_1, topic_model_2, distance="jensen_shannon", normed=True):
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

    # retrieve topic-word distributions from topic models
    if isinstance(topic_model_1, LdaMulticore) and isinstance(topic_model_2, LdaMulticore):
        d1, d2 = topic_model_1.get_topics(), topic_model_2.get_topics()
    elif isinstance(topic_model_1, NeuralLDA) and isinstance(topic_model_2, NeuralLDA):
        d1, d2 = topic_model_1.model.get_topic_word_mat(), topic_model_2.model.get_topic_word_mat()
    else:
        raise ValueError(f">> Error: topic models are not the same instance")

    t1_size, t2_size = d1.shape[0], d2.shape[0]

    # initialize z
    z = np.zeros((t1_size, t2_size))

    # iterate over each cell in the initialized z
    for topic in np.ndindex(z.shape):
        topic1 = topic[0]
        topic2 = topic[1]

        # calculate jensen-shannon distance
        z[topic] = distance_func(d1[topic1], d2[topic2])
    if normed:
        if np.abs(np.max(z)) > 1e-8:
            z /= np.max(z)
    return z


def score_by_topic_coherence(model, texts, corpus, dictionary, topn=20):
    """
    Calculates c_v coherence score
    Note: This is not working stable if texts/corpus contains empty documents and if there are words that do not appear in the whole corpus.
    Solution: Remove all empty documents on load and edit log_ratio_measure() in direct_confirmation_measure.py and
              _cossim in indirect_confirmation_measure.py in gensim.topic_coherence as there could be division by zero! (add EPSILON to denominator)

    :param model: LdaModel or NeuralLDA model
    :param texts: corpus
    :param corpus: list of list of int
    :param dictionary: dictionary
    :param topn: int, optional
        Integer corresponding to the number of top words to be extracted from each topic.

    :return: int, the score
    """

    # retrieve topic-word distributions from topic models
    if isinstance(model, LdaMulticore):
        topics = model.get_topics()
    elif isinstance(model, NeuralLDA):
        topics = model.model.get_topic_word_mat()
    else:
        raise ValueError(f">> Error: topic model instance not defined")
    topics_ = [matutils.argsort(topic, topn=topn, reverse=True) for topic in topics]

    # calculate coherence score
    score = CoherenceModel(processes=48, topics=topics_, texts=texts, corpus=corpus, dictionary=dictionary, coherence='c_v', topn=topn).get_coherence()
    return score


def score_by_topic_corpus_probability(topic_model_1, topic_model_2, corpus_1=None, corpus_2=None, documents_1=None, documents_2=None, distance='jensen_shannon'):
    """
    Calculates the score by 'distance' and weights the strongest similarities by their topic probability over the whole corpus.
    Note: 1. Is not stable for neural topic models as they are very sensitive on the change of the input value magnitude they are trained on.
          Meaning if they were trained on a word count of 1000 it is unstable to predict on a word count of 10000
          2. The difference of using filtered documents vs unfiltered is less than 0.03
    :param topic_model_1: LdaModel or NeuralLDA model
    :param topic_model_2: LdaModel or NeuralLDA model
    :param corpus_1: list of (int, int)
    :param corpus_2: list of (int, int)
    :param documents_1: list of str
    :param documents_2: list of str
    :param distance: str, distance, e.g. jensen_shannon
    :return: int, the tp score
    """

    # calculate the difference matrix
    mdiff1 = diff(topic_model_1, topic_model_2, distance=distance)
    mdiff2 = diff(topic_model_2, topic_model_1, distance=distance)

    # select the best match
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

    # weigh the best matches by their probability and take the mean of both
    return (np.matmul(topic_corpus_prob_1, min1) + np.matmul(topic_corpus_prob_2, min2)) / 2


def score_by_top_topic_corpus_probability(topic_model_1, topic_model_2, corpus_1, corpus_2, distance='jensen_shannon'):
    """
    Calculates the score by 'distance' and weights the strongest similarities
    by the respective normed sum of the most probable topic for each document over the whole corpus

    :param topic_model_1: LdaModel or NeuralLDA model
    :param topic_model_2: LdaModel or NeuralLDA model
    :param corpus_1: list of (int, int)
    :param corpus_2: list of (int, int)
    :param distance: str, distance, e.g. jensen_shannon
    :return: int, the tt score
    """

    # calculate the difference matrix
    mdiff1 = diff(topic_model_1, topic_model_2, distance=distance)
    mdiff2 = diff(topic_model_2, topic_model_1, distance=distance)
    min1 = np.amin(mdiff1, axis=1)
    min2 = np.amin(mdiff2, axis=1)

    if isinstance(topic_model_1, LdaMulticore) and isinstance(topic_model_2, LdaMulticore):
        from pathos.multiprocessing import ProcessingPool as Pool

        def prob_list(topic_model, corpus):
            # helper function to retrieve the most probable topic per document
            cnt = np.zeros(topic_model.num_topics)
            for doc in corpus:
                topic_prob_list = topic_model.get_document_topics(doc, minimum_probability=0.0)
                topic_prob_tupel = max(topic_prob_list, key=itemgetter(1))
                cnt[topic_prob_tupel[0]] += 1
            return cnt

        workers = 8
        pool = Pool(ncpus=workers)

        # retrieve the most probable topic per document
        logging.info("First split")
        cnt_split = pool.map(lambda x: prob_list(topic_model_1, x), list(train_lda.split(corpus_1, workers)))
        cnt1 = np.sum(cnt_split, axis=0)
        assert len(cnt1) == topic_model_1.num_topics, f">> ERROR: Count changed to {len(cnt1)}"

        logging.info("Second split")
        cnt_split = pool.map(lambda x: prob_list(topic_model_2, x), list(train_lda.split(corpus_2, workers)))
        cnt2 = np.sum(cnt_split, axis=0)
        assert len(cnt2) == topic_model_2.num_topics

    elif isinstance(topic_model_1, NeuralLDA) and isinstance(topic_model_2, NeuralLDA):
        num_topics = topic_model_1.model.num_topics

        cnt1 = np.zeros(num_topics)
        cnt2 = np.zeros(num_topics)

        # retrieve the most probable topic per document
        topic_prob_list = topic_model_1.model.get_thetas(topic_model_1.model.train_data)
        for i in topic_prob_list.argmax(axis=1):
            cnt1[i] += 1

        topic_prob_list = topic_model_2.model.get_thetas(topic_model_2.model.train_data)
        for i in topic_prob_list.argmax(axis=1):
            cnt2[i] += 1

    else:
        raise ValueError(">> Error: topic models are not the same instance")

    # norm the topic counts the maximum number of documents
    topic_corpus_prob_1 = cnt1 / np.sum(cnt1)
    topic_corpus_prob_2 = cnt2 / np.sum(cnt2)

    # weigh the best matches by their topic probability and take the mean of both
    return (np.matmul(topic_corpus_prob_1, min1) + np.matmul(topic_corpus_prob_2, min2)) / 2


def save_score(score_path, score, key, idx, array_length):
    """
    helper function to save the score in a dictionary with a key in a json file
    :param score_path: str, the path to the score file or where the score file should be saved if it does not exist
    :param score: int, the calculated score
    :param key: str, key name of the topic comparison, unique for each comparison
    :param idx: int, index into the array
    :param array_length: int, length of the array (equals the number of topic sizes that were tested)
    """

    if os.path.exists(score_path):
        with open(score_path, "r") as file:
            scores = json.load(file)
    else:

        # create new dict if it does not exist
        scores = dict()
    if key not in scores.keys():

        # create entry if the key is not available
        scores[key] = np.ones(array_length).tolist()
    scores[key][idx] = score

    with open(score_path, "w") as file:
        json.dump(scores, file)
    return


def score_iteration(data_folder_path, score_mode, samples, models, topic_models, num_topics, merge_types):
    """
    Function for calculating the score in a partitioned way to speed up total computation time

    :param data_folder_path: str, path to the data folder
    :param score_mode: str, indicates which part to calculate, "tt", "tp", "cv", "img"
    :param samples: int, number of documents in the corpus
    :param models: list of str, which topic model comparison to calculate
    :param topic_models: list of str, "classic_lda", "nerual_lda"
    :param num_topics: list of int, which topic sizes to score
    :param merge_types: str, "intersection", "union"
    """

    length = len(merge_types) * len(topic_models) * len(num_topics)

    # initiailze progressbar
    with tqdm(total=length) as pbar:

        # name preprocessing
        model1_name = models.split("-")[0]
        model2_name = models.split("-")[1]
        model1_name_ = model1_name
        model2_name_ = model2_name
        if model1_name_ == model2_name_:
            model1_name_ += "_1"
            model2_name_ += "_2"

        sampling_method = "-" + models.split("-")[2] if len(models.split("-")) == 3 else ""
        root_path = f"{data_folder_path}{samples}/{model1_name}-{model2_name}{sampling_method}/"
        for merge_type in merge_types:
            # Load doc, cor, dic
            subroot_path1 = f"{root_path}{model1_name_}/{merge_type}/"
            subroot_path2 = f"{root_path}{model2_name_}/{merge_type}/"

            # load data
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
                    var = True
                    var_idx = 0
                    var_key = ""
                    model1_path = f"{subroot_path1}{topic_model}/{topic}/"
                    model2_path = f"{subroot_path2}{topic_model}/{topic}/"
                    model1_path_ = f"{subroot_path1}{topic_model}/{topic}/"
                    model2_path_ = f"{subroot_path2}{topic_model}/{topic}/"
                    while var:
                        # Load models
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
                            score1 = score_by_topic_coherence(model1, documents1, corpus1, dictionary1)
                            score2 = score_by_topic_coherence(model2, documents2, corpus2, dictionary2)

                            key1 = f"{topic_model}-{models}-{model1_name_}-{merge_type}{var_key}"
                            key2 = f"{topic_model}-{models}-{model2_name_}-{merge_type}{var_key}"

                            score_path = f"{root_path}cv_score.json"
                            save_score(score_path, score1, key1, idx, len(num_topics))
                            save_score(score_path, score2, key2, idx, len(num_topics))

                        elif score_mode == "tt":
                            score = score_by_top_topic_corpus_probability(model1, model2, corpus1, corpus2)
                            score_path = f"{root_path}tt_score.json"
                            key = f"{topic_model}-{models}-{merge_type}{var_key}"
                            save_score(score_path, score, key, idx, len(num_topics))

                        elif score_mode == "tp":
                            score = score_by_topic_corpus_probability(model1, model2, corpus1, corpus2, documents1, documents2)
                            score_path = f"{root_path}tp_score.json"
                            key = f"{topic_model}-{models}-{merge_type}{var_key}"
                            save_score(score_path, score, key, idx, len(num_topics))

                        elif score_mode == "img":
                            # Calculate the distance graph and save it
                            distance = 'jensen_shannon'

                            mdiff = diff(model1, model2, distance=distance)

                            fig, axes = plt.subplots(figsize=(18, 14))
                            data = axes.imshow(mdiff, cmap='RdBu_r', vmin=0.0, vmax=1.0, origin='lower')
                            for axis in [axes.xaxis, axes.yaxis]:
                                axis.set_major_locator(MaxNLocator(integer=True))

                            title = f"Topic Model difference {model1_name} vs. {model2_name} ({topic_model}){var_key}"
                            sampling_name = "multinomial" if len(models.split("-")) < 3 else models.split("-")[2]
                            subtitle = f"({samples} samples, {sampling_name} sampling, {topic} topics, {merge_type} dictionaries, {distance} distance)"
                            plt.suptitle(title, fontsize=15)
                            axes.set_title(subtitle, fontsize=8, x=0.6)

                            plt.colorbar(data)
                            plt.savefig(f"{root_path}/diff_{topic_model}_{merge_type}_{topic}{var_key}.png", dpi=300)
                            plt.close('all')

                        else:
                            raise ValueError(">> ERROR: undefined score_mode")

                        var_idx += 1
                        var_key = f"-{var_idx}"
                        model1_path = f"{model1_path_}{var_idx}/"
                        model2_path = f"{model2_path_}{var_idx}/"
                        if not os.path.exists(f"{model1_path}model") or not os.path.exists(f"{model2_path}model"):
                            var = False

                    pbar.update(1)


def main():
    """
    Command:
        python score_lda.py [data_folder_path] [score_mode] [10000] [models]

        score_mode: str
            cv - This is our score in the thesis for evaluating the quality of topic models
            tt - This is our score in the thesis for comparing topic models
            tp - We do not use this score in our thesis, this was just an experiment
            img - this plots the illustrating plots of the distance matrices
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
    topic_models = ["classic_lda", "neural_lda"] if samples <= 10000 else ["classic_lda"]
    num_topics = [2, 3, 5, 10, 20, 50, 100]
    merge_types = ["intersection", "union"]

    score_iteration(data_folder_path, score_mode, samples, models, topic_models, num_topics, merge_types)

    return


def test():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    score_iteration("./data/", "cv", 10000, "gpt2_nt-wiki_nt", ["classic_lda"], [10], ["intersection"])


if __name__ == '__main__':
    main()
    # test()
