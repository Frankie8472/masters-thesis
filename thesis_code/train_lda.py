import itertools
import logging
import os
import random
import numpy as np
import sys
from pathlib import Path
import re
import json
from gensim.utils import SaveLoad
from pprint import pprint
from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore
from nltk import WordNetLemmatizer, RegexpTokenizer
import _pickle as cPickle
import bz2


def compressed_pickle(title, data):
    """
    Function used for compressing and saving data
    :param title: str, filepath without ending
    :param data: any, data
    """
    with bz2.BZ2File(title + ".pbz2", "w") as f:
        cPickle.dump(data, f)


def decompress_pickle(file):
    """
    Function used for loading compressed data
    :param file: str, filepath without ending
    :return: any, loaded data
    """
    data = bz2.BZ2File(file + ".pbz2", "rb")
    data = cPickle.load(data)
    return data


def load_wikitext(path, samples=100000):
    """
    Loads the wikitext corpus by random choosing with replacement from the full corpus
    :param path: str, path to the data folder
    :param samples: int, number of documents to return
    :return: list of str, wikitext corpus
    """
    heading_pattern = '( \n [=\s].*[=\s] \n)'
    train_data = Path(f'{path}datasets/data_wikitext-103-raw/wiki.train.raw').read_text(encoding='utf-8')
    train_split = re.split(heading_pattern, train_data)
    train_headings = [x[7:-7] for x in train_split[1::2]]
    train_articles = [x for x in train_split[2::2]]
    return random.choices(train_articles, k=samples)


def load_arxiv(path, samples=100000):
    """
    Loads the arxiv corpus by random choosing with replacement from the full corpus

    Can lead to freezing in jupyter notebooks, run only in a normal python file

    :param path: str, path to the data folder
    :param samples: int, number of documents to return
    :return: list of str, arxiv corpus
    """

    # preparing a generator as the arxiv file is very large
    def get_metadata():
        with open(f'{path}datasets/data_arxiv-metadata-oai-snapshot.json', 'r') as f:
            for line in f:
                yield line

    metadata = get_metadata()
    corpus = []

    # extracting all abstracts
    for paper in metadata:
        corpus.append(json.loads(paper)['abstract'])
    return random.choices(corpus, k=samples)


def load_json(filename, samples=100000):
    """
    Loads the generated corpus from a json file by picking the first n samples, random chosing is not neccessary as the sampling process from a language model is random
    :param filename: str, filepath and name
    :param samples: int, number of documents to return
    :return: list of str, corpus
    """
    with open(filename, 'r') as file:
        train_articles = json.load(file)
    return train_articles[:min(samples, len(train_articles))]


def load_dataset(data_path, set_name, sampling_method, samples=100000):
    """
    Defines the parameters of the datasets from the thesis to be loaded accordingly
    :param data_path: str, path to the "data" folder
    :param set_name: str, "arxiv", "gpt-2","wikitext", "gpt-2_nt", "trafo_xl_nt", "trafo_xl"
    :param sampling_method: str, "multinomial", "top-p", "typ-p"
    :param samples: int, number of documents in a corpus
    :return: list of str, the loaded dataset
    """
    random.seed(42)
    np.random.seed(42)

    if sampling_method == "multinomial":
        sampling = ""
    else:
        sampling = "-" + sampling_method

    dataset = "dataset1"
    if set_name[-2:] == "_2":
        dataset = "dataset2"

    docs = None
    if "arxiv" in set_name:
        if set_name[-2:] == "_2":
            random.seed(1337)
            np.random.seed(1337)
        docs = load_arxiv(data_path, samples)
    elif "wiki_nt" in set_name:
        if set_name[-2:] == "_2":
            random.seed(1337)
            np.random.seed(1337)
        docs = load_wikitext(data_path, samples)
    elif "gpt2_nt" in set_name:
        docs = load_json(f"{data_path}datasets/{dataset}-gpt2-wiki_nt{sampling}.json", samples)
    elif "gpt2" in set_name:
        docs = load_json(f"{data_path}datasets/{dataset}-gpt2.json", samples)
    elif "trafo_xl_nt" in set_name:
        docs = load_json(f"{data_path}datasets/{dataset}-trafo_xl-wiki_nt{sampling}.json", samples)
    elif "trafo_xl" in set_name:
        docs = load_json(f"{data_path}datasets/{dataset}-trafo_xl.json", samples)

    assert len(docs) == samples
    return docs


def tokenize_text(docs, add_bigrams=True, add_trigrams=True):
    """
    Used for tokenizing documents for topic model generation
    :param docs: list of str, the corpus
    :param add_bigrams: bool, if bigrams should be added
    :param add_trigrams: bool, if trigrams should be added
    :return:
    """
    # Tokenize the documents.
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
    bigram = Phrases(docs, min_count=5, threshold=10) if add_bigrams or add_trigrams else None

    trigram = Phrases(bigram[docs], min_count=5, threshold=10) if add_trigrams else None

    if add_bigrams or add_trigrams:
        for idx in range(len(docs)):
            doc = docs[idx]
            bigrams = list()
            trigrams = list()
            if add_bigrams:
                for token in bigram[doc]:
                    if '_' in token:
                        # Token is a bigram, add to document.
                        bigrams.append(token)
            if add_trigrams:
                for token in trigram[bigram[doc]]:
                    cnt = token.count('_')
                    if cnt == 2:
                        # Token is a trigram, add to document.
                        trigrams.append(token)

            if add_bigrams:
                for token in bigrams:
                    docs[idx].append(token)

            if add_trigrams:
                for token in trigrams:
                    docs[idx].append(token)

    return docs


def tokenize_create_dictionary(docs):
    """
    Helper function for creating dictionaries and filtering extreme word occurrences (min and max)
    :param docs:
    :return: list of str, gensim.corpora.dictionary.Dictionary
    """
    # Remove rare and common tokens.
    # Create a dictionary representation of the documents.
    dic = Dictionary(docs)

    # Filter out words that occur less than 20 documents, or more than 50% of the documents.
    dic.filter_extremes(no_below=20, no_above=0.5)

    return docs, dic


def split(a, n):
    """
    Helper function for splitting up list of any into equal parts (apart from the last one, the rest)
    :param a: list of any, the list of aby to be splitted
    :param n: int, the number of splits
    :return: list of list of any, splitet list of any
    """
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def filter_docs(docs, dic):
    """
    Helper function for filtering documents if the word occurs in the dictionary
    :param docs: list of list of int, the tokenized corpus
    :param dic: list of int, the dictionary
    :return: list of list of int, the content reduced corpus to the words in the dictionary
    """
    for i, _ in enumerate(docs):
        docs[i] = [word for word in docs[i] if word in dic]
    return docs


def tokenize_bow_dual(docs0, docs1, union=False, workers=None, return_filtered_docs=False):
    """
    Function for creating two dictionaries (dic0, dic1), two tokenized corpora (cor0, cor1) and two filtered non-tokenized corpora (docs0, docs1) out of two corpora (docs0, docs1)

    cor is of type "list of (int, int)",
    doc is of type "list of str",
    dic is of type "gensim.corpora.dictionary.Dictionary",

    :param docs0: list of str, corpus 0
    :param docs1: list of str, corpus 1
    :param union: bool, if we take the union or the intersection from the two dictionaries
    :param workers: int, number of workers to parallelize the filtering of the non-tokenized corpus
    :param return_filtered_docs: bool, if the filtered non-tokenized corpus should be returned, if false it will not be calculated
    :return: docs0, dic0, cor0, docs1, dic1, cor1


    """
    assert docs0 is not None
    assert docs1 is not None
    assert len(docs0) == len(docs1)

    # tokenize corpus
    docs0 = tokenize_text(docs0, add_bigrams=True, add_trigrams=False)
    docs1 = tokenize_text(docs1, add_bigrams=True, add_trigrams=False)

    # create dictionary and filter extremes
    docs0, dic0 = tokenize_create_dictionary(docs0)
    docs1, dic1 = tokenize_create_dictionary(docs1)

    # decide on the merge method
    if union:
        transformer = dic0.merge_with(dic1)
    else:
        good_ids0 = []
        good_ids1 = []
        # search for the same words and keep them
        for good_value in set(dic0.values()).intersection(set(dic1.values())):
            good_ids0.append(dic0.token2id[good_value])
            good_ids1.append(dic1.token2id[good_value])
        dic0.filter_tokens(good_ids=good_ids0)

    # set both dictionaries as the same, needed for comparing two topic models
    dic1 = dic0

    assert len(docs0) == len(docs1), ">> ERROR: Corpus length not the same"

    corpus_length = len(docs0)

    # filter the documents so that only words in the vocabulary remain, used for C_v score calculation
    if return_filtered_docs:
        import pathos
        from pathos.multiprocessing import ProcessingPool as Pool

        dictionary = list(dic0.token2id.keys())
        cnt = pathos.helpers.cpu_count() if workers is None else workers
        if cnt % 2:
            cnt -= 1
        half = int(cnt / 2)
        docs_split = list(split(docs0, half))
        docs_split += list(split(docs1, half))
        pool = Pool(ncpus=cnt)
        docs_split = pool.map(lambda x: filter_docs(x, dictionary), docs_split)
        docs0 = list(itertools.chain.from_iterable(docs_split[:half]))
        docs1 = list(itertools.chain.from_iterable(docs_split[half:]))

        assert len(docs0) == len(docs1), f">> ERROR: Corpus length not the same anymore; {len(docs0)} | {len(docs1)}"
        assert len(docs0) == corpus_length, f">> ERROR: Corpus length not the same as before; {corpus_length} -> {len(docs0)}"

    # Bag-of-words representation of the documents.
    cor0 = [dic0.doc2bow(doc) for doc in docs0]
    cor1 = [dic1.doc2bow(doc) for doc in docs1]
    print('>> Number of unique tokens in dic0: %d' % len(dic0))
    print('>> Number of unique tokens in dic1: %d' % len(dic1))
    print('>> Number of documents of cor0: %d' % len(cor0))
    print('>> Number of documents of cor1: %d' % len(cor1))

    return docs0, dic0, cor0, docs1, dic1, cor1


def train_topic_model(docs, dictionary, corpus, num_topics, seed, file_path, topic_model):
    """
    Helper function for calling the respective topic model classes
    :param docs: list of str
    :param dictionary: gensim.corpora.dictionary.Dictionary
    :param corpus: list of (int, int)
    :param num_topics: int, the number of topic the topic model should be trained for
    :param seed: int, the seed for reproducibility
    :param file_path: str, filepath to where the model shall be saved
    :param topic_model: str, the topic model to be used, "classic_lda", "neural_lda"
    """

    if topic_model == "classic_lda":
        train_classic_lda(dictionary, corpus, num_topics, seed, file_path)
    elif topic_model == "neural_lda":
        train_neural_lda(docs, dictionary, num_topics, seed, file_path)
    else:
        raise ValueError("wrong topic model classifier")
    return


def train_classic_lda(dictionary, corpus, num_topics, seed, file_path):
    """
    Trains a classic lda model and saves the model in filepath
    :param dictionary: gensim.corpora.dictionary.Dictionary
    :param corpus: list of (int, int)
    :param num_topics: int, the number of topic the topic model should be trained for
    :param seed: int, the seed for reproducibility
    :param file_path: str, filepath to where the model shall be saved
    """

    # setting seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Train LDA model.
    # Set training parameters.
    chunksize = 1000000
    passes = 30
    iterations = 500
    eval_every = 0  # 1 - Don't evaluate model perplexity, takes too much time.

    # Make a index to word dictionary.
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    # initalizing the model
    model = LdaMulticore(
        corpus=corpus,
        num_topics=num_topics,
        id2word=id2word,
        workers=3,
        chunksize=chunksize,
        passes=passes,
        alpha='symmetric',
        eta='auto',
        eval_every=eval_every,
        iterations=iterations,
        random_state=seed
    )

    # save model
    model.save(f"{file_path}model")

    top_topics = model.top_topics(corpus)
    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    print('>> Average topic coherence: %.4f.' % avg_topic_coherence)

    pprint(top_topics)
    return


def train_neural_lda(documents, dictionary, num_topics, seed, file_path):
    """
    Trains a neural lda model and saves the model in filepath

    The parameter tuning can be manually be set to true to evaluate the best hyperparameters and save the results in filepath

    :param documents: list of str
    :param dictionary: gensim.corpora.dictionary.Dictionary
    :param num_topics: int, the number of topic the topic model should be trained for
    :param seed: int, the seed for reproducibility
    :param file_path: str, filepath to where the model shall be saved
    """

    # Train neural LDA model.
    import torch
    from octis.dataset.dataset import Dataset
    from octis.models.NeuralLDA import NeuralLDA
    from skopt.space.space import Real, Categorical, Integer
    from octis.evaluation_metrics.coherence_metrics import Coherence
    from octis.optimization.optimizer import Optimizer

    # settings seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # tuning boolean if one wants to do hyperparameter tuning, set here manually
    tuning = False

    # initialize dataset
    dataset_object = Dataset(
        corpus=documents,
        vocabulary=list(dictionary.token2id.keys()),
        metadata=dict()
    )

    # Drop_last parameter not available, therefore adjust batch_size to not leave a one-set
    batch_size = 64
    rest = len(documents) % batch_size
    while rest == 1:
        batch_size += 1
        rest = len(documents) % batch_size

    print(f">> batch_size: {batch_size}")

    # initialize neural lda model
    model = NeuralLDA(
        num_topics=num_topics,
        activation='sigmoid',
        dropout=0.1,    # high dropout is bad
        learn_priors=True,
        batch_size=batch_size,
        lr=2e-3,
        momentum=0.99,
        solver='adam',
        num_epochs=200,
        reduce_on_plateau=False,
        prior_mean=0.0,
        prior_variance=None,
        num_layers=1,   # a lot of layers is bad
        num_neurons=2000,   # a lot of neurons is good
        num_samples=100,     # Get the document-topic distribution for a dataset of topics. Includes multiple sampling to reduce variation via the parameter num_samples
        use_partitions=False
    )

    if tuning:
        # define search space, keep in mind the more parameters, the less efficient bayesian optimization. maximum 4 categories
        search_space = {
            'dropout': Categorical({0.0, 0.03, 0.1, 0.2, 0.9}),
            'activation': Categorical({'sigmoid', 'tanh', 'softplus'}),
            'num_layers': Categorical({1, 2, 5}),
            'num_neurons': Categorical({500, 1000, 1500, 2000, 10000}),
        }

        # optimize the c_v coherence score when searching optimal parameters
        coherence = Coherence(texts=dataset_object.get_corpus(), measure='c_v')

        # define optimization runs
        optimization_runs = len(search_space.keys()) * 15

        # define model runs to rule out variation
        model_runs = 6

        # initialize optimizer
        optimizer = Optimizer()

        # run optimizer
        optimization_result = optimizer.optimize(
            model=model,
            dataset=dataset_object,
            metric=coherence,
            search_space=search_space,
            extra_metrics=None,
            number_of_call=optimization_runs,
            model_runs=model_runs,
            random_state=seed,
            save_models=False,
            save_path=file_path
        )

        # save results
        optimization_result.save_to_csv(f"{file_path}results_neuralLDA.csv")
    else:
        # train neural lda model
        model.train_model(dataset_object)

        # save model
        compressed_pickle(f"{file_path}model", model)
    return


def save_data(file_path, dic=None, cor=None, docs=None, ):
    """
    Saves documents, dictionary and corpus to disk
    :param file_path: save folder path
    :param dic: gensim.corpora.dictionary.Dictionary
    :param cor: list of (int, int)
    :param docs: list of str
    """

    # append names for respective data
    os.makedirs(os.path.dirname(f"{file_path}dictionary"), exist_ok=True)
    docs_path = f"{file_path}documents"
    dic_path = f"{file_path}dictionary"
    cor_path = f"{file_path}corpus"

    if dic is not None:
        dic.save(dic_path)
    if cor is not None:
        with open(cor_path, 'w') as file:
            json.dump(cor, file)
    if docs is not None:
        with open(docs_path, 'w') as file:
            json.dump(docs, file)


def load_data(docs_path=None, dic_path=None, cor_path=None):
    """
    Loads documents, dictionary and corpus from disk
    cor is of type "list of (int, int)",
    doc is of type "list of str",
    dic is of type "gensim.corpora.dictionary.Dictionary",

    :param docs_path: str, path to documents
    :param dic_path: str, path to dictionary
    :param cor_path: str, path to corpus
    :return: documents, dictionary, corpus
    """

    # initialize parameters to avoid error when returning
    documents = dictionary = corpus = None

    # load dictionary
    if dic_path is not None:
        dictionary = SaveLoad.load(dic_path)

    # load corpus
    if cor_path is not None:
        with open(cor_path, 'r') as file:
            corpus = json.load(file)
        corpus = [x for x in corpus if x]

    # load documents
    if docs_path is not None:
        with open(docs_path, 'r') as file:
            documents = json.load(file)
        documents = [x for x in documents if x]

    return documents, dictionary, corpus


def main():
    """
    Command:
        python train_lda.py [data_path] [topic_model_type] [first_corpus] [second_corpus] [focus] [sampling_method] [number_of_topics] [merge_technique] [corpus_size] [variance_index]

        Corpus Options (always use gpt2_nt, gpt2 or wiki first, in that order (convention)): str
            gpt2_nt, gpt2, trafo_xl_nt, trafo_xl, wiki_nt, arxiv
        Topic Model Type: str
            classic_lda, neural_lda
        Focus options: str
            first: First corpus is used for lda model creation
            second: Second corpus is used for lda model creation
        Sampling method: str
            multinomial, typ_p, top_p
        Topic options: int
            2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100, ...
        Merge technique: str
            union: Unionize dictionaries
            intersection: Intersect dictionaries
        Corpus size: int
            > 0 and < 100'000
        Variance index: int
            Model index when calculating the variance (changes the seed)

    Examples:
        python train_lda.py /cluster/work/cotterell/knobelf/data/ classic_lda gpt2_nt wiki_nt first multinomial 5 union 10000
        python train_lda.py /cluster/work/cotterell/knobelf/data/ classic_lda gpt2_nt gpt2 first typ_p 10 intersection 1 100000
        python train_lda.py /cluster/work/cotterell/knobelf/data/ classic_lda gpt2_nt gpt2 second typ_p 10 intersection 2 10000
        python train_lda.py /cluster/work/cotterell/knobelf/data/ classic_lda gpt2 arxiv first typ_p 10 intersection 3 1000
        python train_lda.py /cluster/work/cotterell/knobelf/data/ classic_lda wiki_nt arxiv first typ_p 10 intersection 4 10000

    """
    assert len(sys.argv) >= 10, ">> ERROR: Incorrect number of input arguments"

    # assign input arguments and cast to appropriate type
    data_path = sys.argv[1]
    topic_model = sys.argv[2]
    first = sys.argv[3]
    second = sys.argv[4]
    focus = sys.argv[5]
    sampling = sys.argv[6]
    num_topics = int(sys.argv[7])
    combi = sys.argv[8]
    samples = int(sys.argv[9])

    assert 100000 >= samples > 0, ">> ERROR: invalid sample size"

    if data_path[-1] != "/":
        data_path += "/"

    assert first in ["gpt2_nt", "gpt2", "trafo_xl_nt", "trafo_xl", "wiki_nt", "arxiv"], ">> ERROR: undefined first input"
    assert second in ["gpt2_nt", "gpt2", "trafo_xl_nt", "trafo_xl", "wiki_nt", "arxiv"], ">> ERROR: undefined second input"
    assert sampling in ["multinomial", "typ_p", "top_p"], ">> ERROR: undefined sampling input"

    if topic_model not in ["classic_lda", "neural_lda"]:
        raise ValueError(f">> ERROR: undefinded topic_model")

    folder_name = f"{first}-{second}"
    if sampling != "multinomial":
        folder_name += "-" + sampling

    if first == second:
        first += "_1"
        second += "_2"

    assert num_topics >= 1, ">> ERROR: undefined num_topics input"

    if combi == "intersection":
        union = False
    elif combi == "union":
        union = True
    else:
        raise ValueError(">> ERROR: undefined combi input")

    # modify seed if the model is used for variance testing
    seed = 42
    index = ""
    if len(sys.argv) == 11:
        var_idx = int(sys.argv[10])
        assert int(sys.argv[10]) > 0, ">> ERROR: undefinded variation index"
        index = f"{var_idx}/"
        seed = 42 + 7 * var_idx
    print(f">> SEED for topic model generation: {seed}")

    # define and modify file paths
    file_path_first = f"{data_path}{samples}/{folder_name}/{first}/{combi}/"
    file_path_second = f"{data_path}{samples}/{folder_name}/{second}/{combi}/"

    lda_file_path_first = f"{file_path_first}{topic_model}/{num_topics}/{index}"
    lda_file_path_second = f"{file_path_second}{topic_model}/{num_topics}/{index}"

    if focus == "first":
        file_path = file_path_first
        lda_file_path = lda_file_path_first
    elif focus == "second":
        file_path = file_path_second
        lda_file_path = lda_file_path_second
    elif focus == "tokenization":
        file_path = file_path_first
        lda_file_path = ""
    else:
        raise ValueError(">> ERROR: undefined focus input")

    docs_path = f"{file_path}documents"
    dic_path = f"{file_path}dictionary"
    cor_path = f"{file_path}corpus"

    # set seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # initiate logging for debugging and controlling (check if the model has converged)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

    # tokenize corpus if the tokenized version does not exist
    if not os.path.exists(dic_path):

        # load the corpora
        docs1 = load_dataset(data_path, first, sampling, samples)
        docs2 = load_dataset(data_path, second, sampling, samples)
        assert docs1 is not None
        assert docs2 is not None
        assert len(docs1) == samples
        assert len(docs2) == samples

        # reset seed for reproducibility
        random.seed(42)
        np.random.seed(42)

        # tokenize the corpora
        docs1, dic1, cor1, docs2, dic2, cor2 = tokenize_bow_dual(docs1, docs2, union, workers=7, return_filtered_docs=True)

        # save tokenized corpora
        save_data(file_path_first, dic=dic1, docs=docs1, cor=cor1)
        save_data(file_path_second, dic=dic2, docs=docs2, cor=cor2)

    # load the tokenized corpus
    documents, dictionary, corpus = load_data(docs_path=docs_path, dic_path=dic_path, cor_path=cor_path)

    if num_topics == 1:
        # there is no topic model with 1 topic...
        return

    # create folder if it does not exist
    os.makedirs(os.path.dirname(lda_file_path), exist_ok=True)

    # train topic model
    train_topic_model(documents, dictionary, corpus, num_topics, seed, lda_file_path, topic_model)
    return


if __name__ == "__main__":
    main()
