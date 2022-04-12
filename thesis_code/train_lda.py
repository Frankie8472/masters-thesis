import itertools
import logging
import os
import pickle
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


def load_wikitext(path, samples=100000):
    heading_pattern = '( \n [=\s].*[=\s] \n)'
    train_data = Path(f'{path}datasets/data_wikitext-103-raw/wiki.train.raw').read_text(encoding='utf-8')
    train_split = re.split(heading_pattern, train_data)
    train_headings = [x[7:-7] for x in train_split[1::2]]
    train_articles = [x for x in train_split[2::2]]
    return random.choices(train_articles, k=samples)


def load_arxiv(path, samples=100000):
    def get_metadata():
        with open(f'{path}datasets/data_arxiv-metadata-oai-snapshot.json', 'r') as f:
            for line in f:
                yield line

    metadata = get_metadata()
    corpus = []
    for paper in metadata:
        corpus.append(json.loads(paper)['abstract'])
    return random.choices(corpus, k=samples)


def load_json_choices(filename, samples=100000):
    with open(filename, 'r') as file:
        train_articles = json.load(file)
    return random.choices(train_articles, k=samples)


def load_json(filename, samples=100000):
    with open(filename, 'r') as file:
        train_articles = json.load(file)
    return train_articles[:min(samples, len(train_articles)-1)]


def load_dataset(data_path, set_name, sampling_method, samples=100000):
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
    return docs


def tokenize_text(docs, add_bigrams=True, add_trigrams=True):
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
    # Remove rare and common tokens.
    # Create a dictionary representation of the documents.
    dic = Dictionary(docs)

    # Filter out words that occur less than 20 documents, or more than 50% of the documents.
    dic.filter_extremes(no_below=20, no_above=0.5)

    return docs, dic


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def filter_docs(docs, dic):
    for i, _ in enumerate(docs):
        docs[i] = [word for word in docs[i] if word in dic]
    return docs


def tokenize_bow_single(docs, workers=None, return_filtered_docs=False):
    docs = tokenize_text(docs, add_bigrams=True, add_trigrams=True)
    docs, dic = tokenize_create_dictionary(docs)

    if return_filtered_docs:
        import pathos
        from pathos.multiprocessing import ProcessingPool as Pool

        dictionary = list(dic.token2id.keys())
        cnt = pathos.helpers.cpu_count() if workers is None else workers
        docs_split = list(split(docs, cnt))
        pool = Pool(ncpus=cnt)
        docs_split = pool.map(lambda x: filter_docs(x, dictionary), docs_split)
        docs = list(itertools.chain.from_iterable(docs_split))

    # Bag-of-words representation of the documents.
    corpus = [dic.doc2bow(doc) for doc in docs]
    print('>> Number of unique tokens: %d' % len(dic))
    print('>> Number of documents: %d' % len(corpus))

    return docs, dic, corpus


def tokenize_bow_dual(docs0, docs1, union=False, workers=None, return_filtered_docs=False):
    assert docs0 is not None
    assert docs1 is not None

    docs0 = tokenize_text(docs0, add_bigrams=True, add_trigrams=True)
    docs1 = tokenize_text(docs1, add_bigrams=True, add_trigrams=True)
    docs0, dic0 = tokenize_create_dictionary(docs0)
    docs1, dic1 = tokenize_create_dictionary(docs1)

    if union:
        transformer = dic0.merge_with(dic1)
    else:
        good_ids0 = []
        good_ids1 = []
        for good_value in set(dic0.values()).intersection(set(dic1.values())):
            good_ids0.append(dic0.token2id[good_value])
            good_ids1.append(dic1.token2id[good_value])
        dic0.filter_tokens(good_ids=good_ids0)

    dic1 = dic0

    assert len(docs0) == len(docs1), ">> ERROR: Corpus length not the same"

    corpus_length = len(docs0)

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


def train_topic_model(docs, dictionary, corpus, num_topics, seed, file_path, data_path, topic_model):
    if topic_model == "classic_lda":
        train_classic_lda(dictionary, corpus, num_topics, seed, file_path)
    elif topic_model == "neural_lda":
        train_neural_lda(docs, dictionary, num_topics, seed, file_path, data_path)
    else:
        raise ValueError("wrong topic model classifier")
    return


def train_classic_lda(dictionary, corpus, num_topics, seed, file_path):
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

    top_topics = model.top_topics(corpus)
    model.save(f"{file_path}ldamodel_{num_topics}")

    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    print('>> Average topic coherence: %.4f.' % avg_topic_coherence)

    pprint(top_topics)
    return


def train_neural_lda(documents, dictionary, num_topics, seed, file_path, data_path):
    # Train neural LDA model.
    import torch
    from octis.dataset.dataset import Dataset
    from octis.models.NeuralLDA import NeuralLDA
    from skopt.space.space import Real, Categorical, Integer
    from octis.evaluation_metrics.coherence_metrics import Coherence
    from octis.optimization.optimizer import Optimizer

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tuning = False

    dataset_object = Dataset(
        corpus=documents,
        vocabulary=list(dictionary.token2id.keys()),
        metadata=dict()
    )

    model = NeuralLDA(
        num_topics=num_topics,
        activation='softplus',
        dropout=0.2,
        learn_priors=True,
        batch_size=64,
        lr=2e-3,
        momentum=0.99,
        solver='adam',
        num_epochs=100,
        reduce_on_plateau=False,
        prior_mean=0.0,
        prior_variance=None,
        num_layers=2,
        num_neurons=100,
        num_samples=10,
        use_partitions=False
    )

    if tuning:
        search_space = {
            'dropout': Real(0.0, 0.95),
            'activation': Categorical({'softplus', 'relu', 'sigmoid', 'tanh', 'rrelu', 'elu'}),
            'num_layers': Integer(1, 20),
            'num_neurons': Categorical({50, 100, 200, 500, 1000}),
            'num_samples': Categorical({10, 20, 50, 100})
        }

        coherence = Coherence(texts=dataset_object.get_corpus(), measure='c_v')

        optimization_runs = len(search_space.keys()) * 15
        model_runs = 10

        optimizer = Optimizer()

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

        optimization_result.save_to_csv(f"{file_path}results_neuralLDA.csv")
    else:
        model.train_model(dataset_object)
        with open(f"{file_path}model.pickle", "wb") as file:
            pickle.dump(model, file)
    return


def save_data(file_path, dic=None, cor=None, docs=None, ):
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
    documents = dictionary = corpus = None
    if dic_path is not None:
        dictionary = SaveLoad.load(dic_path)
    if cor_path is not None:
        with open(cor_path, 'r') as file:
            corpus = json.load(file)
    if docs_path is not None:
        with open(docs_path, 'r') as file:
            documents = json.load(file)

    return documents, dictionary, corpus


def main():
    """
    Command:
        python train_lda.py [data_path] [topic_model_type] [first_corpus] [second_corpus] [focus] [sampling_method] [number_of_topics] [merge_technique] [corpus_size] [variance_index]

        Corpus Options (always use gpt2_nt, gpt2 or wiki first, in that order (convention)):
            gpt2_nt, gpt2, trafo_xl_nt, trafo_xl, wiki_nt, arxiv
        Topic Model Type:
            classic_lda, neural_lda
        Focus options:
            first: First corpus is used for lda model creation
            second: Second corpus is used for lda model creation
        Sampling method:
            multinomial, typ_p, top_p
        Topic options:
            2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100, ...
        Merge technique:
            union: Unionize dictionaries
            intersection: Intersect dictionaries
        Corpus size:
            Int > 0 and < 100'000
        Variance index:
            Model index when calculating the variance (changes the seed)

    Examples:
        python train_lda.py /cluster/work/cotterell/knobelf/data/ classic_lda gpt2_nt wiki_nt first multinomial 5 union 10000
        python train_lda.py /cluster/work/cotterell/knobelf/data/ classic_lda gpt2_nt gpt2 first typ_p 10 intersection 1 100000
        python train_lda.py /cluster/work/cotterell/knobelf/data/ classic_lda gpt2_nt gpt2 second typ_p 10 intersection 2 10000
        python train_lda.py /cluster/work/cotterell/knobelf/data/ classic_lda gpt2 arxiv first typ_p 10 intersection 3 1000
        python train_lda.py /cluster/work/cotterell/knobelf/data/ classic_lda wiki_nt arxiv first typ_p 10 intersection 4 10000

    """
    assert len(sys.argv) >= 10, ">> ERROR: Incorrect number of input arguments"

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
        raise AssertionError(f">> ERROR: undefinded topic_model")

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
        raise AssertionError(">> ERROR: undefined combi input")

    seed = 42
    index = ""
    if len(sys.argv) == 11:
        var_idx = int(sys.argv[10])
        assert int(sys.argv[10]) > 0, ">> ERROR: undefinded variation index"
        index = f"{var_idx}/"
        seed = 42 + 7 * var_idx
    print(f">> SEED for topic model generation: {seed}")

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
        raise AssertionError(">> ERROR: undefined focus input")

    docs_path = f"{file_path}documents"
    dic_path = f"{file_path}dictionary"
    cor_path = f"{file_path}corpus"

    random.seed(42)
    np.random.seed(42)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

    if not os.path.exists(dic_path):
        docs1 = load_dataset(data_path, first, sampling, samples)
        docs2 = load_dataset(data_path, second, sampling, samples)
        assert docs1 is not None
        assert docs2 is not None

        random.seed(42)
        np.random.seed(42)

        docs1, dic1, cor1, docs2, dic2, cor2 = tokenize_bow_dual(docs1, docs2, union, workers=7, return_filtered_docs=True)

        save_data(file_path_first, dic=dic1, docs=docs1, cor=cor1)
        save_data(file_path_second, dic=dic2, docs=docs2, cor=cor2)

    documents, dictionary, corpus = load_data(docs_path=docs_path, dic_path=dic_path, cor_path=cor_path)

    if num_topics == 1:
        return
    os.makedirs(os.path.dirname(lda_file_path), exist_ok=True)
    train_topic_model(documents, dictionary, corpus, num_topics, seed, lda_file_path, data_path, topic_model)
    return


if __name__ == "__main__":
    main()
