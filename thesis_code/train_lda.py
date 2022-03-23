import gc
import logging
import os
import random
import numpy as np
import sys
from pathlib import Path
import re
import json
from pprint import pprint
from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore
from nltk import WordNetLemmatizer, RegexpTokenizer
from transformers import set_seed


def load_wikitext(path, samples=100000):
    heading_pattern = '( \n [=\s].*[=\s] \n)'
    train_data = Path(f'{path}data_wikitext-103-raw/wiki.train.raw').read_text(encoding='utf-8')
    train_split = re.split(heading_pattern, train_data)
    train_headings = [x[7:-7] for x in train_split[1::2]]
    train_articles = [x for x in train_split[2::2]]
    return random.choices(train_articles, k=samples)


def load_arxiv(path, samples=100000):
    def get_metadata():
        with open(f'{path}data_arxiv-metadata-oai-snapshot.json', 'r') as f:
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


def load_json_choices(filename, samples=100000):
    with open(filename, 'r') as file:
        train_articles = json.load(file)
    return random.choices(train_articles, k=samples)


def load_json(filename):
    with open(filename, 'r') as file:
        train_articles = json.load(file)
    return train_articles


def load_dataset(data_path, set_name, sampling_method):
    if sampling_method == "normal":
        sampling = ""
    else:
        sampling = "-" + sampling_method

    dataset = "dataset1"
    if set_name[-2:] == "_2":
        dataset = "dataset2"

    docs = None
    if set_name == "arxiv":
        if set_name[-2:] == "_2":
            set_seed(1337)
        docs = load_arxiv(data_path)
    elif set_name == "wiki_nt":
        if set_name[-2:] == "_2":
            set_seed(1337)
        docs = load_wikitext(data_path)
    elif set_name == "gpt2_nt":
        docs = load_json(f"{data_path}{dataset}-gpt2-wiki_nt{sampling}.json")
    elif set_name == "gpt2":
        docs = load_json(f"{data_path}{dataset}-gpt2{sampling}.json")

    return docs


def tokenize_preprocessing(docs, add_trigrams=True):
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
    bigram = Phrases(docs, min_count=5, threshold=10.0)

    trigram = Phrases(bigram[docs], min_count=5, threshold=10.0) if add_trigrams else None

    for idx in range(len(docs)):
        doc = docs[idx]
        bigrams = list()
        trigrams = list()
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

        for token in bigrams:
            docs[idx].append(token)

        if add_trigrams:
            for token in trigrams:
                docs[idx].append(token)

    # Remove rare and common tokens.
    # Create a dictionary representation of the documents.
    dictionary = Dictionary(docs)

    # Filter out words that occur less than 20 documents, or more than 50% of the documents.
    dictionary.filter_extremes(no_below=20, no_above=0.5)

    return docs, dictionary


def tokenize_bow_single(docs, dictionary):
    # Bag-of-words representation of the documents.
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))

    return dictionary, corpus


def tokenize_bow_dual(docs0, docs1, union=False):  # False is intersection of dictionaries
    docs_new_0, dic0 = tokenize_preprocessing(docs0)
    docs_new_1, dic1 = tokenize_preprocessing(docs1)

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

    # Bag-of-words representation of the documents.
    cor0 = [dic0.doc2bow(doc) for doc in docs0]
    cor1 = [dic1.doc2bow(doc) for doc in docs1]
    print('Number of unique tokens in dic0: %d' % len(dic0))
    print('Number of unique tokens in dic1: %d' % len(dic1))
    print('Number of documents of cor0: %d' % len(cor0))
    print('Number of documents of cor1: %d' % len(cor1))

    return dic0, cor0, dic1, cor1


def train_lda(dictionary, corpus, topics):
    # Train LDA model.

    # Set training parameters.
    num_topics = topics
    chunksize = 1000000
    passes = 30
    iterations = 500
    eval_every = 1  # Don't evaluate model perplexity, takes too much time.

    # Make a index to word dictionary.
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    model = LdaMulticore(
        corpus=corpus,
        num_topics=num_topics,
        id2word=id2word,
        workers=48,
        chunksize=chunksize,
        passes=passes,
        alpha='symmetric',
        eta='auto',
        eval_every=eval_every,
        iterations=iterations,
    )

    top_topics = model.top_topics(corpus)

    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    print('Average topic coherence: %.4f.' % avg_topic_coherence)

    pprint(top_topics)

    return model


def main():
    """
    Command:
        python train_lda.py [data_path] [first corpus] [second corpus] [focus] [sampling method] [number of topics] [merge technique] [variance index]

        Corpus Options (always use gpt2 or wiki first, in that order (convention)):
            gpt2, gpt2_nt, wiki_nt, arxiv
        Focus options:
            first: First corpus is used for lda model creation
            second: Second corpus is used for lda model creation
        Sampling method:
            normal, typ_p, top_p
        Topic options:
            2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100, ...
        Merge technique:
            union: Unionize dictionaries
            intersection: Intersect dictionaries
        Variance index:
            Model index when calculating the variance (changes the seed)

    Examples:
        python train_lda.py /cluster/work/cotterell/knobelf/data/ gpt2_nt wiki_nt first normal 5 union
        python train_lda.py /cluster/work/cotterell/knobelf/data/ gpt2_nt gpt2 first typ_p 10 intersection 1
        python train_lda.py /cluster/work/cotterell/knobelf/data/ gpt2_nt gpt2_nt second typ_p 10 intersection 2
        python train_lda.py /cluster/work/cotterell/knobelf/data/ gpt2_nt arxiv first typ_p 10 intersection 3
        python train_lda.py /cluster/work/cotterell/knobelf/data/ wiki_nt arxiv first typ_p 10 intersection 4

    """
    if len(sys.argv) < 8:
        print("ERROR: Incorrect number of input arguments")
        return

    data_path = sys.argv[1]
    first = sys.argv[2]
    second = sys.argv[3]
    focus = sys.argv[4]
    sampling = sys.argv[5]
    num_topics = int(sys.argv[6])
    combi = sys.argv[7]

    if data_path[-1] != "/":
        data_path += "/"

    if first not in ["gpt2", "gpt2_nt", "wiki_nt", "arxiv"]:
        print("ERROR: undefined first input")
        return

    if second not in ["gpt2", "gpt2_nt", "wiki_nt", "arxiv"]:
        print("ERROR: undefined second input")
        return

    folder_name = f"lda-{first}-{second}"

    if first == second:
        first += "_1"
        second += "_2"

    if sampling not in ["normal", "typ_p", "top_p"]:
        print("ERROR: undefined sampling input")
        return

    if num_topics <= 1:
        print("ERROR: undefined num_topics input")
        return

    union = None
    if combi == "intersection":
        union = False
    elif combi == "union":
        union = True
    else:
        print("ERROR: undefined combi input")
        return

    set_seed(42)
    docs1 = load_dataset(data_path, first, sampling)
    docs2 = load_dataset(data_path, second, sampling)
    set_seed(42)

    dic1, cor1, dic2, cor2 = tokenize_bow_dual(docs1, docs2, union)

    model_name = None
    if focus == "first":
        model_name = first
        dictionary = dic1
        corpus = cor1
    elif focus == "second":
        model_name = second
        dictionary = dic2
        corpus = cor2
    else:
        print("ERROR: undefined focus input")
        return

    index = ""
    if len(sys.argv) == 9:
        index = sys.argv[8]
        set_seed(42 + 7**int(index))
        index = "/" + index

    model = train_lda(dictionary, corpus, num_topics)

    file_path = f"{data_path}{folder_name}/{model_name}{index}/{combi}/{num_topics}/"

    os.makedirs(os.path.dirname(f"{file_path}corpus_{num_topics}"), exist_ok=True)
    with open(f"{file_path}corpus_{num_topics}", "w") as file:
        json.dump(corpus, file, indent=2)
    dictionary.save(f"{file_path}dictionary_{num_topics}")
    model.save(f"{file_path}ldamodel_{num_topics}")


def main2():
    if len(sys.argv) < 4:
        print("ERROR: Incorrect number of input arguments")
        return

    data_folder = "/cluster/work/cotterell/knobelf/data/"

    union = bool(int(sys.argv[1]))
    dataset = int(sys.argv[2])
    topics = int(sys.argv[3])
    pathname = None

    combi = "union" if union else "intersection"

    extra_folder = ""
    iteration = None
    if len(sys.argv) >= 5:
        iteration = int(sys.argv[4])
        extra_folder = "/" + sys.argv[4]

    seed = seed2 = 42
    if iteration is not None:
        seed2 = 42 + 7**iteration
        if iteration < 6:
            seed = seed2

    print(f"SEED: {seed}")
    print(f"SEED2: {seed2}")

    if not len(sys.argv) >= 6 and dataset == 0:
        print("ERROR: no path argument and dataset == 0")
    elif len(sys.argv) >= 6:
        pathname = sys.argv[5]

    set_seed(seed)
    samples = 100000

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

    if dataset == 0:
        data_train = load_json(pathname)
        dictionary, corpus = tokenize(data_train)
        model = train_lda(dictionary, corpus, topics)
        os.makedirs(os.path.dirname(f"{data_folder}test/corpus2_{topics}"), exist_ok=True)
        with open(f"{data_folder}test/corpus2_{topics}", "w") as file:
            json.dump(corpus, file, indent=2)
        dictionary.save(f"{data_folder}test/dictionary2_{topics}")
        model.save(f"{data_folder}test/ldamodel2_{topics}")

    # Create LDA Model for wiki_nt and gpt2_nt
    elif dataset == 1 or dataset == 2:
        data_train_1 = load_wikitext(samples)
        data_train_2 = load_json(f"{data_folder}dataset1-gpt2-wiki_nt.json")
        dictionary_1, corpus_1, dictionary_2, corpus_2 = tokenize_special(data_train_1, data_train_2, union)

        set_seed(seed2)
        if dataset == 1:
            model_1 = train_lda(dictionary_1, corpus_1, topics)

            os.makedirs(os.path.dirname(f"{data_folder}lda-gpt2_nt-wiki_nt/wiki_nt{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                        exist_ok=True)
            with open(f"{data_folder}lda-gpt2_nt-wiki_nt/wiki_nt{extra_folder}/{combi}/{topics}/corpus_{topics}", "w") as file:
                json.dump(corpus_1, file, indent=2)
            dictionary_1.save(f"{data_folder}lda-gpt2_nt-wiki_nt/wiki_nt{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_1.save(f"{data_folder}lda-gpt2_nt-wiki_nt/wiki_nt{extra_folder}/{combi}/{topics}/ldamodel_{topics}")
        else:
            model_2 = train_lda(dictionary_2, corpus_2, topics)

            os.makedirs(os.path.dirname(f"{data_folder}lda-gpt2_nt-wiki_nt/gpt2_nt{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                        exist_ok=True)
            with open(f"{data_folder}lda-gpt2_nt-wiki_nt/gpt2_nt{extra_folder}/{combi}/{topics}/corpus_{topics}", "w") as file:
                json.dump(corpus_2, file, indent=2)
            dictionary_2.save(f"{data_folder}lda-gpt2_nt-wiki_nt/gpt2_nt{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_2.save(f"{data_folder}lda-gpt2_nt-wiki_nt/gpt2_nt{extra_folder}/{combi}/{topics}/ldamodel_{topics}")

    # Create LDA Model for gpt2_nt and gpt2_nt
    elif dataset == 3 or dataset == 4:
        data_train_1 = load_json(f"{data_folder}dataset1-gpt2-wiki_nt.json")
        data_train_2 = load_json(f"{data_folder}dataset2-gpt2-wiki_nt.json")
        dictionary_1, corpus_1, dictionary_2, corpus_2 = tokenize_special(data_train_1, data_train_2, union)

        set_seed(seed2)
        if dataset == 3:
            model_1 = train_lda(dictionary_1, corpus_1, topics)

            os.makedirs(os.path.dirname(f"{data_folder}lda-gpt2_nt-gpt2_nt/gpt2_nt_1{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                        exist_ok=True)
            with open(f"{data_folder}lda-gpt2_nt-gpt2_nt/gpt2_nt_1{extra_folder}/{combi}/{topics}/corpus_{topics}", "w") as file:
                json.dump(corpus_1, file, indent=2)
            dictionary_1.save(f"{data_folder}lda-gpt2_nt-gpt2_nt/gpt2_nt_1{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_1.save(f"{data_folder}lda-gpt2_nt-gpt2_nt/gpt2_nt_1{extra_folder}/{combi}/{topics}/ldamodel_{topics}")
        else:
            model_2 = train_lda(dictionary_2, corpus_2, topics)

            os.makedirs(os.path.dirname(f"{data_folder}lda-gpt2_nt-gpt2_nt/gpt2_nt_2{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                        exist_ok=True)
            with open(f"{data_folder}lda-gpt2_nt-gpt2_nt/gpt2_nt_2{extra_folder}/{combi}/{topics}/corpus_{topics}", "w") as file:
                json.dump(corpus_2, file, indent=2)
            dictionary_2.save(f"{data_folder}lda-gpt2_nt-gpt2_nt/gpt2_nt_2{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_2.save(f"{data_folder}lda-gpt2_nt-gpt2_nt/gpt2_nt_2{extra_folder}/{combi}/{topics}/ldamodel_{topics}")

    # Create LDA Model for gpt2 and gpt2
    elif dataset == 5 or dataset == 6:
        data_train_1 = load_json(f"{data_folder}dataset1-gpt2.json")
        data_train_2 = load_json(f"{data_folder}dataset2-gpt2.json")
        dictionary_1, corpus_1, dictionary_2, corpus_2 = tokenize_special(data_train_1, data_train_2, union)

        set_seed(seed2)
        if dataset == 5:
            model_1 = train_lda(dictionary_1, corpus_1, topics)

            os.makedirs(os.path.dirname(f"{data_folder}lda-gpt2-gpt2/gpt2_1{extra_folder}/{combi}/{topics}/corpus_{topics}"), exist_ok=True)
            with open(f"{data_folder}lda-gpt2-gpt2/gpt2_1{extra_folder}/{combi}/{topics}/corpus_{topics}", "w") as file:
                json.dump(corpus_1, file, indent=2)
            dictionary_1.save(f"{data_folder}lda-gpt2-gpt2/gpt2_1{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_1.save(f"{data_folder}lda-gpt2-gpt2/gpt2_1{extra_folder}/{combi}/{topics}/ldamodel_{topics}")
        else:
            model_2 = train_lda(dictionary_2, corpus_2, topics)

            os.makedirs(os.path.dirname(f"{data_folder}lda-gpt2-gpt2/gpt2_2{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                        exist_ok=True)
            with open(f"{data_folder}lda-gpt2-gpt2/gpt2_2{extra_folder}/{combi}/{topics}/corpus_{topics}", "w") as file:
                json.dump(corpus_2, file, indent=2)
            dictionary_2.save(f"{data_folder}lda-gpt2-gpt2/gpt2_2{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_2.save(f"{data_folder}lda-gpt2-gpt2/gpt2_2{extra_folder}/{combi}/{topics}/ldamodel_{topics}")

    # Create LDA Model for wiki_nt and arxiv
    elif dataset == 7 or dataset == 8:
        data_train_1 = load_wikitext(samples)
        data_train_2 = load_arxiv(samples)
        dictionary_1, corpus_1, dictionary_2, corpus_2 = tokenize_special(data_train_1, data_train_2, union)

        set_seed(seed2)
        if dataset == 7:
            model_1 = train_lda(dictionary_1, corpus_1, topics)

            os.makedirs(os.path.dirname(f"{data_folder}lda-wiki_nt-arxiv/wiki_nt{extra_folder}/{combi}/{topics}/corpus_{topics}"), exist_ok=True)
            with open(f"{data_folder}lda-wiki_nt-arxiv/wiki_nt{extra_folder}/{combi}/{topics}/corpus_{topics}", "w") as file:
                json.dump(corpus_1, file, indent=2)
            dictionary_1.save(f"{data_folder}lda-wiki_nt-arxiv/wiki_nt{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_1.save(f"{data_folder}lda-wiki_nt-arxiv/wiki_nt{extra_folder}/{combi}/{topics}/ldamodel_{topics}")
        else:
            model_2 = train_lda(dictionary_2, corpus_2, topics)

            os.makedirs(os.path.dirname(f"{data_folder}lda-wiki_nt-arxiv/arxiv{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                        exist_ok=True)
            with open(f"{data_folder}lda-wiki_nt-arxiv/arxiv{extra_folder}/{combi}/{topics}/corpus_{topics}", "w") as file:
                json.dump(corpus_2, file, indent=2)
            dictionary_2.save(f"{data_folder}lda-wiki_nt-arxiv/arxiv{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_2.save(f"{data_folder}lda-wiki_nt-arxiv/arxiv{extra_folder}/{combi}/{topics}/ldamodel_{topics}")

    # Create LDA Model for arxiv and gpt2_nt
    elif dataset == 9 or dataset == 10:
        data_train_1 = load_json(f"{data_folder}dataset1-gpt2-wiki_nt.json")
        data_train_2 = load_arxiv(samples)
        dictionary_1, corpus_1, dictionary_2, corpus_2 = tokenize_special(data_train_1, data_train_2, union)

        set_seed(seed2)
        if dataset == 9:
            model_1 = train_lda(dictionary_1, corpus_1, topics)

            os.makedirs(os.path.dirname(f"{data_folder}lda-gpt2_nt-arxiv/gpt2_nt{extra_folder}/{combi}/{topics}/corpus_{topics}"), exist_ok=True)
            with open(f"{data_folder}lda-gpt2_nt-arxiv/gpt2_nt{extra_folder}/{combi}/{topics}/corpus_{topics}", "w") as file:
                json.dump(corpus_1, file, indent=2)
            dictionary_1.save(f"{data_folder}lda-gpt2_nt-arxiv/gpt2_nt{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_1.save(f"{data_folder}lda-gpt2_nt-arxiv/gpt2_nt{extra_folder}/{combi}/{topics}/ldamodel_{topics}")
        else:
            model_2 = train_lda(dictionary_2, corpus_2, topics)

            os.makedirs(os.path.dirname(f"{data_folder}lda-gpt2_nt-arxiv/arxiv{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                        exist_ok=True)
            with open(f"{data_folder}lda-gpt2_nt-arxiv/arxiv{extra_folder}/{combi}/{topics}/corpus_{topics}", "w") as file:
                json.dump(corpus_2, file, indent=2)
            dictionary_2.save(f"{data_folder}lda-gpt2_nt-arxiv/arxiv{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_2.save(f"{data_folder}lda-gpt2_nt-arxiv/arxiv{extra_folder}/{combi}/{topics}/ldamodel_{topics}")

    # Create LDA Model for arxiv and gpt2_nt-top_p
    elif dataset == 11 or dataset == 12:
        data_train_1 = load_json(f"{data_folder}dataset1-gpt2-wiki_nt-top_p.json")
        data_train_2 = load_arxiv(samples)
        dictionary_1, corpus_1, dictionary_2, corpus_2 = tokenize_special(data_train_1, data_train_2, union)

        set_seed(seed2)
        if dataset == 11:
            model_1 = train_lda(dictionary_1, corpus_1, topics)

            os.makedirs(os.path.dirname(
                f"{data_folder}lda-gpt2_nt-arxiv-top_p/gpt2_nt{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                        exist_ok=True)
            with open(f"{data_folder}lda-gpt2_nt-arxiv-top_p/gpt2_nt{extra_folder}/{combi}/{topics}/corpus_{topics}",
                      "w") as file:
                json.dump(corpus_1, file, indent=2)
            dictionary_1.save(
                f"{data_folder}lda-gpt2_nt-arxiv-top_p/gpt2_nt{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_1.save(f"{data_folder}lda-gpt2_nt-arxiv-top_p/gpt2_nt{extra_folder}/{combi}/{topics}/ldamodel_{topics}")
        else:
            model_2 = train_lda(dictionary_2, corpus_2, topics)

            os.makedirs(
                os.path.dirname(f"{data_folder}lda-gpt2_nt-arxiv-top_p/arxiv{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                exist_ok=True)
            with open(f"{data_folder}lda-gpt2_nt-arxiv-top_p/arxiv{extra_folder}/{combi}/{topics}/corpus_{topics}",
                      "w") as file:
                json.dump(corpus_2, file, indent=2)
            dictionary_2.save(
                f"{data_folder}lda-gpt2_nt-arxiv-top_p/arxiv{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_2.save(f"{data_folder}lda-gpt2_nt-arxiv-top_p/arxiv{extra_folder}/{combi}/{topics}/ldamodel_{topics}")

    # Create LDA Model for wikitext and gpt2_nt-top_p
    elif dataset == 13 or dataset == 14:
        data_train_1 = load_json(f"{data_folder}dataset1-gpt2-wiki_nt-top_p.json")
        data_train_2 = load_wikitext(samples)
        dictionary_1, corpus_1, dictionary_2, corpus_2 = tokenize_special(data_train_1, data_train_2, union)

        set_seed(seed2)
        if dataset == 13:
            model_1 = train_lda(dictionary_1, corpus_1, topics)

            os.makedirs(os.path.dirname(
                f"{data_folder}lda-gpt2_nt-wiki_nt-top_p/gpt2_nt{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                        exist_ok=True)
            with open(f"{data_folder}lda-gpt2_nt-wiki_nt-top_p/gpt2_nt{extra_folder}/{combi}/{topics}/corpus_{topics}",
                      "w") as file:
                json.dump(corpus_1, file, indent=2)
            dictionary_1.save(
                f"{data_folder}lda-gpt2_nt-wiki_nt-top_p/gpt2_nt{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_1.save(f"{data_folder}lda-gpt2_nt-wiki_nt-top_p/gpt2_nt{extra_folder}/{combi}/{topics}/ldamodel_{topics}")
        else:
            model_2 = train_lda(dictionary_2, corpus_2, topics)

            os.makedirs(
                os.path.dirname(f"{data_folder}lda-gpt2_nt-wiki_nt-top_p/wiki_nt{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                exist_ok=True)
            with open(f"{data_folder}lda-gpt2_nt-wiki_nt-top_p/wiki_nt{extra_folder}/{combi}/{topics}/corpus_{topics}",
                      "w") as file:
                json.dump(corpus_2, file, indent=2)
            dictionary_2.save(
                f"{data_folder}lda-gpt2_nt-wiki_nt-top_p/wiki_nt{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_2.save(f"{data_folder}lda-gpt2_nt-wiki_nt-top_p/wiki_nt{extra_folder}/{combi}/{topics}/ldamodel_{topics}")

    # Create LDA Model for gpt2_nt-top_p and gpt2_nt-top_p
    elif dataset == 15 or dataset == 16:
        data_train_1 = load_json(f"{data_folder}dataset1-gpt2-wiki_nt-top_p.json")
        data_train_2 = load_json(f"{data_folder}dataset2-gpt2-wiki_nt-top_p.json")
        dictionary_1, corpus_1, dictionary_2, corpus_2 = tokenize_special(data_train_1, data_train_2, union)

        set_seed(seed2)
        if dataset == 15:
            model_1 = train_lda(dictionary_1, corpus_1, topics)

            os.makedirs(os.path.dirname(f"{data_folder}lda-gpt2_nt-gpt2_nt-top_p/gpt2_nt_1{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                        exist_ok=True)
            with open(f"{data_folder}lda-gpt2_nt-gpt2_nt-top_p/gpt2_nt_1{extra_folder}/{combi}/{topics}/corpus_{topics}", "w") as file:
                json.dump(corpus_1, file, indent=2)
            dictionary_1.save(f"{data_folder}lda-gpt2_nt-gpt2_nt-top_p/gpt2_nt_1{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_1.save(f"{data_folder}lda-gpt2_nt-gpt2_nt-top_p/gpt2_nt_1{extra_folder}/{combi}/{topics}/ldamodel_{topics}")
        else:
            model_2 = train_lda(dictionary_2, corpus_2, topics)

            os.makedirs(os.path.dirname(f"{data_folder}lda-gpt2_nt-gpt2_nt-top_p/gpt2_nt_2{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                        exist_ok=True)
            with open(f"{data_folder}lda-gpt2_nt-gpt2_nt-top_p/gpt2_nt_2{extra_folder}/{combi}/{topics}/corpus_{topics}", "w") as file:
                json.dump(corpus_2, file, indent=2)
            dictionary_2.save(f"{data_folder}lda-gpt2_nt-gpt2_nt-top_p/gpt2_nt_2{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_2.save(f"{data_folder}lda-gpt2_nt-gpt2_nt-top_p/gpt2_nt_2{extra_folder}/{combi}/{topics}/ldamodel_{topics}")
    ##
    # Create LDA Model for arxiv and gpt2_nt-typ_p
    elif dataset == 17 or dataset == 18:
        data_train_1 = load_json(f"{data_folder}dataset1-gpt2-wiki_nt-typ_p.json")
        data_train_2 = load_arxiv(samples)
        dictionary_1, corpus_1, dictionary_2, corpus_2 = tokenize_special(data_train_1, data_train_2, union)

        set_seed(seed2)
        if dataset == 17:
            model_1 = train_lda(dictionary_1, corpus_1, topics)

            os.makedirs(os.path.dirname(
                f"{data_folder}lda-gpt2_nt-arxiv-typ_p/gpt2_nt{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                exist_ok=True)
            with open(f"{data_folder}lda-gpt2_nt-arxiv-typ_p/gpt2_nt{extra_folder}/{combi}/{topics}/corpus_{topics}",
                      "w") as file:
                json.dump(corpus_1, file, indent=2)
            dictionary_1.save(
                f"{data_folder}lda-gpt2_nt-arxiv-typ_p/gpt2_nt{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_1.save(
                f"{data_folder}lda-gpt2_nt-arxiv-typ_p/gpt2_nt{extra_folder}/{combi}/{topics}/ldamodel_{topics}")
        else:
            model_2 = train_lda(dictionary_2, corpus_2, topics)

            os.makedirs(
                os.path.dirname(
                    f"{data_folder}lda-gpt2_nt-arxiv-typ_p/arxiv{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                exist_ok=True)
            with open(f"{data_folder}lda-gpt2_nt-arxiv-typ_p/arxiv{extra_folder}/{combi}/{topics}/corpus_{topics}",
                      "w") as file:
                json.dump(corpus_2, file, indent=2)
            dictionary_2.save(
                f"{data_folder}lda-gpt2_nt-arxiv-typ_p/arxiv{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_2.save(f"{data_folder}lda-gpt2_nt-arxiv-typ_p/arxiv{extra_folder}/{combi}/{topics}/ldamodel_{topics}")

    # Create LDA Model for wikitext and gpt2_nt-typ_p
    elif dataset == 19 or dataset == 20:
        data_train_1 = load_json(f"{data_folder}dataset1-gpt2-wiki_nt-typ_p.json")
        data_train_2 = load_wikitext(samples)
        dictionary_1, corpus_1, dictionary_2, corpus_2 = tokenize_special(data_train_1, data_train_2, union)

        set_seed(seed2)
        if dataset == 19:
            model_1 = train_lda(dictionary_1, corpus_1, topics)

            os.makedirs(os.path.dirname(
                f"{data_folder}lda-gpt2_nt-wiki_nt-typ_p/gpt2_nt{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                exist_ok=True)
            with open(f"{data_folder}lda-gpt2_nt-wiki_nt-typ_p/gpt2_nt{extra_folder}/{combi}/{topics}/corpus_{topics}",
                      "w") as file:
                json.dump(corpus_1, file, indent=2)
            dictionary_1.save(
                f"{data_folder}lda-gpt2_nt-wiki_nt-typ_p/gpt2_nt{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_1.save(
                f"{data_folder}lda-gpt2_nt-wiki_nt-typ_p/gpt2_nt{extra_folder}/{combi}/{topics}/ldamodel_{topics}")
        else:
            model_2 = train_lda(dictionary_2, corpus_2, topics)

            os.makedirs(
                os.path.dirname(
                    f"{data_folder}lda-gpt2_nt-wiki_nt-typ_p/wiki_nt{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                exist_ok=True)
            with open(f"{data_folder}lda-gpt2_nt-wiki_nt-typ_p/wiki_nt{extra_folder}/{combi}/{topics}/corpus_{topics}",
                      "w") as file:
                json.dump(corpus_2, file, indent=2)
            dictionary_2.save(
                f"{data_folder}lda-gpt2_nt-wiki_nt-typ_p/wiki_nt{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_2.save(
                f"{data_folder}lda-gpt2_nt-wiki_nt-typ_p/wiki_nt{extra_folder}/{combi}/{topics}/ldamodel_{topics}")

    # Create LDA Model for gpt2_nt-typ_p and gpt2_nt-typ_p
    elif dataset == 21 or dataset == 22:
        data_train_1 = load_json(f"{data_folder}dataset1-gpt2-wiki_nt-typ_p.json")
        data_train_2 = load_json(f"{data_folder}dataset2-gpt2-wiki_nt-typ_p.json")
        dictionary_1, corpus_1, dictionary_2, corpus_2 = tokenize_special(data_train_1, data_train_2, union)

        set_seed(seed2)
        if dataset == 21:
            model_1 = train_lda(dictionary_1, corpus_1, topics)

            os.makedirs(
                os.path.dirname(
                    f"{data_folder}lda-gpt2_nt-gpt2_nt-typ_p/gpt2_nt_1{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                exist_ok=True)
            with open(
                    f"{data_folder}lda-gpt2_nt-gpt2_nt-typ_p/gpt2_nt_1{extra_folder}/{combi}/{topics}/corpus_{topics}",
                    "w") as file:
                json.dump(corpus_1, file, indent=2)
            dictionary_1.save(
                f"{data_folder}lda-gpt2_nt-gpt2_nt-typ_p/gpt2_nt_1{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_1.save(
                f"{data_folder}lda-gpt2_nt-gpt2_nt-typ_p/gpt2_nt_1{extra_folder}/{combi}/{topics}/ldamodel_{topics}")
        else:
            model_2 = train_lda(dictionary_2, corpus_2, topics)

            os.makedirs(
                os.path.dirname(
                    f"{data_folder}lda-gpt2_nt-gpt2_nt-typ_p/gpt2_nt_2{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                exist_ok=True)
            with open(
                    f"{data_folder}lda-gpt2_nt-gpt2_nt-typ_p/gpt2_nt_2{extra_folder}/{combi}/{topics}/corpus_{topics}",
                    "w") as file:
                json.dump(corpus_2, file, indent=2)
            dictionary_2.save(
                f"{data_folder}lda-gpt2_nt-gpt2_nt-typ_p/gpt2_nt_2{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_2.save(
                f"{data_folder}lda-gpt2_nt-gpt2_nt-typ_p/gpt2_nt_2{extra_folder}/{combi}/{topics}/ldamodel_{topics}")

    # Create LDA Model for gpt2_nt and gpt2
    elif dataset == 23 or dataset == 24:
        data_train_1 = load_json(f"{data_folder}dataset1-gpt2-wiki_nt.json")
        data_train_2 = load_json(f"{data_folder}dataset1-gpt2.json")
        dictionary_1, corpus_1, dictionary_2, corpus_2 = tokenize_special(data_train_1, data_train_2, union)

        set_seed(seed2)
        if dataset == 23:
            model_1 = train_lda(dictionary_1, corpus_1, topics)

            os.makedirs(
                os.path.dirname(f"{data_folder}lda-gpt2_nt-gpt2/gpt2_nt{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                exist_ok=True)
            with open(f"{data_folder}lda-gpt2_nt-gpt2/gpt2_nt{extra_folder}/{combi}/{topics}/corpus_{topics}",
                      "w") as file:
                json.dump(corpus_1, file, indent=2)
            dictionary_1.save(f"{data_folder}lda-gpt2_nt-gpt2/gpt2_nt{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_1.save(f"{data_folder}lda-gpt2_nt-gpt2/gpt2_nt{extra_folder}/{combi}/{topics}/ldamodel_{topics}")
        else:
            model_2 = train_lda(dictionary_2, corpus_2, topics)

            os.makedirs(
                os.path.dirname(f"{data_folder}lda-gpt2_nt-gpt2/gpt2{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                exist_ok=True)
            with open(f"{data_folder}lda-gpt2_nt-gpt2/gpt2{extra_folder}/{combi}/{topics}/corpus_{topics}",
                      "w") as file:
                json.dump(corpus_2, file, indent=2)
            dictionary_2.save(f"{data_folder}lda-gpt2_nt-gpt2/gpt2{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_2.save(f"{data_folder}lda-gpt2_nt-gpt2/gpt2{extra_folder}/{combi}/{topics}/ldamodel_{topics}")
    # Create LDA Model for wiki_nt and gpt2
    elif dataset == 25 or dataset == 26:
        data_train_1 = load_wikitext()
        data_train_2 = load_json(f"{data_folder}dataset1-gpt2.json")
        dictionary_1, corpus_1, dictionary_2, corpus_2 = tokenize_special(data_train_1, data_train_2, union)

        set_seed(seed2)
        if dataset == 25:
            model_1 = train_lda(dictionary_1, corpus_1, topics)

            os.makedirs(
                os.path.dirname(
                    f"{data_folder}lda-gpt2-wiki_nt/wiki_nt{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                exist_ok=True)
            with open(f"{data_folder}lda-gpt2-wiki_nt/wiki_nt{extra_folder}/{combi}/{topics}/corpus_{topics}",
                      "w") as file:
                json.dump(corpus_1, file, indent=2)
            dictionary_1.save(
                f"{data_folder}lda-gpt2-wiki_nt/wiki_nt{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_1.save(f"{data_folder}lda-gpt2-wiki_nt/wiki_nt{extra_folder}/{combi}/{topics}/ldamodel_{topics}")
        else:
            model_2 = train_lda(dictionary_2, corpus_2, topics)

            os.makedirs(
                os.path.dirname(f"{data_folder}lda-gpt2-wiki_nt/gpt2{extra_folder}/{combi}/{topics}/corpus_{topics}"),
                exist_ok=True)
            with open(f"{data_folder}lda-gpt2-wiki_nt/gpt2{extra_folder}/{combi}/{topics}/corpus_{topics}",
                      "w") as file:
                json.dump(corpus_2, file, indent=2)
            dictionary_2.save(f"{data_folder}lda-gpt2-wiki_nt/gpt2{extra_folder}/{combi}/{topics}/dictionary_{topics}")
            model_2.save(f"{data_folder}lda-gpt2-wiki_nt/gpt2{extra_folder}/{combi}/{topics}/ldamodel_{topics}")


main()
