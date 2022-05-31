import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import LdaMulticore
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import ScalarFormatter
from matplotlib.pyplot import cm
from wordcloud import WordCloud

MODELS = ["arxiv-arxiv", "gpt2-arxiv", "gpt2-gpt2", "gpt2-trafo_xl", "gpt2-trafo_xl_nt", "gpt2-trafo_xl_nt-top_p", "gpt2-trafo_xl_nt-typ_p", "gpt2-wiki_nt", "gpt2_nt-arxiv", "gpt2_nt-arxiv-top_p", "gpt2_nt-arxiv-typ_p", "gpt2_nt-gpt2", "gpt2_nt-gpt2-top_p", "gpt2_nt-gpt2-typ_p", "gpt2_nt-gpt2_nt", "gpt2_nt-gpt2_nt-top_p", "gpt2_nt-gpt2_nt-typ_p", "gpt2_nt-trafo_xl", "gpt2_nt-trafo_xl-top_p", "gpt2_nt-trafo_xl-typ_p", "gpt2_nt-trafo_xl_nt", "gpt2_nt-trafo_xl_nt-top_p", "gpt2_nt-trafo_xl_nt-typ_p", "gpt2_nt-wiki_nt", "gpt2_nt-wiki_nt-top_p", "gpt2_nt-wiki_nt-typ_p", "trafo_xl-arxiv", "trafo_xl-trafo_xl", "trafo_xl-wiki_nt", "trafo_xl_nt-arxiv", "trafo_xl_nt-arxiv-top_p", "trafo_xl_nt-arxiv-typ_p", "trafo_xl_nt-trafo_xl", "trafo_xl_nt-trafo_xl-top_p", "trafo_xl_nt-trafo_xl-typ_p", "trafo_xl_nt-trafo_xl_nt", "trafo_xl_nt-trafo_xl_nt-top_p", "trafo_xl_nt-trafo_xl_nt-typ_p", "trafo_xl_nt-wiki_nt", "trafo_xl_nt-wiki_nt-top_p", "trafo_xl_nt-wiki_nt-typ_p", "wiki_nt-arxiv", "wiki_nt-wiki_nt"]
TOPIC_MODELS = ["classic_lda", "neural_lda"]


def string_mod(string=""):
    """
    Helper function for modifying the keys of all scores and measurements after creation. used for "beautiful" plots
    """
    return string.replace('gpt2_nt', 'gpt2_ours').replace('wiki_nt', 'wikitext').replace("trafo_xl_nt", "trafo_xl_ours")


def generate_var_tt_score_plots(data_path, tokenization, sample_size, num_topics):
    """
    Function for generating the plots for the variation in the our score
    """

    models = ["gpt2_nt-gpt2_nt", "gpt2_nt-wiki_nt", "gpt2_nt-arxiv"]
    merge_types = ["intersection", "union"]

    for merge_type in merge_types:

        # defining parameters for the plot
        if merge_type == "intersection":
            tt_title = "Score Variance Graph for classic LDA Models"
            tt_subtitle = f"{sample_size} samples, {tokenization}, intersected dicts\nlower score indicates a stronger similarity between the two topic models"
            tt_y_label = "Metric"
            mode = 'is'
        elif merge_type == "union":
            tt_title = "Score Variance Graph for classic LDA Models"
            tt_subtitle = f"{sample_size} samples, {tokenization}, unionized dicts\nlower score indicates a stronger similarity between the two topic models"
            tt_y_label = "Metric"
            mode = 'un'
        else:
            raise ValueError(f">> ERROR: Unknown merge_type: {merge_type}")

        # plot initialization
        plt.clf()
        fig, axes = plt.subplots()

        for model in models:
            data_path_ = f"{data_path}{tokenization}/{sample_size}/{model}/"

            tt_score_file_name = f"{data_path_}tt_score.json"
            if os.path.isfile(tt_score_file_name):
                with open(tt_score_file_name, "r") as file:
                    tt_scores = json.load(file)
            else:
                raise ValueError(f">> ERROR: wrong tt_score_file_name: {tt_score_file_name}")

            mat = []
            key_split = ""

            # accumulating score values
            for key in tt_scores.keys():
                key_split = key.split("-")
                if merge_type != key_split[-2] and merge_type != key_split[-1]:
                    continue
                mat.append(tt_scores[key])

            # extracting mean, lower and upper boundaries for all score values
            mat = np.asarray(mat)
            y_upper = np.max(mat, axis=0)
            y_lower = np.min(mat, axis=0)
            y_est = np.mean(mat, axis=0)

            # plot values with boundaries
            axes.plot(num_topics, y_est, label=f"{string_mod(key_split[1])} vs. {string_mod(key_split[2])}", linewidth=3)
            axes.fill_between(num_topics, y_lower, y_upper, alpha=0.2)

        # matplotlib settings for beautiful plots
        plt.suptitle(tt_title, fontsize=15)
        axes.set_title(tt_subtitle, fontsize=10)
        axes.set_xscale('log')
        plt.ylim((0, 0.9))
        plt.xlim((2, 100))
        axes.set_xlabel('Number of Topics')
        axes.set_ylabel(tt_y_label)
        axes.set_xticks(num_topics)
        axes.get_xaxis().set_major_formatter(ScalarFormatter())
        plt.grid(linestyle='--', linewidth=0.5)
        font = FontProperties()
        font.set_size(8)
        axes.legend(loc="upper right", prop=font)
        fig.tight_layout()

        plt.savefig(f"{data_path}/evaluation/{tokenization}-{sample_size}-var-tt-{mode}.png", dpi=300)
        plt.close('all')


def generate_var_cv_score_plots(data_path, tokenization, sample_size, num_topics):
    """
        Function for generating plots for the variation in the quality score for our topic models
    """

    models = ["gpt2_nt-gpt2_nt", "gpt2_nt-wiki_nt", "gpt2_nt-arxiv"]
    merge_types = ["intersection", "union"]
    color = iter(cm.tab10(np.linspace(0, 1, 10)))
    for merge_type in merge_types:
        plt.clf()
        fig, axes = plt.subplots()

        if merge_type == "intersection":
            cv_title = f"C_v Score Variance Graph for the classic LDA Model"
            cv_subtitle = f"{sample_size} samples, {tokenization}, intersected dicts\nhigher score indicates a stronger correlation to human-annotated coherence judgements"
            cv_y_label = "C_v Score"
            mode = 'is'
        elif merge_type == "union":
            cv_title = f"C_v Score Variance Graph for the classic LDA Model"
            cv_subtitle = f"{sample_size} samples, {tokenization}, unionized dicts\nhigher score indicates a stronger correlation to human-annotated coherence judgements"
            cv_y_label = "C_v Score"
            mode = 'un'
        else:
            raise ValueError(f">> ERROR: Unknown merge_type: {merge_type}")

        for model in models:
            data_path_ = f"{data_path}{tokenization}/{sample_size}/{model}/"

            cv_score_file_name = f"{data_path_}cv_score.json"
            if os.path.isfile(cv_score_file_name):
                with open(cv_score_file_name, "r") as file:
                    cv_scores = json.load(file)
            else:
                raise ValueError(f">> ERROR: wrong cv_score_file_name: {cv_score_file_name}")

            model_split = model.split("-")
            if model_split[0] == model_split[1]:
                model_split[0] += "_1"
                model_split[1] += "_2"

            for focus in model_split:
                mat = []
                key_split = ""
                for key in cv_scores.keys():
                    key_split = key.split("-")

                    if (focus != key_split[-3] and key[-2] == "-") or (key[-2] != "-" and focus != key_split[-2]) or merge_type not in key or "classic_lda" not in key:
                        continue
                    mat.append(cv_scores[key])

                mat = np.asarray(mat)
                y_upper = np.max(mat, axis=0)
                y_lower = np.min(mat, axis=0)
                y_est = np.mean(mat, axis=0)

                axes.plot(num_topics, y_est, label=f"{string_mod(focus)} ({string_mod(key_split[1])} vs. {string_mod(key_split[2])})", linewidth=3)#, c=next(color))
                axes.fill_between(num_topics, y_lower, y_upper, alpha=0.2)

        # matplotlib settings for beautiful plots
        plt.suptitle(cv_title, fontsize=15)
        axes.set_title(cv_subtitle, fontsize=10)
        axes.set_xscale('log')
        axes.set_xlabel('Number of Topics')
        plt.ylim((0.2, 0.7))
        plt.xlim((2, 100))
        axes.set_ylabel(cv_y_label)
        axes.set_xticks(num_topics)
        axes.get_xaxis().set_major_formatter(ScalarFormatter())
        plt.grid(linestyle='--', linewidth=0.5)
        font = FontProperties()
        font.set_size(8)
        axes.legend(loc="lower right", prop=font)
        fig.tight_layout()

        plt.savefig(f"{data_path}/evaluation/{tokenization}-{sample_size}-var-cv-{mode}.png", dpi=300)
        plt.close('all')


def generate_crossmodel_cv_plots(data_path, tokenization, sample_size, num_topics):
    """
        Function for generating plots containing the quality score for all our topic models
    """

    foci_ = [x for x in ["gpt2_nt", "gpt2", "trafo_xl_nt", "trafo_xl", "wiki_nt", "arxiv"] if (sample_size <= 10000 or "trafo_xl" not in x)]

    for focus_ in foci_:
        models = [x for x in MODELS if focus_ in x and (sample_size <= 10000 or "trafo_xl" not in x)]
        merge_types = ["intersection", "union"]
        topic_models = [x for x in TOPIC_MODELS if sample_size <= 10000 or "neural_lda" not in x]
        sampling_focus = focus_ in ["gpt2_nt", "trafo_xl_nt"]

        for merge_type in merge_types:
            for topic_model in topic_models:
                if merge_type == "intersection":
                    cv_title = f"Crossmodel C_v Score Graph for {string_mod(focus_)}"
                    cv_subtitle = f"{topic_model}, {sample_size} samples, {tokenization}, intersected dicts\nhigher score indicates a stronger correlation to human-annotated coherence judgements"
                    cv_y_label = "C_v Score"
                    mode = 'is'
                elif merge_type == "union":
                    cv_title = f"Crossmodel C_v Score Graph for {string_mod(focus_)}"
                    cv_subtitle = f"{topic_model}, {sample_size} samples, {tokenization}, unionized dicts\nhigher score indicates a stronger correlation to human-annotated coherence judgements"
                    cv_y_label = "C_v Score"
                    mode = 'un'
                else:
                    raise ValueError(f">> ERROR: Unknown merge_type: {merge_type}")

                plt.clf()
                fig, axes = plt.subplots()
                legend = []
                legend_top_p = []
                legend_typ_p = []

                if sample_size <= 10000 and sampling_focus:
                    color_length = 8
                elif sample_size <= 10000 and not sampling_focus:
                    color_length = 12
                elif sampling_focus:
                    color_length = 6
                else:
                    color_length = 8

                color_norm = iter(np.flip(cm.Blues(np.linspace(0, 1, color_length)), axis=0))
                color_top_p = iter(np.flip(cm.Greens(np.linspace(0, 1, color_length)), axis=0))
                color_typ_p = iter(np.flip(cm.Reds(np.linspace(0, 1, color_length)), axis=0))

                for model in models:
                    data_path_ = f"{data_path}{tokenization}/{sample_size}/{model}/"

                    cv_score_file_name = f"{data_path_}cv_score.json"
                    if os.path.isfile(cv_score_file_name):
                        with open(cv_score_file_name, "r") as file:
                            cv_scores = json.load(file)
                    else:
                        raise ValueError(f">> ERROR: wrong cv_score_file_name: {cv_score_file_name}")

                    model_split = model.split("-")
                    foci = [focus_]
                    if model_split[0] == model_split[1]:
                        foci = [focus_ + "_1", focus_ + "_2"]
                    for focus in foci:
                        for key in cv_scores.keys():
                            key_split = key.split("-")
                            if topic_model != key_split[0] or merge_type != key_split[-1] or focus != key_split[-2]:
                                continue

                            if sampling_focus and model_split[-1] == "top_p":
                                style = 'dotted'
                                label = f"{focus} of {'-'.join(model_split[:-1])}"
                                c = next(color_top_p)
                            elif sampling_focus and model_split[-1] == "typ_p":
                                style = 'dashed'
                                label = f"{focus} of {'-'.join(model_split[:-1])}"
                                c = next(color_typ_p)
                            else:
                                style = 'solid'
                                label = f"{focus} of {model}"
                                c = next(color_norm)

                            line, = axes.plot(num_topics, cv_scores[key], label=label, c=c, linestyle=style, linewidth=2)

                            if sampling_focus and model_split[-1] == "top_p":
                                legend_top_p.append(line)
                            elif sampling_focus and model_split[-1] == "typ_p":
                                legend_typ_p.append(line)
                            else:
                                legend.append(line)

                # matplotlib settings for beautiful plots
                plt.suptitle(cv_title, fontsize=15)
                axes.set_title(cv_subtitle, fontsize=8, x=0.6)
                axes.set_xscale('log')
                axes.set_xlabel('Number of Topics')
                plt.ylim((0, 1))
                axes.set_ylabel(cv_y_label)
                axes.set_xticks(num_topics)
                axes.get_xaxis().set_major_formatter(ScalarFormatter())
                plt.grid(linestyle='--', linewidth=0.5)
                font = FontProperties()
                font.set_size(7)

                first_legend = axes.legend(handlelength=2, handles=legend, bbox_to_anchor=(1.04, 1), loc="upper left",
                                           prop=font, title='Ancestral sampling')

                if sampling_focus:
                    axes.add_artist(first_legend)
                    second_legend = axes.legend(handlelength=2, handles=legend_top_p, bbox_to_anchor=(1.04, 0.6),
                                                loc="upper left", prop=font, title='Top_p sampling')
                    axes.add_artist(second_legend)
                    third_legend = axes.legend(handlelength=2, handles=legend_typ_p, bbox_to_anchor=(1.04, 0.2),
                                               loc="upper left", prop=font, title='Typ_p sampling')

                fig.tight_layout()
                plt.savefig(
                    f"{data_path}/evaluation/{tokenization}-{sample_size}-crossmodel-cv-{topic_model}-{focus_}-{mode}.png",
                    dpi=300)
                plt.close('all')


def generate_crossmodel_cv_var_plots(data_path, tokenization, sample_size, num_topics):
    """
        Function for generating plots with the variation of the quality score of our topic models
    """

    merge_types = ["intersection", "union"]
    topic_models = [x for x in TOPIC_MODELS if sample_size <= 10000 or "neural_lda" not in x]

    for merge_type in merge_types:
        for topic_model in topic_models:
            for split in ["gpt2_nt", "trafo_xl_nt", "gpt2", "trafo_xl"]:
                if merge_type == "intersection":
                    cv_title = f"Crossmodel C_v Score Variance Graph"
                    cv_subtitle = f"{topic_model}, {sample_size} samples, {tokenization}, intersected dicts\nhigher score indicates a stronger correlation to human-annotated coherence judgements"
                    cv_y_label = "C_v Score"
                    mode = 'is'
                elif merge_type == "union":
                    cv_title = f"Crossmodel C_v Score Variance Graph"
                    cv_subtitle = f"{topic_model}, {sample_size} samples, {tokenization}, unionized dicts\nhigher score indicates a stronger correlation to human-annotated coherence judgements"
                    cv_y_label = "C_v Score"
                    mode = 'un'
                else:
                    raise ValueError(f">> ERROR: Unknown merge_type: {merge_type}")

                plt.clf()
                fig, axes = plt.subplots()
                legend = []
                legend_top_p = []
                legend_typ_p = []

                if split == "gpt2_nt":
                    foci_ = ["gpt2_nt"]
                elif split == "trafo_xl_nt" and sample_size <= 10000:
                    foci_ = ["trafo_xl_nt"]
                elif split == "gpt2":
                    foci_ = ["gpt2_nt", "gpt2", "wiki_nt", "arxiv"]
                elif split == "trafo_xl" and sample_size <= 10000:
                    foci_ = ["trafo_xl_nt", "trafo_xl", "wiki_nt", "arxiv"]
                else:
                    continue

                for focus_ in foci_:
                    lis = ["anc", "top_p", "typ_p"]
                    if len(foci_) > 1:
                        lis = ["anc"]
                    for sp in lis:
                        if focus_ != "gpt2_nt" and focus_ != "trafo_xl_nt" and sp != "anc":
                            continue
                        mat = []
                        if sp == "anc":
                            models = [x for x in MODELS if focus_ in x.split("-") and (sample_size <= 10000 or "trafo_xl" not in x) and
                                      (((focus_ == "gpt2_nt" or focus_ == "trafo_xl_nt") and x.split("-")[-1] not in ["top_p", "typ_p"]) or
                                       (focus_ != "gpt2_nt" and focus_ != "trafo_xl_nt"))]
                        else:
                            models = [x for x in MODELS if focus_ in x and (sample_size <= 10000 or "trafo_xl" not in x) and
                                      x.split("-")[-1] == sp]

                        for model in models:
                            data_path_ = f"{data_path}{tokenization}/{sample_size}/{model}/"

                            cv_score_file_name = f"{data_path_}cv_score.json"
                            if os.path.isfile(cv_score_file_name):
                                with open(cv_score_file_name, "r") as file:
                                    cv_scores = json.load(file)
                            else:
                                raise ValueError(f">> ERROR: wrong cv_score_file_name: {cv_score_file_name}")

                            model_split = model.split("-")
                            foci = [focus_]
                            if model_split[0] == model_split[1]:
                                foci = [focus_ + "_1", focus_ + "_2"]
                            for focus in foci:
                                for key in cv_scores.keys():
                                    key_split = key.split("-")

                                    if topic_model != key_split[0] or merge_type != key_split[-1] or focus != key_split[-2]:
                                        continue
                                    mat.append(cv_scores[key])

                        mat = np.asarray(mat)
                        y_upper = np.max(mat, axis=0)
                        y_lower = np.min(mat, axis=0)
                        y_est = np.mean(mat, axis=0)

                        if sp == "top_p":
                            style = 'dotted'
                        elif sp == "typ_p":
                            style = 'dashed'
                        else:
                            style = 'solid'

                        line, = axes.plot(num_topics, y_est, label=f"{string_mod(focus_)}", linestyle=style, linewidth=2)
                        axes.fill_between(num_topics, y_lower, y_upper, alpha=0.2)

                        if sp == "top_p":
                            legend_top_p.append(line)
                        elif sp == "typ_p":
                            legend_typ_p.append(line)
                        else:
                            legend.append(line)

                # matplotlib settings for beautiful plots
                plt.suptitle(cv_title, fontsize=15)
                axes.set_title(cv_subtitle, fontsize=10)
                axes.set_xscale('log')
                axes.set_xlabel('Number of Topics')
                plt.ylim((0.1, 0.8))
                plt.xlim((2, 100))
                axes.set_ylabel(cv_y_label)
                axes.set_xticks(num_topics)
                axes.get_xaxis().set_major_formatter(ScalarFormatter())
                plt.grid(linestyle='--', linewidth=0.5)
                font = FontProperties()
                font.set_size(8)

                if len(foci_) == 1:
                    first_legend = axes.legend(handlelength=2, handles=legend, bbox_to_anchor=(1, 0.45),
                                               loc="upper right", prop=font, title='Ancestral sampling')
                    plt.setp(first_legend.get_title(), fontsize=10)

                    axes.add_artist(first_legend)
                    second_legend = axes.legend(handlelength=2, handles=legend_top_p, bbox_to_anchor=(1, 0.3),
                                                loc="upper right", prop=font, title='Top_p sampling')
                    plt.setp(second_legend.get_title(), fontsize=10)
                    axes.add_artist(second_legend)
                    third_legend = axes.legend(handlelength=2, handles=legend_typ_p, bbox_to_anchor=(1, 0.15),
                                               loc="upper right", prop=font, title='Typ_p sampling')
                    plt.setp(third_legend.get_title(), fontsize=10)
                else:
                    first_legend = axes.legend(handlelength=2, handles=legend, bbox_to_anchor=(1, 0), loc="lower right",
                                               prop=font, title='Ancestral sampling')
                    plt.setp(first_legend.get_title(), fontsize=10)
                fig.tight_layout()
                plt.savefig(f"{data_path}/evaluation/{tokenization}-{sample_size}-crossmodel-cv-var-{topic_model}-{split}-{mode}.png", dpi=300)
                plt.close('all')


def generate_crossmodel_tt_plots(data_path, tokenization, sample_size, num_topics):
    """
        Function for generating plots for our score across topic model comparisons
    """

    excluded = ["trafo_xl", "gpt2"] if sample_size <= 10000 else ["trafo_xl"]

    for excl in excluded:
        included = "Transformer XL" if excl == "gpt2" else "GPT2"
        indluded_short = "trafo_xl" if excl == "gpt2" else "gpt2"
        models = [x for x in MODELS if excl not in x and (sample_size <= 10000 or "trafo_xl" not in x)]
        merge_types = ["intersection", "union"]
        topic_models = [x for x in TOPIC_MODELS if sample_size <= 10000 or "neural_lda" not in x]

        for merge_type in merge_types:
            for topic_model in topic_models:
                for sampling in [True, False]:
                    if merge_type == "intersection":
                        cv_title = f"Crossmodel Score Graph for {included}"
                        cv_subtitle = f"{topic_model}, {sample_size} samples, {tokenization}, intersected dicts\nlower score indicates a stronger similarity between the two topic models"
                        cv_y_label = "Metric"
                        mode = 'is'
                    elif merge_type == "union":
                        cv_title = f"Crossmodel Score Graph for {included}"
                        cv_subtitle = f"{topic_model}, {sample_size} samples, {tokenization}, unionized dicts\nlower score indicates a stronger similarity between the two topic models"
                        cv_y_label = "Metric"
                        mode = 'un'
                    else:
                        raise ValueError(f">> ERROR: Unknown merge_type: {merge_type}")

                    plt.clf()
                    fig, axes = plt.subplots()
                    legend = []
                    legend_top_p = []
                    legend_typ_p = []

                    color = iter(cm.tab10(np.linspace(0, 1, 10)))
                    color_map = dict()

                    for model in models:
                        data_path_ = f"{data_path}{tokenization}/{sample_size}/{model}/"

                        tt_score_file_name = f"{data_path_}tt_score.json"
                        if os.path.isfile(tt_score_file_name):
                            with open(tt_score_file_name, "r") as file:
                                tt_scores = json.load(file)
                        else:
                            raise ValueError(f">> ERROR: wrong tt_score_file_name: {tt_score_file_name}")

                        model_split = model.split("-")

                        for key in tt_scores.keys():
                            key_split = key.split("-")
                            if topic_model != key_split[0] or merge_type != key_split[-1] or \
                                    (not sampling and key_split[-2] in ["top_p", "typ_p"]) or \
                                    (sampling and key_split[-2] in ["wiki_nt", "arxiv", "gpt2", "trafo_xl"] and key_split[-3] in ["wiki_nt", "arxiv", "gpt2", "trafo_xl"]):
                                continue

                            label = f"{model_split[0]} vs. {model_split[1]}"
                            color_map_key = f"{model_split[0]}-{model_split[1]}"
                            if color_map_key not in color_map.keys():
                                color_map[color_map_key] = next(color)
                            c = color_map[color_map_key]

                            if model_split[-1] == "top_p":
                                style = 'dotted'
                            elif model_split[-1] == "typ_p":
                                style = 'dashed'
                            else:
                                style = 'solid'

                            line, = axes.plot(num_topics, tt_scores[key], label=string_mod(label), c=c, linestyle=style, linewidth=2)

                            if model_split[-1] == "top_p":
                                legend_top_p.append(line)
                            elif model_split[-1] == "typ_p":
                                legend_typ_p.append(line)
                            else:
                                legend.append(line)

                    # matplotlib settings for beautiful plots
                    plt.suptitle(cv_title, fontsize=15)
                    axes.set_title(cv_subtitle, fontsize=10, x=0.6)
                    axes.set_xscale('log')
                    axes.set_xlabel('Number of Topics')
                    plt.ylim((0, 1))
                    plt.xlim((2, 100))
                    axes.set_ylabel(cv_y_label)
                    axes.set_xticks(num_topics)
                    axes.get_xaxis().set_major_formatter(ScalarFormatter())
                    plt.grid(linestyle='--', linewidth=0.5)
                    font = FontProperties()
                    font.set_size(7)
                    title_font = 9

                    first_legend = axes.legend(handlelength=2, handles=legend, bbox_to_anchor=(1.01, 1), loc="upper left",
                                               prop=font, title='Ancestral sampling', title_fontsize=title_font)
                    if sampling:
                        axes.add_artist(first_legend)
                        second_legend = axes.legend(handlelength=2, handles=legend_top_p, bbox_to_anchor=(1.01, 0.75),
                                                    loc="upper left", prop=font, title='Top_p sampling', title_fontsize=title_font)
                        axes.add_artist(second_legend)
                        third_legend = axes.legend(handlelength=2, handles=legend_typ_p, bbox_to_anchor=(1.01, 0.5),
                                                   loc="upper left", prop=font, title='Typ_p sampling', title_fontsize=title_font)

                    fig.tight_layout()
                    smpl = "b" if sampling else "a"
                    plt.savefig(
                        f"{data_path}/evaluation/{tokenization}-{sample_size}-crossmodel-tt-{indluded_short}-{topic_model}-{smpl}-{mode}.png",
                        dpi=300)
                    plt.close('all')


def generate_crosslm_tt_plots(data_path, tokenization, sample_size, num_topics):
    """
        Function for generating plot for evaluating the score across language models
    """
    if sample_size > 10000:
        return
    """
    models = []
    for x in MODELS:
        x_split = x.split("-")
        if x_split[0] in ["arxiv", "wiki_nt"] or x_split[1] in ["arxiv", "wiki_nt"]:
            continue
        models.append(x)
    """
    models = [x for x in MODELS if ("trafo_xl" in x and "gpt2" in x) or ("trafo_xl" in x and "gpt2" in x)]
    merge_types = ["intersection", "union"]
    topic_models = TOPIC_MODELS

    for merge_type in merge_types:
        for topic_model in topic_models:
            if merge_type == "intersection":
                cv_title = f"CrossLM Score Graph for GPT2 & TrafoXL"
                cv_subtitle = f"{topic_model}, {sample_size} samples, {tokenization}, intersected dicts\nlower score indicates a stronger similarity between the two topic models"
                cv_y_label = "Score"
                mode = 'is'
            elif merge_type == "union":
                cv_title = f"CrossLM Score Graph for GPT2 & TrafoXL"
                cv_subtitle = f"{topic_model}, {sample_size} samples, {tokenization}, unionized dicts\nlower score indicates a stronger similarity between the two topic models"
                cv_y_label = "Score"
                mode = 'un'
            else:
                raise ValueError(f">> ERROR: Unknown merge_type: {merge_type}")

            plt.clf()
            fig, axes = plt.subplots()
            legend = []
            legend_top_p = []
            legend_typ_p = []

            color = iter(cm.tab10(np.linspace(0, 1, 10)))
            color_map = dict()

            for model in models:
                data_path_ = f"{data_path}{tokenization}/{sample_size}/{model}/"

                tt_score_file_name = f"{data_path_}tt_score.json"
                if os.path.isfile(tt_score_file_name):
                    with open(tt_score_file_name, "r") as file:
                        tt_scores = json.load(file)
                else:
                    raise ValueError(f">> ERROR: wrong tt_score_file_name: {tt_score_file_name}")

                model_split = model.split("-")

                for key in tt_scores.keys():
                    key_split = key.split("-")
                    if topic_model != key_split[0] or merge_type != key_split[-1]:
                        continue

                    label = f"{model_split[0]} vs. {model_split[1]}"
                    color_map_key = f"{model_split[0]}-{model_split[1]}"
                    if color_map_key not in color_map.keys():
                        color_map[color_map_key] = next(color)
                    c = color_map[color_map_key]

                    if model_split[-1] == "top_p":
                        style = 'dotted'
                    elif model_split[-1] == "typ_p":
                        style = 'dashed'
                    else:
                        style = 'solid'

                    line, = axes.plot(num_topics, tt_scores[key], label=string_mod(label), c=c, linestyle=style, linewidth=2)

                    if model_split[-1] == "top_p":
                        legend_top_p.append(line)
                    elif model_split[-1] == "typ_p":
                        legend_typ_p.append(line)
                    else:
                        legend.append(line)

            # matplotlib settings for beautiful plots
            plt.suptitle(cv_title, fontsize=15)
            axes.set_title(cv_subtitle, fontsize=8, x=0.6)
            axes.set_xscale('log')
            axes.set_xlabel('Number of Topics')
            plt.ylim((0, 1))
            axes.set_ylabel(cv_y_label)
            axes.set_xticks(num_topics)
            axes.get_xaxis().set_major_formatter(ScalarFormatter())
            plt.grid(linestyle='--', linewidth=0.5)

            font = FontProperties()
            font.set_size(7)
            title_font = 9

            first_legend = axes.legend(handlelength=2, handles=legend, bbox_to_anchor=(1.01, 1), loc="upper left",
                                       prop=font, title='Ancestral sampling', title_fontsize=title_font)
            axes.add_artist(first_legend)
            second_legend = axes.legend(handlelength=2, handles=legend_top_p, bbox_to_anchor=(1.01, 0.7),
                                        loc="upper left", prop=font, title='Top_p sampling', title_fontsize=title_font)
            axes.add_artist(second_legend)
            third_legend = axes.legend(handlelength=2, handles=legend_typ_p, bbox_to_anchor=(1.01, 0.45),
                                       loc="upper left", prop=font, title='Typ_p sampling', title_fontsize=title_font)

            fig.tight_layout()
            plt.savefig(
                f"{data_path}/evaluation/{tokenization}-{sample_size}-crosslm-tt-{topic_model}-{mode}.png",
                dpi=300)
            plt.close('all')


def generate_specific_wordclouds():
    """
    Function for generating wordclouds out of topics
    """
    for topics in [5, 10]:
        path_ldamodel_1 = f"./data/Unigrams/100000/gpt2_nt-wiki_nt/gpt2_nt/intersection/classic_lda/{topics}/model"
        path_ldamodel_2 = f"./data/Unigrams/100000/gpt2_nt-wiki_nt/wiki_nt/intersection/classic_lda/{topics}/model"
        ldamodel_1 = LdaMulticore.load(path_ldamodel_1)
        ldamodel_2 = LdaMulticore.load(path_ldamodel_2)

        topic = 2 if topics == 5 else 3

        plt.clf()
        plt.figure()
        plt.imshow(WordCloud().fit_words(dict(ldamodel_1.show_topic(topic, 50))))
        plt.axis("off")
        # plt.suptitle(f"Wordcloud of gpt2_ours, topic {topic} of {topics}", fontsize=15)
        plt.savefig(f"./data/wordcloud-gpt2_nt-wiki_nt-gpt2_nt-{topics}-{topic}.png", dpi=300)

        plt.clf()
        plt.figure()
        plt.imshow(WordCloud().fit_words(dict(ldamodel_2.show_topic(topic, 50))))
        plt.axis("off")
        # plt.suptitle(f"Wordcloud of wikitext, topic {topic} of {topics}", fontsize=15)
        plt.savefig(f"./data/wordcloud-gpt2_nt-wiki_nt-wiki_nt-{topics}-{topic}.png", dpi=300)


def generate_score_plots(data_folder_path):
    tokenizations = ["Trigrams+Bigrams+Unigrams", "Bigrams+Unigrams", "Unigrams"]
    sample_sizes = [10000, 100000]
    num_topics = [2, 3, 5, 10, 20, 50, 100]

    if data_folder_path[-1] != '/':
        data_folder_path += '/'
    generate_specific_wordclouds()

    for tokenization in tokenizations:
        for sample_size in sample_sizes:
            generate_var_tt_score_plots(data_folder_path, tokenization, sample_size, num_topics)
            generate_var_cv_score_plots(data_folder_path, tokenization, sample_size, num_topics)
            generate_crossmodel_cv_plots(data_folder_path, tokenization, sample_size, num_topics)
            generate_crossmodel_cv_var_plots(data_folder_path, tokenization, sample_size, num_topics)
            generate_crossmodel_tt_plots(data_folder_path, tokenization, sample_size, num_topics)
            generate_crosslm_tt_plots(data_folder_path, tokenization, sample_size, num_topics)


def main():
    """
    Generates plots for all available score values

    Make sure to have generated the same data or adjust parameters accordingly in the 'generate_score_plots' function

    Examples:
        python plot_scores.py /cluster/work/cotterell/knobelf/data/
    """
    generate_score_plots(sys.argv[1])


def test():
    generate_score_plots("./data")


if __name__ == '__main__':
    main()
    # test()
