import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import ScalarFormatter
from matplotlib.pyplot import cm

MODELS = ["arxiv-arxiv", "gpt2-arxiv", "gpt2-gpt2", "gpt2-trafo_xl", "gpt2-trafo_xl_nt", "gpt2-trafo_xl_nt-top_p", "gpt2-trafo_xl_nt-typ_p", "gpt2-wiki_nt", "gpt2_nt-arxiv", "gpt2_nt-arxiv-top_p", "gpt2_nt-arxiv-typ_p", "gpt2_nt-gpt2", "gpt2_nt-gpt2-top_p", "gpt2_nt-gpt2-typ_p", "gpt2_nt-gpt2_nt", "gpt2_nt-gpt2_nt-top_p", "gpt2_nt-gpt2_nt-typ_p", "gpt2_nt-trafo_xl", "gpt2_nt-trafo_xl-top_p", "gpt2_nt-trafo_xl-typ_p", "gpt2_nt-trafo_xl_nt", "gpt2_nt-trafo_xl_nt-top_p", "gpt2_nt-trafo_xl_nt-typ_p", "gpt2_nt-wiki_nt", "gpt2_nt-wiki_nt-top_p", "gpt2_nt-wiki_nt-typ_p", "trafo_xl-arxiv", "trafo_xl-trafo_xl", "trafo_xl-wiki_nt", "trafo_xl_nt-arxiv", "trafo_xl_nt-arxiv-top_p", "trafo_xl_nt-arxiv-typ_p", "trafo_xl_nt-trafo_xl", "trafo_xl_nt-trafo_xl-top_p", "trafo_xl_nt-trafo_xl-typ_p", "trafo_xl_nt-trafo_xl_nt", "trafo_xl_nt-trafo_xl_nt-top_p", "trafo_xl_nt-trafo_xl_nt-typ_p", "trafo_xl_nt-wiki_nt", "trafo_xl_nt-wiki_nt-top_p", "trafo_xl_nt-wiki_nt-typ_p", "wiki_nt-arxiv", "wiki_nt-wiki_nt"]
TOPIC_MODELS = ["classic_lda", "neural_lda"]

def generate_var_tt_score_plots(data_path, tokenization, sample_size, num_topics):
    models = ["gpt2_nt-gpt2_nt", "gpt2_nt-wiki_nt", "gpt2_nt-arxiv"]
    merge_types = ["intersection", "union"]

    for merge_type in merge_types:
        if merge_type == "intersection":
            tt_title = "Variance TT-Score Graph for classic LDA Models"
            tt_subtitle = f"{sample_size} samples, {tokenization}, intersected dicts\ndifferent seed for each model pair, lower score means more similar"
            tt_y_label = "TT Score"
            mode = 'is'
        elif merge_type == "union":
            tt_title = "Variance TT-Score Graph for classic LDA Models"
            tt_subtitle = f"{sample_size} samples, {tokenization}, unionized dicts\ndifferent seed for each model pair, lower score means more similar"
            tt_y_label = "TT Score"
            mode = 'un'
        else:
            raise ValueError(f">> ERROR: Unknown merge_type: {merge_type}")

        plt.clf()
        fig, axes = plt.subplots()
        color1 = iter(cm.PiYG(np.linspace(0, 1, 19)))
        color2 = iter(cm.bwr(np.linspace(0, 1, 19)))
        color3 = iter(cm.PiYG(np.linspace(0, 1, 19)))
        for _ in range(9):
            next(color3)
        colors = iter([color1, color2, color3])

        for model in models:
            data_path_ = f"{data_path}{tokenization}/{sample_size}/{model}/"

            tt_score_file_name = f"{data_path_}tt_score.json"
            if os.path.isfile(tt_score_file_name):
                with open(tt_score_file_name, "r") as file:
                    tt_scores = json.load(file)
            else:
                raise ValueError(f">> ERROR: wrong tt_score_file_name: {tt_score_file_name}")

            color = next(colors)
            for key in tt_scores.keys():
                key_split = key.split("-")
                if key[-2] != "-" or merge_type != key_split[-2]:
                    continue

                c = next(color)
                axes.plot(num_topics, tt_scores[key], label=f"{key_split[1]} vs. {key_split[2]} #{key_split[-1]}", c=c)

        plt.suptitle(tt_title, fontsize=15)
        axes.set_title(tt_subtitle, fontsize=8, x=0.6)
        axes.set_xscale('log')
        plt.ylim((0, 1))
        axes.set_xlabel('Number of Topics')
        axes.set_ylabel(tt_y_label)
        axes.set_xticks(num_topics)
        axes.get_xaxis().set_major_formatter(ScalarFormatter())

        font = FontProperties()
        font.set_size('xx-small')
        axes.legend(bbox_to_anchor=(1.04, 1), loc="upper left", prop=font)
        fig.tight_layout()

        plt.savefig(f"{data_path}/evaluation/{tokenization}-{sample_size}-var-tt-{mode}.png", dpi=300)
        plt.close('all')


def generate_var_cv_score_plots(data_path, tokenization, sample_size, num_topics):
    models = ["gpt2_nt-gpt2_nt", "gpt2_nt-wiki_nt", "gpt2_nt-arxiv"]
    merge_types = ["intersection", "union"]

    for merge_type in merge_types:
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
                if merge_type == "intersection":
                    cv_title = f"Variance CV-Score Graph for the classic LDA Model"
                    cv_subtitle = f"{focus} ({model}), {sample_size} samples, {tokenization}, intersected dicts\ndifferent seed for each model pair, lower score means more similar"
                    cv_y_label = "C_v Score"
                    mode = 'is'
                elif merge_type == "union":
                    cv_title = f"Variance CV-Score Graph for the classic LDA Model"
                    cv_subtitle = f"{focus} ({model}), {sample_size} samples, {tokenization}, unionized dicts\ndifferent seed for each model pair, lower score means more similar"
                    cv_y_label = "C_v Score"
                    mode = 'un'
                else:
                    raise ValueError(f">> ERROR: Unknown merge_type: {merge_type}")

                plt.clf()
                fig, axes = plt.subplots()
                color = iter(cm.bwr(np.linspace(0, 1, 19)))

                for key in cv_scores.keys():
                    key_split = key.split("-")
                    if key[-2] != "-" or merge_type != key_split[-2] or focus != key_split[-3]:
                        continue

                    c = next(color)
                    axes.plot(num_topics, cv_scores[key], label=f"{focus} #{key_split[-1]}", c=c)

                plt.suptitle(cv_title, fontsize=15)
                axes.set_title(cv_subtitle, fontsize=8, x=0.6)
                axes.set_xscale('log')
                axes.set_xlabel('Number of Topics')
                plt.ylim((0, 1))
                axes.set_ylabel(cv_y_label)
                axes.set_xticks(num_topics)
                axes.get_xaxis().set_major_formatter(ScalarFormatter())

                font = FontProperties()
                font.set_size('xx-small')
                axes.legend(bbox_to_anchor=(1.04, 1), loc="upper left", prop=font)
                fig.tight_layout()

                plt.savefig(f"{data_path}/evaluation/{tokenization}-{sample_size}-var-cv-{model}-{focus}-{mode}.png", dpi=300)
                plt.close('all')


def generate_var_crossmodel_plots(data_path, tokenization, sample_size, num_topics):
    focus_ = "gpt2_nt"
    models = [x for x in MODELS if focus_ in x and (sample_size <= 10000 or "trafo_xl" not in x)]
    merge_types = ["intersection", "union"]
    topic_models = [x for x in TOPIC_MODELS if sample_size <= 10000 or "neural_lda" not in x]

    for merge_type in merge_types:
        for topic_model in topic_models:
            if merge_type == "intersection":
                cv_title = f"Cross Variance CV-Score Graph for the {topic_model.split('_')[0]} LDA Model"
                cv_subtitle = f"all {focus_} in {topic_model}, {sample_size} samples, {tokenization}, intersected dicts\ndifferent seed for each model pair, lower score means more similar"
                cv_y_label = "C_v Score"
                mode = 'is'
            elif merge_type == "union":
                cv_title = f"Cross Variance CV-Score Graph for the {topic_model.split('_')[0]} LDA Model"
                cv_subtitle = f"all {focus_} in {topic_model}, {sample_size} samples, {tokenization}, unionized dicts\ndifferent seed for each model pair, lower score means more similar"
                cv_y_label = "C_v Score"
                mode = 'un'
            else:
                raise ValueError(f">> ERROR: Unknown merge_type: {merge_type}")

            plt.clf()
            fig, axes = plt.subplots()
            legend = []
            legend_top_p = []
            legend_typ_p = []

            color_length = 6 if sample_size <= 100000 else 4

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
                focus = focus_ if model_split[0] != model_split[1] else focus_ + "_1"
                for key in cv_scores.keys():
                    key_split = key.split("-")
                    if topic_model != key_split[0] or merge_type != key_split[-1] or focus != key_split[-2]:
                        continue

                    if model_split[-1] == "top_p":
                        style = 'dotted'
                        label = f"{focus} of {'-'.join(model_split[:-1])}"
                        c = next(color_top_p)
                    elif model_split[-1] == "typ_p":
                        style = 'dashed'
                        label = f"{focus} of {'-'.join(model_split[:-1])}"
                        c = next(color_typ_p)
                    else:
                        style = 'solid'
                        label = f"{focus} of {model}"
                        c = next(color_norm)

                    line, = axes.plot(num_topics, cv_scores[key], label=label, c=c, linestyle=style, linewidth=2)

                    if model_split[-1] == "top_p":
                        legend_top_p.append(line)
                    elif model_split[-1] == "typ_p":
                        legend_typ_p.append(line)
                    else:
                        legend.append(line)

            plt.suptitle(cv_title, fontsize=15)
            axes.set_title(cv_subtitle, fontsize=8, x=0.6)
            axes.set_xscale('log')
            axes.set_xlabel('Number of Topics')
            plt.ylim((0, 1))
            axes.set_ylabel(cv_y_label)
            axes.set_xticks(num_topics)
            axes.get_xaxis().set_major_formatter(ScalarFormatter())

            font = FontProperties()
            font.set_size(7)

            first_legend = axes.legend(handlelength=2, handles=legend, bbox_to_anchor=(1.04, 1), loc="upper left",
                                       prop=font, title='Normal sampling')
            axes.add_artist(first_legend)
            second_legend = axes.legend(handlelength=2, handles=legend_top_p, bbox_to_anchor=(1.04, 0.65),
                                        loc="upper left", prop=font, title='Top_p sampling')
            axes.add_artist(second_legend)
            third_legend = axes.legend(handlelength=2, handles=legend_typ_p, bbox_to_anchor=(1.04, 0.3),
                                       loc="upper left", prop=font, title='Typ_p sampling')

            fig.tight_layout()
            plt.savefig(
                f"{data_path}/evaluation/{tokenization}-{sample_size}-crossvar-cv-{topic_model}-{focus_}-{mode}.png",
                dpi=300)
            plt.close('all')


def generate_score_plots(data_folder_path):
    tokenizations = ["Trigrams+Bigrams+Unigrams"]#, "Bigrams+Unigrams", "Unigrams"]
    sample_sizes = [10000, 100000]
    num_topics = [2, 3, 5, 10, 20, 50, 100]

    if data_folder_path[-1] != '/':
        data_folder_path += '/'

    for tokenization in tokenizations:
        for sample_size in sample_sizes:
            #generate_var_tt_score_plots(data_folder_path, tokenization, sample_size, num_topics)
            #generate_var_cv_score_plots(data_folder_path, tokenization, sample_size, num_topics)
            generate_var_crossmodel_plots(data_folder_path, tokenization, sample_size, num_topics)
            #generate_coherence_plots(data_folder_path)




def _generate_score_plot():
    for case in [1, 2, 3, 4]:
        if case == 1:
            score_file_path = "./data/score_by_top_topic.json"
            title = "'Score by Top Topic'-Topic Graph for LDA Models"
            subtitle = "(intersected dicts, lower score means more similar)"
            y_label = "Score by Top Topic"
            mode = 'is'
        elif case == 2:
            score_file_path = "./data/score_by_top_topic.json"
            title = "'Score by Top Topic'-Topic Graph for LDA Models"
            subtitle = "(unionized dicts, lower score means more similar)"
            y_label = "Score by Top Topic"
            mode = 'un'
        elif case == 3:
            score_file_path = "./data/score_by_topic_probability_values.json"
            title = "'Score by Topic Prob.'-Topic Graph for LDA Models"
            subtitle = "(intersected dicts, lower score means more similar)"
            y_label = "Score by Topic Probability"
            mode = 'is'
        elif case == 4:
            score_file_path = "./data/score_by_topic_probability_values.json"
            title = "'Score by Topic Prob.'-Topic Graph for LDA Models"
            subtitle = "(unionized dicts, lower score means more similar)"
            y_label = "Score by Topic Probability"
            mode = 'un'
        else:
            print("ERROR")
            return
        if os.path.isfile(score_file_path):
            with open(score_file_path, 'r') as file:
                score_values = json.load(file)

        names = list(score_values.keys())
        values = list(score_values.values())
        topics = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]

        color = iter(cm.tab10(np.linspace(0, 1, 10)))
        plt.clf()
        legend = []
        legend_typ = []
        legend_top = []
        fig, axes = plt.subplots()
        c = next(color)
        for idx, name in enumerate(names):
            if mode not in name:
                continue
            style = 'solid'
            label = name.split('-')[:-1]
            if "typ_p" in name:
                style = 'dashed'
                del label[-1]
            elif "top_p" in name:
                style = 'dotted'
                del label[-1]
            else:
                c = next(color)
            label = " - ".join(label)
            line, = axes.plot(topics, values[idx], label=label, c=c, linestyle=style, linewidth=2)
            if "typ_p" in name:
                legend_typ.append(line)
            elif "top_p" in name:
                legend_top.append(line)
            else:
                legend.append(line)

        plt.ylim(0, 1)
        plt.suptitle(title, fontsize=15)
        axes.set_title(subtitle, fontsize=8, x=0.6)
        axes.set_xscale('log')
        axes.set_xlabel('Number of Topics')
        axes.set_ylabel(y_label)
        axes.set_xticks(topics)
        axes.get_xaxis().set_major_formatter(ScalarFormatter())

        font = FontProperties()
        font.set_size(8)

        first_legend = axes.legend(handlelength=2, handles=legend, bbox_to_anchor=(1.04, 1), loc="upper left", prop=font, title='Normal sampling')
        axes.add_artist(first_legend)
        second_legend = axes.legend(handlelength=2, handles=legend_top, bbox_to_anchor=(1.04, 0.58), loc="upper left", prop=font, title='Top_p sampling')
        axes.add_artist(second_legend)
        third_legend = axes.legend(handlelength=2, handles=legend_typ, bbox_to_anchor=(1.04, 0.35), loc="upper left", prop=font, title='Typ_p sampling')

        fig.tight_layout()
        plt.savefig(f"{score_file_path[:-5]}_{mode}.png", dpi=300)
        plt.close('all')


def _generate_coherence_plot():
    for case in [1, 2]:
        if case == 1:
            score_file_path = "./data/score_by_topic_coherence.json"
            title = "Topic Coherence Score (C_v) for LDA Models"
            subtitle = "(intersected dicts, higher score means more semantic similarity between high scoring words)"
            y_label = "C_v Score"
            mode = 'is'
        elif case == 2:
            score_file_path = "./data/score_by_topic_coherence.json"
            title = "Topic Coherence Score (C_v) for LDA Models"
            subtitle = "(unionized dicts, higher score means more semantic similarity between high scoring words)"
            y_label = "C_v Score"
            mode = 'un'
        else:
            print("ERROR")
            return
        if os.path.isfile(score_file_path):
            with open(score_file_path, 'r') as file:
                score_values = json.load(file)

        names = list(score_values.keys())
        values = list(score_values.values())
        topics = [2, 5, 10, 20, 50, 100]

        color = iter(cm.Dark2(np.linspace(0, 1, 8)))
        plt.clf()
        legend = []
        legend_typ = []
        legend_top = []
        fig, axes = plt.subplots()
        c = next(color)
        for idx, name in enumerate(names):
            if mode not in name:
                continue
            style = 'solid'
            label = name.split('-')[:-1]
            if "typ_p" in name:
                style = 'dashed'
                del label[-1]
            elif "top_p" in name:
                style = 'dotted'
                del label[-1]
            else:
                c = next(color)
            label = " - ".join(label)
            line, = axes.plot(topics, values[idx], label=label, c=c, linestyle=style, linewidth=2)
            if "typ_p" in name:
                legend_typ.append(line)
            elif "top_p" in name:
                legend_top.append(line)
            else:
                legend.append(line)

        plt.ylim(0, 1)
        plt.suptitle(title, fontsize=15)
        axes.set_title(subtitle, fontsize=8, x=0.6)
        axes.set_xscale('log')
        axes.set_xlabel('Number of Topics')
        axes.set_ylabel(y_label)
        axes.set_xticks(topics)
        axes.get_xaxis().set_major_formatter(ScalarFormatter())

        font = FontProperties()
        font.set_size(8)

        first_legend = axes.legend(handlelength=2, handles=legend, bbox_to_anchor=(1.04, 1), loc="upper left", prop=font, title='Normal sampling')
        axes.add_artist(first_legend)
        second_legend = axes.legend(handlelength=2, handles=legend_top, bbox_to_anchor=(1.04, 0.75), loc="upper left", prop=font, title='Top_p sampling')
        axes.add_artist(second_legend)
        third_legend = axes.legend(handlelength=2, handles=legend_typ, bbox_to_anchor=(1.04, 0.56), loc="upper left", prop=font, title='Typ_p sampling')

        fig.tight_layout()
        plt.savefig(f"{score_file_path[:-5]}_{mode}_coherence.png", dpi=300)
        plt.close('all')


def _generate_plot_var_diff():
    for case in [1, 2, 3, 4]:
        if case == 1:
            score_file_path = "./data/score_by_top_topic_var_diff.json"
            title = "Variance TT-Score Graph for LDA Models"
            subtitle = f"(intersected dicts, different seed for sampling corpus, lower score means more similar)"
            y_label = "Score by Top Topic"
            mode = 'is'
        elif case == 2:
            score_file_path = "./data/score_by_top_topic_var_diff.json"
            title = "Variance TT-Score Graph for LDA Models"
            subtitle = f"(unionized dicts, different seed for sampling corpus, lower score means more similar)"
            y_label = "Score by Top Topic"
            mode = 'un'
        elif case == 3:
            score_file_path = "./data/score_by_topic_probability_values_var_diff.json"
            title = "Variance TP-Score Graph for LDA Models"
            subtitle = f"(intersected dicts, different seed for sampling corpus, lower score means more similar)"
            y_label = "Score by Topic Probability"
            mode = 'is'
        elif case == 4:
            score_file_path = "./data/score_by_topic_probability_values_var_diff.json"
            title = "Variance TP-Score Graph for LDA Models"
            subtitle = f"(unionized dicts, different seed for sampling corpus, lower score means more similar)"
            y_label = "Score by Topic Probability"
            mode = 'un'
        else:
            print("ERROR")
            return
        if os.path.isfile(score_file_path):
            with open(score_file_path, 'r') as file:
                score_values = json.load(file)

        names = list(score_values.keys())
        values = list(score_values.values())
        color = iter(cm.PiYG(np.linspace(0, 1, 13)))
        topics = [2, 5, 10]

        plt.clf()
        fig, axes = plt.subplots()
        for idx, name in enumerate(names):
            if mode not in name:
                continue
            for i in range(5):
                c = next(color)
                axes.plot(topics, values[idx][i], label="-".join(name.split('-')[1:])+f"{i}", c=c)

        plt.ylim(0, 1)
        plt.suptitle(title, fontsize=15)
        axes.set_title(subtitle, fontsize=8, x=0.6)
        axes.set_xscale('linear')
        axes.set_xlabel('Number of Topics')
        axes.set_ylabel(y_label)
        axes.set_xticks(topics)
        axes.get_xaxis().set_major_formatter(ScalarFormatter())

        font = FontProperties()
        font.set_size('xx-small')
        axes.legend(bbox_to_anchor=(1.04, 1), loc="upper left", prop=font)
        fig.tight_layout()
        plt.savefig(f"{score_file_path[:-5]}_{mode}_var_diff.png", dpi=300)
        plt.close('all')


def _generate_plot_var_same():
    for case in [1, 2, 3, 4]:
        if case == 1:
            score_file_path = "./data/score_by_top_topic_var_same.json"
            title = "Variance TT-Score Graph for LDA Models"
            subtitle = f"(intersected dicts, same seed for sampling corpus, lower score means more similar)"
            y_label = "Score by Top Topic"
            mode = 'is'
        elif case == 2:
            score_file_path = "./data/score_by_top_topic_var_same.json"
            title = "Variance TT-Score Graph for LDA Models"
            subtitle = f"(unionized dicts, same seed for sampling corpus, lower score means more similar)"
            y_label = "Score by Top Topic"
            mode = 'un'
        elif case == 3:
            score_file_path = "./data/score_by_topic_probability_values_var_same.json"
            title = "Variance TP-Score Graph for LDA Models"
            subtitle = f"(intersected dicts, same seed for sampling corpus, lower score means more similar)"
            y_label = "Score by Topic Probability"
            mode = 'is'
        elif case == 4:
            score_file_path = "./data/score_by_topic_probability_values_var_same.json"
            title = "Variance TP-Score Graph for LDA Models"
            subtitle = f"(unionized dicts, same seed for sampling corpus, lower score means more similar)"
            y_label = "Score by Topic Probability"
            mode = 'un'
        else:
            print("ERROR")
            return
        if os.path.isfile(score_file_path):
            with open(score_file_path, 'r') as file:
                score_values = json.load(file)

        names = list(score_values.keys())
        values = list(score_values.values())
        color = iter(cm.PiYG(np.linspace(0, 1, 19)))
        topics = [2, 5, 10]

        plt.clf()
        fig, axes = plt.subplots()
        for idx, name in enumerate(names):
            if mode not in name:
                continue
            for i in range(9):
                c = next(color)
                axes.plot(topics, values[idx][i], label="-".join(name.split('-')[1:])+f"{i}", c=c)

        plt.suptitle(title, fontsize=15)
        axes.set_title(subtitle, fontsize=8, x=0.6)
        axes.set_xscale('linear')
        axes.set_xlabel('Number of Topics')
        axes.set_ylabel(y_label)
        axes.set_xticks(topics)
        axes.get_xaxis().set_major_formatter(ScalarFormatter())

        font = FontProperties()
        font.set_size('xx-small')
        axes.legend(bbox_to_anchor=(1.04, 1), loc="upper left", prop=font)
        fig.tight_layout()
        plt.savefig(f"{score_file_path[:-5]}_{mode}_var_same.png", dpi=300)
        plt.close('all')


def main():
    generate_score_plots("./data")


if __name__ == '__main__':
    main()
