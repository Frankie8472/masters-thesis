import gc
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import ScalarFormatter
from matplotlib.pyplot import cm


def generate_var_plots(data_path, tokenization, sample_size, num_topics):
    models = ["gpt2_nt-gpt2_nt", "gpt2_nt-wiki_nt", "gpt2_nt-arxiv"]
    merge_types = ["intersection", "union"]

    for model in models:
        data_path_ = f"{data_path}{tokenization}/{sample_size}/{model}/"

        cv_score_file_name = f"{data_path_}cv_score.json"
        if os.path.isfile(cv_score_file_name):
            with open(cv_score_file_name, "r") as file:
                cv_scores = json.load(file)
        else:
            raise ValueError(f">> ERROR: wrong cv_score_file_name: {cv_score_file_name}")

        tt_score_file_name = f"{data_path_}tt_score.json"
        if os.path.isfile(tt_score_file_name):
            with open(tt_score_file_name, "r") as file:
                tt_scores = json.load(file)
        else:
            raise ValueError(f">> ERROR: wrong tt_score_file_name: {tt_score_file_name}")

        for merge_type in merge_types:
            if merge_type == "intersection":
                tt_title = "Variance TT-Score Graph for classic LDA Models"
                tt_subtitle = f"(intersected dicts, different seed for each model pair, lower score means more similar)"
                tt_y_label = "tt score"
                cv_title = "Variance CV-Score Graph for classic LDA Models"
                cv_subtitle = f"(intersected dicts, different seed for each model, higher means better correlation with human judgement)"
                cv_y_label = "c_v score"
                mode = 'is'
            elif merge_type == "union":
                tt_title = "Variance TT-Score Graph for classic LDA Models"
                tt_subtitle = f"(unionized dicts, different seed for each model pair, lower score means more similar)"
                tt_y_label = "tt score"
                cv_title = "Variance CV-Score Graph for classic LDA Models"
                cv_subtitle = f"(unionized dicts, different seed for each model, higher means better correlation with human judgement)"
                cv_y_label = "c_v score"
                mode = 'un'
            else:
                raise ValueError(f">> ERROR: Unknown merge_type: {merge_type}")

            # Plot TT Score
            plt.clf()
            fig, axes = plt.subplots()
            color = iter(cm.PiYG(np.linspace(0, 1, 19)))        # TODO

            for key in tt_scores.keys():
                key_split = key.split("-")
                if key[-2] != "-" or merge_type != key_split[-2]:
                    continue
                for i, _ in enumerate(num_topics):
                    c = next(color)
                    axes.plot(num_topics, tt_scores[key], label=f"{key_split[1]} vs. {key_split[2]} #{key_split[-1]}", c=c)

            plt.suptitle(tt_title, fontsize=15)
            axes.set_title(tt_subtitle, fontsize=8, x=0.6)
            axes.set_xscale('linear')
            axes.set_xlabel('Number of Topics')
            axes.set_ylabel(tt_y_label)
            axes.set_xticks(num_topics)
            axes.get_xaxis().set_major_formatter(ScalarFormatter())

            font = FontProperties()
            font.set_size('xx-small')
            axes.legend(bbox_to_anchor=(1.04, 1), loc="upper left", prop=font)
            fig.tight_layout()

            plt.savefig(f"{data_path}/evaluation/{tokenization}_{sample_size}_var_cv.png", dpi=300)
            plt.close('all')



def generate_coherence_plots(data_path):
    pass


def generate_score_plots(data_folder_path):
    tokenizations = ["Trigrams+Bigrams+Unigrams", "Bigrams+Unigrams", "Unigrams"]
    sample_sizes = [10000, 100000]
    num_topics = [2, 3, 5, 10, 20, 50, 100]

    if data_folder_path[-1] != '/':
        data_folder_path += '/'

    for tokenization in tokenizations:
        for sample_size in sample_sizes:
            generate_var_plots(data_folder_path, tokenization, sample_size, num_topics)
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

    generate_score_plots()


if __name__ == '__main__':
    main()
