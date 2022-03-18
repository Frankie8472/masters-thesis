import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import ScalarFormatter
from matplotlib.pyplot import cm
from transformers import set_seed


def generate_bjobs():
    topics = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]
    models = [23, 24, 25, 26]   #[1, 2, 7, 8]
    samples = [1]     # [1, 2, 3, 4, 5]
    for model in models:
        for topic in topics:
            for union in [0, 1]:
                for i in samples:
                    combi = "un" if union else "is"
                    if model == 1:
                        name = f"wiki_nt-gpt2_nt-{combi}-{topic}.txt"
                    elif model == 2:
                        name = f"gpt2_nt-wiki_nt-{combi}-{topic}.txt"
                    elif model == 3:
                        name = f"gpt2_nt_1-gpt2_nt_2-{combi}-{topic}.txt"
                    elif model == 4:
                        name = f"gpt2_nt_2-gpt2_nt_1-{combi}-{topic}.txt"
                    elif model == 5:
                        name = f"gpt2_1-gpt2_2-{combi}-{topic}.txt"
                    elif model == 6:
                        name = f"gpt2_2-gpt2_1-{combi}-{topic}.txt"
                    elif model == 7:
                        name = f"wiki_nt-arxiv-{combi}-{topic}.txt"
                    elif model == 8:
                        name = f"arxiv-wiki_nt-{combi}-{topic}.txt"
                    elif model == 9:
                        name = f"gpt2_nt-arxiv-{combi}-{topic}.txt"
                    elif model == 10:
                        name = f"arxiv-gpt2_nt-{combi}-{topic}.txt"
                    elif model == 11:
                        name = f"gpt2_nt-arxiv-top_p-{combi}-{topic}.txt"
                    elif model == 12:
                        name = f"arxiv-gpt2_nt-top_p-{combi}-{topic}.txt"
                    elif model == 13:
                        name = f"gpt2_nt-wiki_top_p-{combi}-{topic}.txt"
                    elif model == 14:
                        name = f"wiki-gpt2_nt-top_p-{combi}-{topic}.txt"
                    elif model == 15:
                        name = f"gpt2_nt_1-gpt2_nt_2-top_p-{combi}-{topic}.txt"
                    elif model == 16:
                        name = f"gpt2_np_2-gpt2_nt_1-top_p-{combi}-{topic}.txt"
                    elif model == 17:
                        name = f"gpt2_nt-arxiv-typ_p-{combi}-{topic}.txt"
                    elif model == 18:
                        name = f"arxiv-gpt2_nt-typ_p-{combi}-{topic}.txt"
                    elif model == 19:
                        name = f"gpt2_nt-wiki_typ_p-{combi}-{topic}.txt"
                    elif model == 20:
                        name = f"wiki-gpt2_nt-typ_p-{combi}-{topic}.txt"
                    elif model == 21:
                        name = f"gpt2_nt_1-gpt2_nt_2-typ_p-{combi}-{topic}.txt"
                    elif model == 22:
                        name = f"gpt2_nt_2-gpt2_nt_1-typ_p-{combi}-{topic}.txt"
                    elif model == 23:
                        name = f"gpt2_nt-gpt2-{combi}-{topic}.txt"
                    elif model == 24:
                        name = f"gpt2-gpt2_nt-{combi}-{topic}.txt"
                    elif model == 25:
                        name = f"gpt2-wiki_nt-{combi}-{topic}.txt"
                    elif model == 26:
                        name = f"wiki_nt-gpt2-{combi}-{topic}.txt"
                    else:
                        print("ERROR")
                        return
                    if len(samples) > 1:
                        name = f"{name[:-4]}-{i}{name[-4:]}"
                        print(f"bsub -N -W 24:00 -n 48 -R \"rusage[mem=2666]\" -o logs/log-{name} \"python /cluster/work/cotterell/knobelf/train_lda.py {union} {model} {topic} {i}\"")

                    else:
                        print(f"bsub -N -W 24:00 -n 48 -R \"rusage[mem=2666]\" -o logs/log-{name} \"python /cluster/work/cotterell/knobelf/train_lda.py {union} {model} {topic}\"")


def generate_score_plot():
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


def generate_coherence_plot():
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


def generate_plot_var_diff():
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


def generate_plot_var_same():
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
    set_seed(42)
    generate_score_plot()


if __name__ == '__main__':
    main()
