import gc
import json
import os
import random

import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import ScalarFormatter
from matplotlib.pyplot import cm


def generate_bjobs(topics, file_path, data_path, topic_models, models, models_copy, duals, sampling_methods, samples, combis, time="04:00", num_cpu=1, cpu_mem=1024, use_gpu=False, num_gpus=1, gpu_mem=8192, gpu_model=None, num_var=1, calc_var=False, calc_score=False, score_mode=None):
    if calc_score and score_mode is None:
        raise AssertionError(">> ERROR: calc_score needs score_mode not to be None")
    models_copy_old = models_copy.copy()
    if calc_var and num_var > 0:
        var_range = np.arange(1, num_var + 1).astype(str)
    else:
        var_range = [""]
    for var in var_range:
        var_log = var_cmd = ""
        if var != "":
            var_log = f"-{var}"
            var_cmd = f" {var}"
        for topic_model in topic_models:
            for dual in duals:
                for sampling_method in sampling_methods:
                    sampling_method_short = "" if sampling_method == "multinomial" else "-" + sampling_method
                    for first in models:
                        for second in models_copy:
                            if sampling_method != "multinomial" and first != "gpt2_nt" and second != "gpt2_nt":
                                continue
                            for topic in topics:
                                if topic == 1:
                                    topic_model_short = ""
                                else:
                                    topic_model_short = topic_model + "-"
                                for combi in combis:
                                    combi_short = "is" if combi == "intersection" else "un"
                                    if calc_score:
                                        for score in score_mode:
                                            if use_gpu:
                                                if gpu_model is None:
                                                    print(f"bsub -N -W {time} -n {num_cpu} -R \"rusage[mem={cpu_mem},ngpus_excl_p={num_gpus}]\" -R \"select[gpu_mtotal0>={gpu_mem}]\" -oo logs/{samples}/log-{score}-{first}-{second}{sampling_method_short}.txt \"python {file_path}score_lda.py {data_path} {score} {samples} {first}-{second}{sampling_method_short}\"")
                                                else:
                                                    print(f"bsub -N -W {time} -n {num_cpu} -R \"rusage[mem={cpu_mem},ngpus_excl_p={num_gpus}]\" -R \"select[gpu_model0=={gpu_model}]\" -oo logs/{samples}/log-{score}-{first}-{second}{sampling_method_short}.txt \"python {file_path}score_lda.py {data_path} {score} {samples} {first}-{second}{sampling_method_short}\"")
                                            else:
                                                print(f"bsub -N -W {time} -n {num_cpu} -R \"rusage[mem={cpu_mem}]\" -oo logs/{samples}/log-{score}-{first}-{second}{sampling_method_short}.txt \"python {file_path}score_lda.py {data_path} {score} {samples} {first}-{second}{sampling_method_short}\"")
                                    else:
                                        if use_gpu:
                                            if gpu_model is None:
                                                print(f"bsub -N -W {time} -n {num_cpu} -R \"rusage[mem={cpu_mem},ngpus_excl_p={num_gpus}]\" -R \"select[gpu_mtotal0>={gpu_mem}]\" -oo logs/{samples}/log-{topic_model_short}{first}-{second}-{combi_short}-{topic}{sampling_method_short}-{dual}{var_log}.txt \"python {file_path}train_lda.py {data_path} {topic_model} {first} {second} {dual} {sampling_method} {topic} {combi} {samples}{var_cmd}\"")
                                            else:
                                                print(f"bsub -N -W {time} -n {num_cpu} -R \"rusage[mem={cpu_mem},ngpus_excl_p={num_gpus}]\" -R \"select[gpu_model0=={gpu_model}]\" -oo logs/{samples}/log-{topic_model_short}{first}-{second}-{combi_short}-{topic}{sampling_method_short}-{dual}{var_log}.txt \"python {file_path}train_lda.py {data_path} {topic_model} {first} {second} {dual} {sampling_method} {topic} {combi} {samples}{var_cmd}\"")
                                        else:
                                            print(f"bsub -N -W {time} -n {num_cpu} -R \"rusage[mem={cpu_mem}]\" -oo logs/{samples}/log-{topic_model_short}{first}-{second}-{combi_short}-{topic}{sampling_method_short}-{dual}{var_log}.txt \"python {file_path}train_lda.py {data_path} {topic_model} {first} {second} {dual} {sampling_method} {topic} {combi} {samples}{var_cmd}\"")
                        models_copy.remove(first)
                    models_copy = models_copy_old.copy()


def generate_tokenize_bjobs(samples=10000):
    models = ["gpt2_nt", "gpt2", "trafo_xl_nt", "trafo_xl", "wiki_nt", "arxiv"]
    if samples > 10000:
        models.remove("trafo_xl_nt")
        models.remove("trafo_xl")

    models_copy = models.copy()

    generate_bjobs(
        topics=[1],
        file_path="/cluster/work/cotterell/knobelf/",
        data_path="/cluster/work/cotterell/knobelf/data/",
        topic_models=["classic_lda"],
        models=models,
        models_copy=models_copy,
        duals=["tokenization"],
        sampling_methods=["multinomial", "top_p", "typ_p"],
        samples=samples,
        combis=["intersection", "union"],
        time="24:00",
        num_cpu=8,
        cpu_mem=5120,
        use_gpu=False,
        num_gpus=1,
        gpu_mem=30000,
        gpu_model=None,
        num_var=10,
        calc_var=False
    )


def generate_classic_lda_bjobs(samples=10000):
    models = ["gpt2_nt", "gpt2", "trafo_xl_nt", "trafo_xl", "wiki_nt", "arxiv"]
    if samples > 10000:
        models.remove("trafo_xl_nt")
        models.remove("trafo_xl")

    models_copy = models.copy()

    generate_bjobs(
        topics=[2, 3, 5, 10, 20, 50, 100],
        file_path="/cluster/work/cotterell/knobelf/",
        data_path="/cluster/work/cotterell/knobelf/data/",
        topic_models=["classic_lda"],
        models=models,
        models_copy=models_copy,
        duals=["first", "second"],
        sampling_methods=["multinomial", "top_p", "typ_p"],
        samples=samples,
        combis=["intersection", "union"],
        time="24:00",
        num_cpu=4,
        cpu_mem=5120,
        use_gpu=False,
        num_gpus=1,
        gpu_mem=30000,
        gpu_model=None,
        num_var=10,
        calc_var=False
    )


def generate_classic_lda_variation_bjobs(samples=10000):
    generate_bjobs(
        topics=[2, 3, 5, 10, 20, 50, 100],
        file_path="/cluster/work/cotterell/knobelf/",
        data_path="/cluster/work/cotterell/knobelf/data/",
        topic_models=["classic_lda"],
        models=["gpt2_nt"],
        models_copy=["gpt2_nt", "wiki_nt", "arxiv"],
        duals=["first", "second"],
        sampling_methods=["multinomial"],
        samples=samples,
        combis=["intersection", "union"],
        time="24:00",
        num_cpu=4,
        cpu_mem=10240,
        use_gpu=False,
        num_gpus=1,
        gpu_mem=30000,
        gpu_model=None,
        num_var=9,
        calc_var=True
    )


def generate_neural_lda_bjobs(samples=10000):
    models = ["gpt2_nt", "gpt2", "trafo_xl_nt", "trafo_xl", "wiki_nt", "arxiv"]
    if samples > 10000:
        models.remove("trafo_xl_nt")
        models.remove("trafo_xl")
        print(">> ERROR: Gpus not able to handle sizes bigger than 10k. Octis is not able to use generators for data.")
        return

    models_copy = models.copy()

    generate_bjobs(
        topics=[2, 3, 5, 10, 20, 50, 100],
        file_path="/cluster/work/cotterell/knobelf/",
        data_path="/cluster/work/cotterell/knobelf/data/",
        topic_models=["neural_lda"],
        models=models,
        models_copy=models_copy,
        duals=["first", "second"],
        sampling_methods=["multinomial", "top_p", "typ_p"],
        samples=samples,
        combis=["intersection", "union"],
        time="4:00",
        num_cpu=4,
        cpu_mem=10240,
        use_gpu=True,
        num_gpus=1,
        gpu_mem=10000,
        gpu_model=None,     # NVIDIAGeForceRTX3090, NVIDIAGeForceGTX1080, NVIDIAGeForceGTX1080Ti, NVIDIAGeForceRTX2080Ti, NVIDIATITANRTX, QuadroRTX6000, TeslaV100_SXM2_32GB, NVIDIAA100_PCIE_40GB
        num_var=10,
        calc_var=False
    )


def generate_score_bjobs(samples=10000):
    models = ["gpt2_nt", "gpt2", "trafo_xl_nt", "trafo_xl", "wiki_nt", "arxiv"]
    if samples > 10000:
        models.remove("trafo_xl_nt")
        models.remove("trafo_xl")

    models_copy = models.copy()

    generate_bjobs(
        topics=[1],
        file_path="/cluster/work/cotterell/knobelf/",
        data_path="/cluster/work/cotterell/knobelf/data/",
        topic_models=["classic_lda"],
        models=models,
        models_copy=models_copy,
        duals=["first"],
        sampling_methods=["multinomial", "top_p", "typ_p"],
        samples=samples,
        combis=["intersection"],
        time="24:00",
        num_cpu=4,
        cpu_mem=10240,
        use_gpu=True,
        num_gpus=1,
        gpu_mem=10000,
        calc_score=True,
        score_mode=["cv", "tt", "tp", "img"]
    )


def assemble_score():
    pass


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
    gc.collect()
    torch.cuda.empty_cache()

    #generate_tokenize_bjobs(100000)
    #generate_tokenize_bjobs(10000)
    
    #generate_classic_lda_bjobs(100000)
    #generate_classic_lda_bjobs(10000)
    
    #generate_classic_lda_variation_bjobs(100000)
    #generate_classic_lda_variation_bjobs(10000)

    #generate_neural_lda_bjobs(10000)

    generate_score_bjobs(10000)
    generate_score_bjobs(100000)

if __name__ == '__main__':
    main()
