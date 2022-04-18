import numpy as np


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
                            if sampling_method != "multinomial" and first not in ["gpt2_nt", "trafo_xl_nt"] and second not in ["gpt2_nt", "trafo_xl_nt"]:
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
                                                    print(f"bsub -W {time} -n {num_cpu} -R \"rusage[mem={cpu_mem},ngpus_excl_p={num_gpus}]\" -R \"select[gpu_mtotal0>={gpu_mem}]\" -oo logs/{samples}/log-{score}-{first}-{second}{sampling_method_short}.txt \"python {file_path}score_lda.py {data_path} {score} {samples} {first}-{second}{sampling_method_short}\"")
                                                else:
                                                    print(f"bsub -W {time} -n {num_cpu} -R \"rusage[mem={cpu_mem},ngpus_excl_p={num_gpus}]\" -R \"select[gpu_model0=={gpu_model}]\" -oo logs/{samples}/log-{score}-{first}-{second}{sampling_method_short}.txt \"python {file_path}score_lda.py {data_path} {score} {samples} {first}-{second}{sampling_method_short}\"")
                                            else:
                                                print(f"bsub -W {time} -n {num_cpu} -R \"rusage[mem={cpu_mem}]\" -oo logs/{samples}/log-{score}-{first}-{second}{sampling_method_short}.txt \"python {file_path}score_lda.py {data_path} {score} {samples} {first}-{second}{sampling_method_short}\"")
                                    else:
                                        if use_gpu:
                                            if gpu_model is None:
                                                print(f"bsub -W {time} -n {num_cpu} -R \"rusage[mem={cpu_mem},ngpus_excl_p={num_gpus}]\" -R \"select[gpu_mtotal0>={gpu_mem}]\" -oo logs/{samples}/log-{topic_model_short}{first}-{second}-{combi_short}-{topic}{sampling_method_short}-{dual}{var_log}.txt \"python {file_path}train_lda.py {data_path} {topic_model} {first} {second} {dual} {sampling_method} {topic} {combi} {samples}{var_cmd}\"")
                                            else:
                                                print(f"bsub -W {time} -n {num_cpu} -R \"rusage[mem={cpu_mem},ngpus_excl_p={num_gpus}]\" -R \"select[gpu_model0=={gpu_model}]\" -oo logs/{samples}/log-{topic_model_short}{first}-{second}-{combi_short}-{topic}{sampling_method_short}-{dual}{var_log}.txt \"python {file_path}train_lda.py {data_path} {topic_model} {first} {second} {dual} {sampling_method} {topic} {combi} {samples}{var_cmd}\"")
                                        else:
                                            print(f"bsub -W {time} -n {num_cpu} -R \"rusage[mem={cpu_mem}]\" -oo logs/{samples}/log-{topic_model_short}{first}-{second}-{combi_short}-{topic}{sampling_method_short}-{dual}{var_log}.txt \"python {file_path}train_lda.py {data_path} {topic_model} {first} {second} {dual} {sampling_method} {topic} {combi} {samples}{var_cmd}\"")
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


def main():
    #generate_tokenize_bjobs(100000)
    #generate_tokenize_bjobs(10000)
    
    #generate_classic_lda_bjobs(100000)
    #generate_classic_lda_bjobs(10000)
    
    #generate_classic_lda_variation_bjobs(100000)
    #generate_classic_lda_variation_bjobs(10000)

    #generate_neural_lda_bjobs(10000)

    #generate_score_bjobs(10000)
    #generate_score_bjobs(100000)
    return


if __name__ == '__main__':
    main()
