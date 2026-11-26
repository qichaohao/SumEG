import csv
import os
import json
import time
import argparse
import requests
from tqdm import tqdm
import nltk
from rouge_score import rouge_scorer, scoring


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("yes", "true", "t", "1", "y")


API_URL = "https://api.siliconflow.cn/v1/chat/completions"
MAX_RETRIES = 20
RETRY_BACKOFF = 1.0
SLEEP_BETWEEN_REQUESTS = 0.5
TEMP_INPUT_SUFFIX = ".tmp"


def fuse_scores(sample, alpha, beta):
    u_scores = sample.get("u_scores", {})
    f_scores = sample.get("f_scores", {})
    w_scores = sample.get("w_scores", {})
    fused = {}
    for r in sample.get("relations", []):
        rid = str(r.get("r_id"))
        su = float(u_scores.get(rid, 0.0))
        if not r.get("is_empty", False):
            sf = float(f_scores.get(rid, 0.0))
            sw = float(w_scores.get(rid, 0.0))
        else:
            sf = 0.0
            sw = 0.0
        fused[rid] = round(alpha * su + sf + beta * sw, 6)
    return fused


def build_prompt_relations(sample, top_relations):
    text = sample.get("text", "")
    lines = []
    for r in top_relations:
        sp = r.get("speaker", "")
        ac = r.get("action", "")
        ct = r.get("content", "")
        lines.append(f"Speaker: {sp}; Action: {ac}; Content: {ct}")
    rel_block = "\n".join(lines)
    prompt = (
        "You are a dialogue summarization assistant.\n"
        "Please generate a concise one-sentence summary that is grounded in the following important relation triples.\n"
        "The relation triples define the core factual content that the summary should be based on.\n"
        "The dialogue is provided as background knowledge and may be used to improve fluency or resolve references,\n"
        "but the summary should not introduce factual content that goes beyond or contradicts the relation triples.\n"
        "\n"
        "Important relation triples:\n"
        f"{rel_block}\n"
        "\n"
        "Dialogue (background knowledge only):\n"
        f"{text}\n"
        "\n"
        "Summary:"
    )
    return prompt


def build_prompt_dialogue(sample):
    text = sample.get("text", "")
    prompt = (
        "You are a dialogue summarization assistant.\n"
        "Given the following conversation, write a concise and accurate summary.\n"
        "Conversation:\n"
        f"{text}\n"
        "Summary:"
    )
    return prompt


def compute_rouge_from_paths(pred_path, refs_path):
    refs = []
    with open(refs_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            refs.append(data["summary"])
    preds = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            preds.append(line.strip())
    assert len(preds) == len(refs)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()
    def postprocess_text(preds, labels):
        preds = ["\n".join(nltk.sent_tokenize(p.strip())) for p in preds]
        labels = ["\n".join(nltk.sent_tokenize(l.strip())) for l in labels]
        return preds, labels
    preds, refs = postprocess_text(preds, refs)
    for pred, ref in zip(preds, refs):
        scores = scorer.score(pred, ref)
        aggregator.add_scores(scores)
    result = aggregator.aggregate()
    print("ROUGE-1: {:.4f}".format(result["rouge1"].mid.fmeasure * 100))
    print("ROUGE-2: {:.4f}".format(result["rouge2"].mid.fmeasure * 100))
    print("ROUGE-L: {:.4f}".format(result["rougeL"].mid.fmeasure * 100))
    return result


def call_api(api_url, api_key, model_name, prompt, max_tokens=128, temperature=0.7, top_p=0.7, top_k=50, frequency_penalty=0.5, timeout=60, seed=None):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "frequency_penalty": frequency_penalty,
        "n": 1
    }
    if seed is not None and seed >= 0:
        payload["seed"] = int(seed)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        resp = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
    except Exception:
        return None
    if resp.status_code == 200:
        try:
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            return None
    return None


def generate_via_api(api_url, api_key, model_name, prompt, max_tokens=128, temperature=0.7, top_p=0.7, top_k=50, frequency_penalty=0.5, base_backoff=RETRY_BACKOFF, sleep_between=SLEEP_BETWEEN_REQUESTS, seed=None):
    backoff = base_backoff
    attempt = 0
    while True:
        t = 0.0 if (seed is not None and seed >= 0) else temperature
        p = 1.0 if (seed is not None and seed >= 0) else top_p
        fp = 0.0 if (seed is not None and seed >= 0) else frequency_penalty
        result = call_api(api_url, api_key, model_name, prompt, max_tokens=max_tokens, temperature=t, top_p=p, top_k=top_k, frequency_penalty=fp, seed=seed)
        if result is not None:
            text = result
            if "Summary:" in text:
                return text.split("Summary:", 1)[1].strip()
            time.sleep(sleep_between)
            return text.strip()
        time.sleep(backoff)
        backoff = min(backoff * 2.0, 60.0)
        attempt += 1
        if MAX_RETRIES > 0 and attempt >= MAX_RETRIES:
            backoff = RETRY_BACKOFF


def run(scores_path, output_txt, model_name, use_relations, k_top, alpha, beta, run_rouge, refs_path, api_url, base_backoff, sleep_between, api_key, temperature, top_p, top_k, frequency_penalty, seed):
    key = api_key or os.environ.get("SILICONFLOW_API_KEY", "")
    if not key:
        raise RuntimeError("API key not provided. Set --api_key or environment SILICONFLOW_API_KEY.")
    endpoint = api_url or API_URL
    print(f"使用API: {endpoint}，模型: {model_name}，seed: {seed if (seed is not None and seed >= 0) else 'None'}")
    base_dir = os.path.join("data", "dialogsum")
    os.makedirs(base_dir, exist_ok=True)
    def _basename(name: str) -> str:
        name = name.replace("\\", "/")
        return name.split("/")[-1]
    if not output_txt:
        # 从scores_path中提取切片编号, e.g., "test_scores_5.jsonl" -> "slice5"
        try:
            slice_num = os.path.basename(scores_path).split('_')[-1].split('.')[0]
            slice_info = f"slice{slice_num}" if slice_num.isdigit() else "noslice"
        except IndexError:
            slice_info = "noslice" # Fallback if naming is unexpected

        if use_relations:
            fname = f"{slice_info}-{_basename(model_name)}-{alpha}-{beta}-{k_top}-predictions.txt"
        else:
            fname = f"{_basename(model_name)}-predictions.txt"
        output_txt = os.path.join(base_dir, fname)
    print(f"输出文件: {output_txt}")
    with open(scores_path, "r", encoding="utf-8") as fin:
        lines = [ln.rstrip("\n") for ln in fin if ln.strip()]
    bar = tqdm(total=len(lines), desc="Samples", leave=True, ncols=80)
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as fout:
        for line in lines:
            sample = json.loads(line)
            if use_relations:
                fused = fuse_scores(sample, alpha, beta)
                rs = sample.get("relations", [])
                ranked = sorted(
                    [r for r in rs if not r.get("is_empty", False)],
                    key=lambda r: fused.get(str(r.get("r_id")), 0.0),
                    reverse=True,
                )
                top_rel = ranked[:k_top] if k_top > 0 else ranked
                prompt = build_prompt_relations(sample, top_rel)
                bar.set_postfix_str("relations")
            else:
                prompt = build_prompt_dialogue(sample)
                bar.set_postfix_str("dialogue")
            summary = generate_via_api(endpoint, key, model_name, prompt, max_tokens=128, temperature=temperature, top_p=top_p, top_k=top_k, frequency_penalty=frequency_penalty, base_backoff=base_backoff, sleep_between=sleep_between, seed=seed)
            fout.write(summary.replace("\r", " ").replace("\n", " ").strip() + "\n")
            fout.flush()
            os.fsync(fout.fileno())
            bar.update(1)
            bar.refresh()
    bar.close()
    if run_rouge:
        result = compute_rouge_from_paths(output_txt, refs_path)
        log_path = os.path.join("data", "dialogsum", "rouge_results.csv")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # Extract slice number from scores_path
        try:
            slice_num = os.path.basename(scores_path).split('_')[-1].split('.')[0]
        except IndexError:
            slice_num = "N"

        r1 = result["rouge1"].mid.fmeasure * 100.0
        r2 = result["rouge2"].mid.fmeasure * 100.0
        rL = result["rougeL"].mid.fmeasure * 100.0

        header = ["model_name", "is_baseline", "slice_num", "alpha", "beta", "k_top", "rouge1", "rouge2", "rougeL"]
        data_row = [
            model_name,
            not use_relations,
            slice_num,
            alpha if use_relations else "N/A",
            beta if use_relations else "N/A",
            k_top if use_relations else "N/A",
            f"{r1:.4f}",
            f"{r2:.4f}",
            f"{rL:.4f}"
        ]

        file_exists = os.path.isfile(log_path)
        with open(log_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(data_row)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scores_path", type=str, default="")
    p.add_argument("--output_txt", type=str, default="")
    p.add_argument("--model_name", type=str, default="")
    p.add_argument("--use_relations", type=str2bool, default=False)
    p.add_argument("--k_top", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.4)
    p.add_argument("--beta", type=float, default=0.3)
    p.add_argument("--run_rouge", type=str2bool, default=True)
    p.add_argument("--refs_path", type=str, default="data/dialogsum/test.json")
    p.add_argument("--api_url", type=str, default=API_URL)
    p.add_argument("--api_key", type=str, default="sk-xhigfntcupcfpcumberfhwbgxjtbbyugsszyvnknebtinvbg")
    p.add_argument("--base_backoff", type=float, default=RETRY_BACKOFF)
    p.add_argument("--sleep_between", type=float, default=SLEEP_BETWEEN_REQUESTS)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--frequency_penalty", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=45)
    return p.parse_args()



if __name__ == "__main__":
    args = parse_args()

    # --- 实验配置中心 ---
    # 格式: (slice_config, model_name, k_top_range, use_relations_flag)
    # slice_config: 0 表示基准模式, range(1, 7) 表示遍历1-6号切片.
    # k_top_range: k_top的取值范围, e.g., range(1, 16) 表示 1 到 15.
    # use_relations_flag: True 表示关系模式, False 表示基准模式.
    experiments = [
        # 实验一: 运行基准模型, 遍历1-6号切片
        #(range(1, 2), "THUDM/GLM-4-32B-0414", None, False),
        (range(1, 2), "deepseek-ai/DeepSeek-R1", None, False),

        # 实验二: 运行关系模型, 遍历1-6号切片, k_top从1到15
        #(range(1, 9), "Qwen/Qwen3-8B", range(1, 16), True),
    ]

    output_base_dir = os.path.join("data", "dialogsum", "silicon")

    def _basename(name: str) -> str:
        name = name.replace("\\", "/")
        return name.split("/")[-1]

    for slice_config, model_name, k_top_range, use_relations_flag in experiments:
        args.model_name = model_name
        args.use_relations = use_relations_flag
        model_basename = _basename(model_name)


        # 模式一: 基准运行
        if not use_relations_flag:
            for slice_num in slice_config:
                print(f"\n========== [基准模式] 开始执行: slice={slice_num}, model={model_name} ==========")
                args.scores_path = f"data/dialogsum/scores/test_scores_{slice_num}.jsonl"
                
                # 文件直接保存在 silicon 目录下
                fname = f"{model_basename}-predictions.txt"
                args.output_txt = os.path.join(output_base_dir, fname)
                
                run(**vars(args))
                print(f"========== [基准模式] 完成: slice={slice_num}, model={model_name} ==========\n")
            continue

        # 模式二: 关系感知批量运行
        if use_relations_flag:
            for slice_num in slice_config:
                for k in k_top_range:
                    args.k_top = k
                    print(f"\n========== [关系模式] 开始执行: slice={slice_num}, k_top={k}, model={model_name} ==========")
                    args.scores_path = f"data/dialogsum/scores/test_scores_{slice_num}.jsonl"

                    # 创建模型和切片子目录
                    target_dir = os.path.join(output_base_dir, model_basename, str(slice_num))
                    fname = f"{model_basename}-{args.alpha}-{args.beta}-{k}-predictions.txt"
                    args.output_txt = os.path.join(target_dir, fname)

                    run(**vars(args))
                    print(f"========== [关系模式] 完成: slice={slice_num}, k_top={k} ==========\n")
