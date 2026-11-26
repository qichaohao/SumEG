# 第183行更换数据集
import os
import json
import math
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
def cosine(a, b):
    na = torch.norm(a)
    nb = torch.norm(b)
    if na.item() == 0 or nb.item() == 0:
        return 0.0
    return torch.dot(a, b).item() / (na.item() * nb.item())

def load_encoder(model_name):
    local_only = os.path.isdir(model_name) or os.path.exists(os.path.join(model_name, "config.json"))
    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only, use_auth_token=(None if local_only else token))
    model = AutoModel.from_pretrained(model_name, local_files_only=local_only, use_auth_token=(None if local_only else token))
    model.eval()
    return tokenizer, model

def encode_text(text, tokenizer, model, device, max_length=512):
    if not text:
        return torch.zeros(model.config.hidden_size, device=device)
    max_pos = getattr(model.config, "max_position_embeddings", 512)
    effective_max = min(max_length, max_pos - 2 if isinstance(max_pos, int) and max_pos > 2 else 512)
    inputs = tokenizer(text, truncation=True, max_length=effective_max, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            vec = out.pooler_output[0]
        else:
            vec = out.last_hidden_state[0, 0]
    return vec.detach()

def normalize_list(xs):
    if not xs:
        return []
    mn = min(xs)
    mx = max(xs)
    if mx == mn:
        return [1.0 for _ in xs]
    return [(x - mn) / (mx - mn) for x in xs]

def compute_utterance_scores(sample, tokenizer, model, device):
    utts = sample.get("utterances", [])
    texts = []
    for u in utts:
        sp = str(u.get("speaker", "") or "")
        tx = str(u.get("text", "") or "")
        texts.append("[SPEAKER] " + sp + " : " + tx)
    dialogue_text = " ".join(texts)
    d_vec = encode_text(dialogue_text, tokenizer, model, device)        # 编码
    u_vecs = []
    for t in texts:
        u_vecs.append(encode_text(t, tokenizer, model, device))
    sims = [cosine(v, d_vec) for v in u_vecs]
    norm = normalize_list(sims)
    return norm

def compute_fact_scores(sample, tokenizer, model, device):
    rels = sample.get("relations", [])
    items = []
    idx_map = []
    for r in rels:
        if r.get("is_empty", False):
            continue
        sp = str(r.get("speaker", "") or "")
        ac = str(r.get("action", "") or "")
        ct = str(r.get("content", "") or "")
        txt = "[SPEAKER] " + sp + " [ACTION] " + ac + " [CONTENT] " + ct
        items.append(txt)
        idx_map.append(r.get("r_id"))
    n = len(items)
    if n == 0:
        return {}
    vecs = []
    for t in items:
        vecs.append(encode_text(t, tokenizer, model, device))
    if n == 1:
        return {str(idx_map[0]): 1.0}
    scores = []
    for i in range(n):
        s = 0.0
        cnt = 0
        for j in range(n):
            if i == j:
                continue
            s += cosine(vecs[i], vecs[j])
            cnt += 1
        scores.append(s / max(cnt, 1))
    mx = max(scores) if scores else 1.0
    norm = [s / mx if mx > 0 else 0.0 for s in scores]
    out = {}
    for i, rid in enumerate(idx_map):
        out[str(rid)] = norm[i]
    return out

def compute_word_scores(sample):
    rel_scores = sample.get("relation_word_scores")
    if isinstance(rel_scores, dict) and rel_scores:
        out = {}
        for k, v in rel_scores.items():
            out[str(k)] = float(v)
        return out
    rel_words = sample.get("relation_words", {})
    word_scores = sample.get("word_scores", {})
    out = {}
    for k, words in rel_words.items():
        if not words:
            out[str(k)] = 0.0
            continue
        s = 0.0
        c = 0
        for w in words:
            v = float(word_scores.get(w, 0.0))
            s += v
            c += 1
        out[str(k)] = round(s / c, 6) if c > 0 else 0.0
    return out

def fuse_scores(u_scores, f_scores, w_scores, relations, alpha=0.4, beta=0.3):
    final = {}
    for r in relations:
        rid = str(r.get("r_id"))
        uid = r.get("u_id")
        su = 0.0
        sf = 0.0
        sw = 0.0
        if isinstance(uid, int) and uid < len(u_scores):
            su = float(u_scores[uid])
        if not r.get("is_empty", False):
            sf = float(f_scores.get(rid, 0.0))
            sw = float(w_scores.get(rid, 0.0))
        final[rid] = round(alpha * su + sf + beta * sw, 6)
    ranking = sorted(final.items(), key=lambda x: x[1], reverse=True)
    ordered = [int(k) if k.isdigit() else k for k, _ in ranking]
    return final, ordered

def process_file(input_path, output_path, model_name="bert-large-uncased"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_encoder(model_name)
    model.to(device)
    print(f"使用编码器: {model_name}，设备: {device.type}")
    with open(input_path, "r", encoding="utf-8") as fin:
        lines = [ln.rstrip("\n") for ln in fin]
    with open(output_path, "w", encoding="utf-8") as fout:
        outer = tqdm(total=len(lines), desc="Samples")
        for line in lines:
            line = line.strip()
            if not line:
                outer.update(1)
                continue
            sample = json.loads(line)
            u_scores = compute_utterance_scores(sample, tokenizer, model, device)
            f_scores = compute_fact_scores(sample, tokenizer, model, device)
            w_scores = compute_word_scores(sample)
            final_scores, final_ranking = fuse_scores(u_scores, f_scores, w_scores, sample.get("relations", []))
            record = dict(sample)
            record["u_scores"] = {str(r.get("r_id")): (float(u_scores[r.get("u_id")]) if isinstance(r.get("u_id"), int) and r.get("u_id") < len(u_scores) else 0.0) for r in sample.get("relations", [])}
            record["f_scores"] = f_scores
            record["w_scores"] = w_scores
            record["final_scores"] = final_scores
            record["final_ranking"] = final_ranking
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            outer.update(1)
        outer.close()

if __name__ == "__main__":
    import sys
    group_size = 8
    model_name = "roberta-large"
    for i, arg in enumerate(sys.argv):
        if arg in ("--group_size", "-g") and i + 1 < len(sys.argv):
            try:
                group_size = max(1, int(sys.argv[i + 1]))
            except Exception:
                group_size = 1
        if arg in ("--model", "-m") and i + 1 < len(sys.argv):
            model_name = sys.argv[i + 1]
    base = os.path.join("data", "samsum")
    input_path = os.path.join(base, "word-level", f"test_word-level_{group_size}.jsonl")
    output_path = os.path.join(base, "scores", f"test_scores_{group_size}.jsonl")
    process_file(input_path, output_path, model_name=model_name)
