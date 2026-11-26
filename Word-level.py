# 第278行更换数据集
import os
import json
import re
import math
from collections import defaultdict
STOPWORDS = {
    "a","an","the","and","or","but","if","then","so","of","to","in","on","for","with","at","by","from",
    "is","are","was","were","be","been","am","do","does","did","have","has","had","will","would","should","can","could",
    "i","you","he","she","it","we","they","me","him","her","us","them","my","your","his","her","its","our","their",
    "this","that","these","those","here","there","now","today","yesterday","tomorrow",
    "yes","no","ok","okay","alright","sure","lol","xd","xD"
}

EXTRA_STOPWORDS = {
    "just","very","too","out","turn","turns","turning","turned",
    "lemme","gonna","wanna","gotta","nah","yeah","yep","nope",
    "thanks","thank","please","btw","haha","lol","rofl","lmao",
    "dont","cant","wouldnt","shouldnt","couldnt","isnt","arent",
    "im","ill","hes","shes","youre","theyre","ive","weve","theyve",
    "didnt","doesnt","havent","hasnt","hadnt"
}

DIALOGUE_ACTS = {
    "agree","check","ask","inform","report","encourage",
    "request","deny","confirm","refuse","offer","answer",
    "explain","suggest","watch"
}

def lemmatize(token):
    if len(token) > 4 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 3 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token

def extract_words(text):
    raw = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
    proper = {}
    cleaned = []
    for w in raw:
        is_proper = bool(re.match(r"^[A-Z][a-zA-Z'-]+$", w))
        if "'" in w:
            if not (is_proper and re.match(r"^[A-Z][a-zA-Z]+('[A-Za-z]+)+$", w)):
                continue
        lw = lemmatize(w.lower())
        if len(lw) < 3:
            continue
        if lw in DIALOGUE_ACTS:
            continue
        if lw not in STOPWORDS and lw not in EXTRA_STOPWORDS and re.search(r"[aeiou]", lw):
            cleaned.append(lw)
            if is_proper:
                if lw not in STOPWORDS:
                    proper[lw] = w
    return cleaned, proper

def build_sample_graph(rel_words):
    g = defaultdict(set)
    for i, words in enumerate(rel_words):
        s = list(set(words))
        for a in range(len(s)):
            for b in range(a + 1, len(s)):
                wa = s[a]; wb = s[b]
                if wa != wb:
                    g[wa].add(wb)
                    g[wb].add(wa)
        if i + 1 < len(rel_words):
            t = list(set(rel_words[i + 1]))
            for wa in s:
                for wb in t:
                    if wa != wb:
                        g[wa].add(wb)
                        g[wb].add(wa)
    return g

def textrank_graph(graph, iterations=20, damping=0.85):
    nodes = list(graph.keys())
    if not nodes:
        return {}
    scores = {n: 1.0 for n in nodes}
    for _ in range(iterations):
        ns = {}
        for n in nodes:
            neighbors = graph[n]
            s = (1 - damping)
            for m in neighbors:
                deg = len(graph[m]) or 1
                s += damping * (scores[m] / deg)
            ns[n] = s
        scores = ns
    mx = max(scores.values()) if scores else 1.0
    return {k: (v / mx) for k, v in scores.items()}

def compute_iaid_scores(relations, rel_words, is_empty_flags, iterations=30, damping=0.85, delta=1.0):
    all_words = set()
    for ws in rel_words:
        for w in ws:
            all_words.add(w)
    if not all_words:
        return {}
    n_rel = len(rel_words)
    speakers = []
    for r in relations:
        speakers.append((r.get("speaker", "") or "").strip())
    while len(speakers) < n_rel:
        speakers.append("")
    word_rel_indices = defaultdict(set)
    word_speaker_counts = defaultdict(lambda: defaultdict(int))
    cooc_counts = defaultdict(int)
    for idx, words in enumerate(rel_words):
        if idx < len(is_empty_flags) and is_empty_flags[idx]:
            continue
        s = list(set(words))
        if not s:
            continue
        sp = speakers[idx] if idx < len(speakers) else ""
        for w in s:
            word_rel_indices[w].add(idx)
            if sp:
                word_speaker_counts[w][sp] += 1
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                a = s[i]
                b = s[j]
                if a == b:
                    continue
                if a > b:
                    a, b = b, a
                cooc_counts[(a, b)] += 1
    n_windows = max(1, n_rel)
    freq_rel = {}
    for w, idxs in word_rel_indices.items():
        freq_rel[w] = len(idxs)
    eps = 1e-9
    pmi = {}
    for (a, b), c in cooc_counts.items():
        pa = freq_rel.get(a, 0) / n_windows
        pb = freq_rel.get(b, 0) / n_windows
        pij = c / n_windows
        if pa <= 0 or pb <= 0 or pij <= 0:
            v = 0.0
        else:
            v = math.log(pij / (pa * pb + eps) + eps)
            if v < 0:
                v = 0.0
        pmi[(a, b)] = v
        pmi[(b, a)] = v
    H = {}
    for w in all_words:
        counts = word_speaker_counts.get(w, {})
        if not counts or len(counts) <= 1:
            H[w] = 0.0
            continue
        total = sum(counts.values())
        if total <= 0:
            H[w] = 0.0
            continue
        ps = []
        for c in counts.values():
            ps.append(c / total)
        h = 0.0
        for p in ps:
            if p > 0:
                h -= p * math.log(p + eps)
        norm = math.log(len(counts) + eps)
        if norm <= 0:
            H[w] = 0.0
        else:
            H[w] = h / norm
    graph = defaultdict(dict)
    for idx, words in enumerate(rel_words):
        if idx < len(is_empty_flags) and is_empty_flags[idx]:
            continue
        s = list(set(words))
        if not s:
            continue
        for i in range(len(s)):
            for j in range(len(s)):
                if i == j:
                    continue
                a = s[i]
                b = s[j]
                w = pmi.get((a, b), 0.0)
                if w <= 0:
                    continue
                graph[a][b] = graph[a].get(b, 0.0) + w
    n = len(rel_words)
    for i in range(n):
        if i < len(is_empty_flags) and is_empty_flags[i]:
            continue
        s_i = list(set(rel_words[i]))
        if not s_i:
            continue
        sp_i = speakers[i] if i < len(speakers) else ""
        for j in range(i + 1, n):
            if j < len(is_empty_flags) and is_empty_flags[j]:
                continue
            s_j = list(set(rel_words[j]))
            if not s_j:
                continue
            sp_j = speakers[j] if j < len(speakers) else ""
            factor = 1.0
            if sp_i and sp_j and sp_i != sp_j:
                factor = math.exp(delta)
            for a in s_i:
                for b in s_j:
                    if a == b:
                        continue
                    w = pmi.get((a, b), 0.0)
                    if w <= 0:
                        continue
                    wv = w * factor
                    graph[a][b] = graph[a].get(b, 0.0) + wv
    nodes = set()
    for a, nbrs in graph.items():
        nodes.add(a)
        for b in nbrs.keys():
            nodes.add(b)
    if not nodes:
        return {}
    out_sum = {}
    for a, nbrs in graph.items():
        s = 0.0
        for v in nbrs.values():
            s += v
        out_sum[a] = s
    in_neighbors = defaultdict(list)
    for a, nbrs in graph.items():
        for b, w in nbrs.items():
            in_neighbors[b].append((a, w))
    H_node = {}
    for w in nodes:
        H_node[w] = H.get(w, 0.0)
    all_zero = True
    for v in H_node.values():
        if v > 0:
            all_zero = False
            break
    if all_zero:
        for w in nodes:
            H_node[w] = 1.0
    scores = {}
    for w in nodes:
        scores[w] = H_node[w]
    for _ in range(iterations):
        ns = {}
        for i in nodes:
            base = (1.0 - damping) * H_node[i]
            acc = 0.0
            for j, w in in_neighbors.get(i, []):
                denom = out_sum.get(j, 0.0)
                if denom <= 0:
                    continue
                acc += w * (scores.get(j, 0.0) / denom)
            ns[i] = base + damping * acc
        scores = ns
    mx = max(scores.values()) if scores else 1.0
    if mx <= 0:
        mx = 1.0
    for k in scores.keys():
        scores[k] = scores[k] / mx
    return scores

def main():
    import sys
    group_size = 8
    for i, arg in enumerate(sys.argv):
        if arg in ("--group_size", "-g") and i + 1 < len(sys.argv):
            try:
                group_size = max(1, int(sys.argv[i + 1]))
            except Exception:
                group_size = 1
    base = os.path.join("data", "samsum")
    input_path = os.path.join(base, "relation", f"test_relations_{group_size}.jsonl")
    output_path = os.path.join(base, "word-level", f"test_word-level_{group_size}.jsonl")
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            sid = data.get("id", "")
            rels = data.get("relations", [])
            rel_words = []
            proper_map = {}
            is_empty_flags = []
            for r in rels:
                action = r.get("action", "") or ""
                content = r.get("content", "") or ""
                is_empty = bool(r.get("is_empty", False))
                is_empty_flags.append(is_empty)
                if is_empty:
                    rel_words.append([])
                    continue
                t = (action + " " + content).strip()
                tokens, proper = extract_words(t)
                rel_words.append(tokens)
                for k, v in proper.items():
                    if k not in proper_map:
                        proper_map[k] = v
            scores = compute_iaid_scores(rels, rel_words, is_empty_flags)
            relation_words = {}
            relation_word_scores = {}
            for i, words in enumerate(rel_words):
                if is_empty_flags[i] or not words:
                    relation_words[str(i)] = []
                    relation_word_scores[str(i)] = 0.0
                    continue
                uniq = list(set(words))
                ranked = sorted(((w, scores.get(w, 0.0)) for w in uniq), key=lambda x: x[1], reverse=True)
                top = [w for w, _ in ranked[:5]]
                show = []
                for w in top:
                    show.append(proper_map.get(w, w))
                relation_words[str(i)] = show
                if top:
                    avg = sum(scores.get(w, 0.0) for w in top) / len(top)
                else:
                    avg = 0.0
                relation_word_scores[str(i)] = round(avg, 6)
            record = dict(data)
            record["relation_words"] = relation_words
            record["word_scores"] = {proper_map.get(w, w): v for w, v in scores.items()}
            record["relation_word_scores"] = relation_word_scores
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
