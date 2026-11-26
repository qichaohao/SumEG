# 第283行更换数据集
import re
import json
import requests
from tqdm import tqdm
import time
import os
from typing import List, Optional
# === SiliconFlow API 配置 ===
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL_NAME = "deepseek-ai/DeepSeek-V3"
API_KEY = "sk-qxbctwhswdbnmhpobtshejniudvutcqsbusswvwveldwiwqv"

# ====== 可配置项 ======
MAX_RETRIES = 10
RETRY_BACKOFF = 1.0
SLEEP_BETWEEN_REQUESTS = 0.5
TEMP_INPUT_SUFFIX = ".tmp"

# ===============================
# 🔧 JSON 自动修复函数
# ===============================
def fix_json_string(text: str) -> str:
    if not isinstance(text, str):
        return text

    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        json.loads(text)
        return text
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        text = m.group(0)

    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = re.sub(r"[\u200b-\u200f\u2028\u2029\u00a0\uFEFF]", "", text)

    text = re.sub(r'(?<=\{|\s)([A-Za-z_][A-Za-z0-9_]*)\s*:', r'"\1":', text)
    text = re.sub(r"\'([^']*?)\'", r'"\1"', text)
    text = re.sub(r",\s*(\}|\])", r"\1", text)
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    try:
        json.loads(text)
    except Exception:
        text = text.replace("\\", "")
        text = re.sub(r"\s+", " ", text).strip()

    return text.strip()

# ===============================
# 🔧 调用大模型 API
# ===============================
def call_api(messages: List[dict]) -> Optional[str]:
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "max_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=60)
    except Exception:
        return None

    if response.status_code == 200:
        try:
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            return None
    return None

# ===============================
# 🔧 utterance → relation 抽取
# ===============================
GREETINGS = {"hi", "hello", "hey", "bye", "bye bye", "goodbye", "ttyl"}

BAD_ACTIONS = {
    "know", "knows", "think", "feel", "believe",
    "rather", "maybe", "guess", "like"
}

def is_media_only(text: str) -> bool:
    return bool(re.fullmatch(r"\s*(<file_[^>]+>\s*)+", text))

def is_greeting_or_bye(text: str) -> bool:
    t = text.strip().lower()
    if t in GREETINGS:
        return True
    if re.search(r"\bsee (you|ya)\b", t):
        return True
    return False

def extract_relation(speaker: str, utterance: str) -> dict:
    candidates = [s.strip() for s in speaker.split("|")] if speaker else []
    content_only = re.sub(r"(?m)^[^:]+:\s*", "", utterance)
    if not content_only.strip() or is_media_only(content_only) or is_greeting_or_bye(content_only):
        return {"speaker": "", "action": "", "content": ""}

    prompt = (
        "Role:\n"
        "You are an expert in dialogue relation extraction for dialogue summarization.\n\n"
        "Task:\n"
        "Given a group of up to K dialogue lines, extract EXACTLY ONE factual relation that could appear in a summary.\n\n"
        "Candidates:\n"
        f"{json.dumps(candidates, ensure_ascii=False)}\n\n"
        "Rules:\n"
        "1. Choose the 'speaker' STRICTLY from Candidates.\n"
        "2. Output exactly ONE JSON object.\n"
        "3. If no clear factual action exists, output an EMPTY relation with speaker=\"\".\n"
        "4. Use a short verb phrase for 'action'.\n"
        "5. Do not infer intentions or emotions.\n\n"
        "Allowed actions (examples):\n"
        "ask, answer, request, suggest, inform, explain, report, confirm, deny, agree, refuse, encourage, offer, check, call, text, watch, go, leave\n\n"
        "Forbidden actions:\n"
        "know, think, feel, believe, like, rather, maybe, guess, greetings, fillers\n\n"
        "Output format:\n"
        "{\n"
        '  "speaker": "one of Candidates or empty",\n'
        '  "action": "...",\n'
        '  "content": "..."\n'
        "}\n\n"
        "Group:\n"
        f"{utterance}"
    )

    messages = [{"role": "user", "content": prompt}]

    for attempt in range(MAX_RETRIES):
        if attempt > 0:
            time.sleep(RETRY_BACKOFF * (2 ** (attempt - 1)))

        result = call_api(messages)
        if result is None:
            continue

        fixed = fix_json_string(result)
        try:
            rel = json.loads(fixed)
            action = rel.get("action", "").strip().lower()
            content = rel.get("content", "").strip()
            chosen = rel.get("speaker", "")
            if chosen is None:
                chosen = ""
            chosen = str(chosen).strip()
            if candidates and chosen not in candidates:
                chosen = ""

            if action in BAD_ACTIONS:
                return {"speaker": "", "action": "", "content": ""}

            if not action and not content:
                return {"speaker": "", "action": "", "content": ""}

            return {
                "speaker": chosen,
                "action": rel.get("action", "").strip(),
                "content": rel.get("content", "").strip()
            }
        except Exception:
            continue

    return {"speaker": "", "action": "", "content": ""}

# ===============================
# 🔧 原子写入
# ===============================
def atomic_write_lines(filepath: str, lines: List[str]):
    tmp = filepath + TEMP_INPUT_SUFFIX
    with open(tmp, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")
    os.replace(tmp, filepath)

# ===============================
# 🔧 数据集处理主流程
# ===============================
def process_dataset(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    pbar = tqdm(total=len(lines), desc="Extracting relations")
    processed_count = 0
    processed_ids = set()

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as fout:
            for ln in fout:
                try:
                    processed_ids.add(json.loads(ln).get("id"))
                except Exception:
                    pass

    while lines:
        raw = lines.pop(0)
        pbar.update(1)

        if not raw.strip():
            continue

        try:
            data = json.loads(raw)
        except Exception:
            continue

        sid = data.get("id")
        if sid in processed_ids:
            continue

        utterances_list = data.get("utterances", [])
        total_utts = len(utterances_list)
        if total_utts > 0:
            pbar.set_postfix_str(f"id={sid} u=0/{total_utts} (0%)")

        relations = []
        for i, u in enumerate(utterances_list):
            rel = extract_relation(u["speaker"], u["text"])
            action = rel["action"]
            content = rel["content"]
            relations.append({
                "r_id": i,
                "u_id": u["u_id"],
                "speaker": rel.get("speaker", ""),
                "action": action,
                "content": content,
                "is_empty": (not action and not content)
            })
            if total_utts > 0:
                pct = int(((i + 1) / total_utts) * 100)
                pbar.set_postfix_str(f"id={sid} u={i+1}/{total_utts} ({pct}%)")

        data["relations"] = relations

        with open(output_path, "a", encoding="utf-8") as fout:
            json.dump(data, fout, ensure_ascii=False)
            fout.write("\n")
            fout.flush()
            os.fsync(fout.fileno())

        atomic_write_lines(input_path, lines)
        processed_count += 1
        sample_id = data.get("id", "")
        print(f"✅ 样本 {sample_id} 处理成功并写入输出（已处理 {processed_count} 条）")
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    pbar.close()

    print("✅ Relation extraction finished.")


# ===============================
# 🔧 主入口

# ===============================
if __name__ == "__main__":
    import sys
    group_size = 8
    for i, arg in enumerate(sys.argv):
        if arg in ("--group_size", "-g") and i + 1 < len(sys.argv):
            try:
                group_size = max(1, int(sys.argv[i + 1]))
            except Exception:
                group_size = 1
    base = os.path.join("data", "samsum")
    input_jsonl = os.path.join(base, "utterance", f"test_utterance_{group_size}.jsonl")
    output_jsonl = os.path.join(base, "relation", f"test_relations_{group_size}.jsonl")
    process_dataset(input_jsonl, output_jsonl)
