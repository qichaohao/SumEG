# 第6行和41行更换数据集
import os
import json
import re
def read_input_path():
    base = os.path.join("data", "samsum")   # dialogsum,samsum
    p_jsonl = os.path.join(base, "test.jsonl")
    p_json = os.path.join(base, "test.json")
    if os.path.exists(p_jsonl):
        return p_jsonl
    return p_json

def split_dialog_text(text):
    lines = re.split(r"\r?\n", text)
    items = []
    for line in lines:
        if not line:
            continue
        idx = line.find(":")
        if idx == -1:
            speaker = ""
            content = line
        else:
            speaker = line[:idx].strip()
            content = line[idx + 1:]
            if len(content) > 0 and content[0] == " ":
                content = content[1:]
        items.append((speaker, content))
    return items

def main():
    import sys
    input_path = read_input_path()
    group_size = 8
    for i, arg in enumerate(sys.argv):
        if arg in ("--group_size", "-g") and i + 1 < len(sys.argv):
            try:
                group_size = max(1, int(sys.argv[i + 1]))
            except Exception:
                group_size = 1
    base = os.path.join("data", "samsum")
    output_path = os.path.join(base, "utterance", f"test_utterance_{group_size}.jsonl")
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            pairs = split_dialog_text(text)
            utterances = []
            if group_size <= 1:
                position = 0
                for speaker, content in pairs:
                    utterances.append({
                        "u_id": position,
                        "speaker": speaker,
                        "text": content,
                        "position": position,
                    })
                    position += 1
            else:
                position = 0
                i = 0
                while i < len(pairs):
                    chunk = pairs[i:i+group_size]
                    speakers = []
                    lines = []
                    for sp, ct in chunk:
                        speakers.append(sp)
                        lines.append(f"{sp}: {ct}")
                    speaker_str = " | ".join(speakers)
                    text_chunk = "\n".join(lines)
                    utterances.append({
                        "u_id": position,
                        "speaker": speaker_str,
                        "text": text_chunk,
                        "position": position,
                    })
                    position += 1
                    i += group_size
            out_obj = {
                "id": obj.get("id"),
                "text": text,
                "summary": obj.get("summary", ""),
                "utterances": utterances
            }
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
