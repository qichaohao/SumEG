import os
import json
import argparse
import time
import requests
from tqdm import tqdm
import nltk
from rouge_score import rouge_scorer, scoring


# API 配置
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
MAX_RETRIES = 20
RETRY_BACKOFF = 1.0
SLEEP_BETWEEN_REQUESTS = 0.5
MAX_ITERATIONS = 5


def str2bool(v):
    """字符串转布尔值"""
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("yes", "true", "t", "1", "y")


def call_api(api_url, api_key, model_name, prompt, max_tokens=512, temperature=0.7, 
             top_p=0.7, top_k=50, frequency_penalty=0.5, timeout=60, seed=None):
    """调用 API 生成文本"""
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
    except Exception as e:
        print(f"API请求异常: {e}")
        return None
    
    if resp.status_code == 200:
        try:
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            return None
    else:
        print(f"API返回错误: {resp.status_code}, {resp.text[:200]}")
        return None


def generate_via_api(api_url, api_key, model_name, prompt, max_tokens=512, temperature=0.7, 
                     top_p=0.7, top_k=50, frequency_penalty=0.5, base_backoff=RETRY_BACKOFF, 
                     sleep_between=SLEEP_BETWEEN_REQUESTS, seed=None):
    """带重试机制的 API 调用"""
    backoff = base_backoff
    attempt = 0
    
    while True:
        # 如果设置了seed，使用确定性生成参数
        t = 0.0 if (seed is not None and seed >= 0) else temperature
        p = 1.0 if (seed is not None and seed >= 0) else top_p
        fp = 0.0 if (seed is not None and seed >= 0) else frequency_penalty
        
        result = call_api(
            api_url, api_key, model_name, prompt,
            max_tokens=max_tokens, temperature=t, top_p=p, 
            top_k=top_k, frequency_penalty=fp, seed=seed
        )
        
        if result is not None:
            time.sleep(sleep_between)
            return result.strip()
        
        # 重试逻辑
        time.sleep(backoff)
        backoff = min(backoff * 2.0, 60.0)
        attempt += 1
        
        if MAX_RETRIES > 0 and attempt >= MAX_RETRIES:
            print(f"警告: 达到最大重试次数 {MAX_RETRIES}，重置退避时间")
            backoff = RETRY_BACKOFF
            # 继续重试，不退出


def build_evaluator_prompt(dialogue, summary):
    """构建评估器 Prompt（统一版，含 STOP 判断）"""
    prompt = """You are an ABSTRACT REFINEMENT EVALUATOR for dialogue summaries.

Your task is NOT to judge whether the summary is acceptable.
Your task is to decide whether the summary has CONVERGED
to a stable, intent-level abstract form,
or whether further ABSTRACT REFINEMENT is still possible.

--------------------------------------------------
CORE PRINCIPLES
--------------------------------------------------
- The input is already a summary, NOT a dialogue.
- Revisions are NORMAL as long as higher-level abstraction is possible.
- The goal is NOT correctness, but ABSTRACT CONVERGENCE.
- Do NOT add dialogue-level details or surface expressions.

--------------------------------------------------
EVALUATION DIMENSIONS
--------------------------------------------------

1. Intent-Level Abstraction
- Is the summary still describing multiple conversational steps
  when they can be unified into a single intent or action?
- If YES, revision is REQUIRED.

2. Outcome Consolidation
- Does the summary clearly express the final outcome,
  agreement, decision, or conversational resolution?
- If the outcome is implicit but recoverable, revision is REQUIRED.

3. Abstract Redundancy
- From an abstract perspective, are there multiple sentences
  expressing the same intent, hesitation, or confirmation?
- If YES, revision is REQUIRED.

4. Relation–Intent Alignment
- Is the summary focused on the core interaction and intent,
  rather than peripheral chat, jokes, or emotional fillers?
- If misaligned, revision is REQUIRED.

--------------------------------------------------
DECISION RULE
--------------------------------------------------

You MUST first determine whether ANY of the above
dimensions still allow abstract-level improvement.

- Output STOP = true ONLY IF:
  - No further intent unification is possible;
  - No outcome clarification is possible;
  - No abstract redundancy remains;
  - The summary has reached an abstract fixed point.

- Otherwise, output STOP = false.

--------------------------------------------------
EDIT RULES (if STOP = false)
--------------------------------------------------
Propose ONLY minimal abstract edits:
- Rephrase: to merge multiple events into one intent
- Remove: to eliminate abstract-level redundancy
- Add: ONLY to clarify a missing outcome

--------------------------------------------------
OUTPUT FORMAT (STRICT)
--------------------------------------------------

STOP = true

OR

STOP = false
- Rephrase: [...]
- Remove: [...]
- Add: [...]

--------------------------------------------------
Dialogue:
{dialogue}

Current Summary:
{summary}
""".format(dialogue=dialogue, summary=summary)

    return prompt


def build_refiner_prompt(dialogue, summary, edit_instructions):
    """构建优化器 Prompt（与 STOP 机制对齐）"""
    prompt = """You are a SUMMARY EDITING EXECUTOR.

Your task is to APPLY the given edit instructions to the summary.

IMPORTANT RULES (MUST FOLLOW):
1. Do NOT explain your reasoning.
2. Do NOT describe what you are doing.
3. Do NOT include thoughts, analysis, or meta-commentary.
4. Do NOT output <think> tags or similar content.
5. Do NOT add new information.
6. Do NOT expand the summary.
7. Do NOT quote dialogue utterances.
8. Preserve an abstract summary style.

Execution Rules:
- Apply ONLY the listed edit instructions.
- If an edit says "remove", delete the relevant part.
- If an edit says "rephrase", minimally rewrite the relevant part.
- If an edit says "add", add ONLY the missing major event.

Output Rules (STRICT):
- Output ONLY the revised summary text.
- Do NOT include headings, labels, or explanations.
- Do NOT include phrases like "The revised summary is".
- If no edits are needed, output the original summary unchanged.

Dialogue:
<<<DIALOGUE>>>
{dialogue}
<<<DIALOGUE>>>

Current Summary:
{summary}

Edit Instructions:
{edit_instructions}
""".format(
        dialogue=dialogue,
        summary=summary,
        edit_instructions=edit_instructions
    )

    return prompt


def parse_evaluation_result(evaluation_text):
    """解析评估结果，判断是否 STOP 或提取编辑指令"""
    import re
    evaluation_text = evaluation_text.strip()
    
    # 检查是否包含 STOP（不区分大小写，但要求是独立单词）
    # 优先检查：如果文本开头或单独一行就是 STOP
    lines = evaluation_text.split('\n')
    first_line = lines[0].strip().upper() if lines else ""
    
    # 情况1: 第一行就是 STOP（最严格的情况）
    if first_line == "STOP":
        return {"stop": True, "edit_instructions": None}
    
    # 情况2: 检查是否包含独立的 STOP 单词（前后是空白或标点）
    stop_pattern = r'\bSTOP\b'
    if re.search(stop_pattern, evaluation_text, re.IGNORECASE):
        # 进一步检查：如果 STOP 后面没有实质性的编辑指令，认为是停止信号
        stop_match = re.search(stop_pattern, evaluation_text, re.IGNORECASE)
        stop_pos = stop_match.end()
        remaining_text = evaluation_text[stop_pos:].strip()
        
        # 如果 STOP 后面只有很少的内容（可能是误判），或者明确是停止
        if len(remaining_text) < 50 or not any(keyword in remaining_text for keyword in 
                                                ["Issue:", "Edit:", "Faithfulness", "Coverage", "Coherence", "Conciseness"]):
            return {"stop": True, "edit_instructions": None}
    
    # 提取编辑指令（保留原始格式）
    return {"stop": False, "edit_instructions": evaluation_text}


def evaluate_summary(api_url, api_key, model_name, dialogue, summary, 
                     max_tokens=512, temperature=0.7, seed=None):
    """评估摘要质量，返回是否停止和编辑指令"""
    prompt = build_evaluator_prompt(dialogue, summary)
    
    result = generate_via_api(
        api_url=api_url,
        api_key=api_key,
        model_name=model_name,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed
    )
    
    if result is None:
        print("警告: 评估 API 调用失败")
        return {"stop": True, "edit_instructions": None}  # 失败时保守处理：停止
    
    return parse_evaluation_result(result)


def refine_summary(api_url, api_key, model_name, dialogue, summary, edit_instructions,
                   max_tokens=256, temperature=0.7, seed=None):
    """根据编辑指令优化摘要"""
    prompt = build_refiner_prompt(dialogue, summary, edit_instructions)
    
    result = generate_via_api(
        api_url=api_url,
        api_key=api_key,
        model_name=model_name,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed
    )
    
    if result is None:
        print("警告: 优化 API 调用失败，返回原摘要")
        return summary  # 失败时返回原摘要
    
    return result


def process_sample(dialogue, initial_summary, api_url, api_key, model_name,
                   max_iterations=MAX_ITERATIONS, temperature=0.7, seed=None,
                   verbose=False, sample_index=None, total_samples=None):
    """处理单个样本的迭代优化，返回最终摘要、迭代次数和完整迭代历史"""
    current_summary = initial_summary
    iteration_history = []
    
    for iteration in range(1, max_iterations + 1):
        if verbose:
            prefix = ""
            if sample_index is not None and total_samples is not None:
                prefix = f"[样本 {sample_index + 1}/{total_samples}] "
            print(f"{prefix}[迭代 {iteration}/{max_iterations}] 调用评估器(Evaluator)...")
        
        # 评估当前摘要
        eval_result = evaluate_summary(
            api_url=api_url,
            api_key=api_key,
            model_name=model_name,
            dialogue=dialogue,
            summary=current_summary,
            temperature=temperature,
            seed=seed
        )
        
        # 记录本次迭代的评估结果
        iteration_data = {
            "iteration": iteration,
            "summary_before": current_summary,
            "evaluation_result": eval_result.get("edit_instructions", ""),
            "stop_signal": eval_result["stop"]
        }
        
        # 如果评估器输出 STOP，则停止迭代
        if eval_result["stop"]:
            if verbose:
                print(f"  评估器输出 STOP，停止迭代")
            iteration_data["summary_after"] = current_summary
            iteration_history.append(iteration_data)
            return current_summary, iteration, iteration_history
        
        # 否则，根据编辑指令优化摘要
        edit_instructions = eval_result["edit_instructions"]
        if not edit_instructions:
            # 如果没有编辑指令，也停止
            if verbose:
                prefix = ""
                if sample_index is not None and total_samples is not None:
                    prefix = f"[样本 {sample_index + 1}/{total_samples}] "
                print(f"{prefix}[迭代 {iteration}/{max_iterations}] 无编辑指令，停止迭代")
            iteration_data["summary_after"] = current_summary
            iteration_history.append(iteration_data)
            return current_summary, iteration, iteration_history
        
        if verbose:
            prefix = ""
            if sample_index is not None and total_samples is not None:
                prefix = f"[样本 {sample_index + 1}/{total_samples}] "
            print(f"{prefix}[迭代 {iteration}/{max_iterations}] 调用编辑器(Editor)...")
        
        # 执行优化
        refined_summary = refine_summary(
            api_url=api_url,
            api_key=api_key,
            model_name=model_name,
            dialogue=dialogue,
            summary=current_summary,
            edit_instructions=edit_instructions,
            temperature=temperature,
            seed=seed
        )
        
        iteration_data["summary_after"] = refined_summary
        iteration_history.append(iteration_data)
        current_summary = refined_summary
    
    # 达到最大迭代次数
    if verbose:
        prefix = ""
        if sample_index is not None and total_samples is not None:
            prefix = f"[样本 {sample_index + 1}/{total_samples}] "
        print(f"{prefix}达到最大迭代次数 {max_iterations}")
    return current_summary, max_iterations, iteration_history


def compute_rouge_from_paths(pred_path, refs_path):
    """计算ROUGE分数"""
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


def extract_model_basename(model_name):
    """从模型名称中提取基础名称（用于创建目录）"""
    # 例如: "Qwen/Qwen3-8B" -> "Qwen3-8B"
    # 例如: "meta-llama/Meta-Llama-3-8B" -> "Meta-Llama-3-8B"
    name = model_name.replace("\\", "/")
    parts = name.split("/")
    return parts[-1] if parts else model_name


def run(initial_summaries_path, dialogues_path, output_path, api_url, api_key, model_name,
        max_iterations=MAX_ITERATIONS, temperature=0.7, seed=None, verbose=True, refs_path=None, run_rouge=True):
    """主运行函数"""
    # 验证 API key
    key = api_key or os.environ.get("SILICONFLOW_API_KEY", "")
    if not key:
        raise RuntimeError("API key not provided. Set --api_key or environment SILICONFLOW_API_KEY.")
    
    endpoint = api_url or API_URL
    
    # 自动创建模型名字目录
    model_basename = extract_model_basename(model_name)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        # 如果输出路径包含目录，确保模型名字目录存在
        model_dir = os.path.join(output_dir, model_basename)
        os.makedirs(model_dir, exist_ok=True)
        # 更新输出路径
        output_filename = os.path.basename(output_path)
        output_path = os.path.join(model_dir, output_filename)
    else:
        # 如果输出路径只是文件名，创建模型名字目录
        model_dir = model_basename
        os.makedirs(model_dir, exist_ok=True)
        output_path = os.path.join(model_dir, output_path)
    
    # 创建迭代历史文件路径（JSONL格式）
    history_path = output_path.replace(".txt", "-history.jsonl")
    
    print(f"使用API: {endpoint}，模型: {model_name}")
    print(f"最大迭代次数: {max_iterations}")
    print(f"初始摘要文件: {initial_summaries_path}")
    print(f"对话文件: {dialogues_path}")
    print(f"输出文件: {output_path}")
    print(f"迭代历史文件: {history_path}")
    
    # 读取初始摘要
    initial_summaries = []
    with open(initial_summaries_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                initial_summaries.append(line.strip())
    
    # 读取对话数据
    dialogues = []
    with open(dialogues_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # 提取对话文本（根据实际格式调整）
                if "text" in data:
                    dialogues.append(data["text"])
                elif "dialogue" in data:
                    dialogues.append(data["dialogue"])
                else:
                    # 尝试从其他字段提取
                    dialogues.append(str(data))
    
    # 验证数据长度匹配
    if len(initial_summaries) != len(dialogues):
        raise ValueError(
            f"数据长度不匹配: 初始摘要数量={len(initial_summaries)}, "
            f"对话数量={len(dialogues)}"
        )
    
    print(f"共 {len(initial_summaries)} 个样本需要处理")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    # 处理每个样本
    iteration_counts = []
    bar = tqdm(total=len(initial_summaries), desc="处理样本", leave=True, ncols=80)

    with open(output_path, "w", encoding="utf-8") as fout, \
         open(history_path, "w", encoding="utf-8") as fhistory:
        for idx, (initial_summary, dialogue) in enumerate(zip(initial_summaries, dialogues)):
            if verbose:
                bar.set_postfix_str(f"样本 {idx+1}/{len(initial_summaries)}")

            # 执行迭代优化
            final_summary, iterations, iteration_history = process_sample(
                dialogue=dialogue,
                initial_summary=initial_summary,
                api_url=endpoint,
                api_key=key,
                model_name=model_name,
                max_iterations=max_iterations,
                temperature=temperature,
                seed=seed,
                verbose=verbose,
                sample_index=idx,
                total_samples=len(initial_summaries)
            )

            iteration_counts.append(iterations)

            # 写入最终摘要（清理换行符）
            cleaned_summary = final_summary.replace("\r", " ").replace("\n", " ").strip()
            fout.write(cleaned_summary + "\n")
            fout.flush()
            os.fsync(fout.fileno())

            # 保存完整的迭代历史
            sample_history = {
                "sample_id": idx,
                "dialogue": dialogue,
                "initial_summary": initial_summary,
                "final_summary": final_summary,
                "total_iterations": iterations,
                "iterations": iteration_history
            }
            fhistory.write(json.dumps(sample_history, ensure_ascii=False) + "\n")
            fhistory.flush()
            os.fsync(fhistory.fileno())

            bar.update(1)
            bar.refresh()
    
    bar.close()
    
    # 输出统计信息
    avg_iterations = sum(iteration_counts) / len(iteration_counts) if iteration_counts else 0
    print(f"\n处理完成!")
    print(f"平均迭代次数: {avg_iterations:.2f}")
    print(f"迭代次数分布: {dict((i, iteration_counts.count(i)) for i in range(1, max_iterations + 2))}")
    print(f"结果已保存到: {output_path}")
    print(f"迭代历史已保存到: {history_path}")
    
    # 计算ROUGE分数
    if run_rouge and refs_path:
        print(f"\n计算ROUGE分数...")
        print(f"参考文件: {refs_path}")
        try:
            result = compute_rouge_from_paths(output_path, refs_path)
        except Exception as e:
            print(f"ROUGE计算失败: {e}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Quality-Aware Iterative Refinement Module: "
                    "LLM-driven Evaluation with Explicit STOP Decision"
    )
    parser.add_argument("--initial_summaries_path", type=str, default="data/samsum/ali/1-qwen3-8b-baseline-predictions.txt", help="初始摘要文件路径（TXT格式，每行一个摘要）")
    parser.add_argument("--dialogues_path",type=str,default="data/samsum/test.json",help="测试集对话文件路径（JSONL格式）")
    parser.add_argument("--output_path",type=str,default="data/samsum/final/1-qwen3-8b-refined-predictions.txt",help="输出文件路径")
    parser.add_argument("--api_url",type=str,default=API_URL,help="API 端点 URL")
    parser.add_argument("--api_key",type=str,default="",help="API Key")
    parser.add_argument("--model_name",type=str,default="Qwen/Qwen3-8B",help="模型名称")
    parser.add_argument("--max_iterations",type=int,default=MAX_ITERATIONS,help="最大迭代次数（默认3，作为保险）")
    parser.add_argument("--temperature",type=float,default=0.7,help="生成温度")
    parser.add_argument("--seed",type=int,default=None,help="随机种子（None表示不使用，>=0表示使用确定性生成）")
    parser.add_argument("--verbose", type=str2bool, default=True, help="是否显示详细进度")
    parser.add_argument("--refs_path", type=str, default="data/samsum/test.json", help="参考摘要文件路径（用于计算ROUGE）")
    parser.add_argument("--run_rouge", type=str2bool, default=True, help="是否计算ROUGE分数")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    run(
        initial_summaries_path=args.initial_summaries_path,
        dialogues_path=args.dialogues_path,
        output_path=args.output_path,
        api_url=args.api_url,
        api_key=args.api_key,
        model_name=args.model_name,
        max_iterations=args.max_iterations,
        temperature=args.temperature,
        seed=args.seed if args.seed is not None and args.seed >= 0 else None,
        verbose=args.verbose,
        refs_path=args.refs_path,
        run_rouge=args.run_rouge
    )
