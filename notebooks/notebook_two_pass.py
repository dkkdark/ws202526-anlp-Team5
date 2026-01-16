# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "bert-score==0.3.13",
#     "evaluate==0.4.6",
#     "langchain-core==1.2.6",
#     "langchain-openai==1.1.7",
#     "marimo",
#     "numpy==2.2.6",
#     "pandas==2.3.3",
#     "scikit-learn==1.8.0",
#     "torch==2.9.1",
#     "transformers==4.57.3",
# ]
# ///

import marimo

__generated_with = "0.19.4"
app = marimo.App(
    width="medium",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell
def _():
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    import marimo as mo
    return ChatOpenAI, ChatPromptTemplate, mo


@app.cell
def _(mo):
    api_key = mo.ui.text_area(
        label="",
        rows=3,
    )
    api_key
    return (api_key,)


@app.cell
def _(ChatOpenAI, ChatPromptTemplate, api_key):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        api_key=api_key.value
    )

    simplification_system_prompt = """You are a text simplification expert. Your task is to rewrite complex sentences into simpler versions that are easy to understand at the requested CEFR level.

    Rules:
    1. Preserve all original meaning and facts.
    2. Do not add new facts, names, places, dates, or numbers.
    3. Do not remove important facts or change quantities.
    4. Use simpler words and shorter sentences.
    5. Avoid jargon, idioms, metaphors, and figurative language.
    6. Keep the original tone (neutral, factual) and do not add opinions.
    7. If the original is uncertain, keep the same uncertainty.
    8. Prefer clear, direct sentences; split long sentences if needed.
    9. Keep references (people, places, organizations) consistent.
    10. Output only the simplified text. No explanations or labels.
    """

    user_prompt_template = """Simplify this sentence to CEFR {level}:
    {sentence}
    """

    retry_prompt_template = """Simplify this sentence to CEFR {level}.
    Use the critique to fix problems from the previous attempt.

    Critique:
    {critique}

    Sentence:
    {sentence}
    """

    single_pass_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", simplification_system_prompt),
            ("user", user_prompt_template),
        ]
    )

    retry_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", simplification_system_prompt),
            ("user", retry_prompt_template),
        ]
    )

    judge_prompt_template = """Compare the original and simplified text.
    Target CEFR level: {level}

    Check:
    1) No new facts, names, places, dates, or numbers were added.
    2) Meaning is preserved (no important information removed or changed).
    3) The simplified text is at or simpler than CEFR {level}.

    Return ONLY:
    OK
    or
    FAIL: <short reason>

    Original:
    {sentence}

    Simplified:
    {draft}
    """

    judge_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a strict evaluator. Do not rewrite. Only judge."),
            ("user", judge_prompt_template),
        ]
    )

    def simplify_sentence_single_pass(sentence: str, level: str) -> str:
        response = llm.invoke(
            single_pass_prompt.format_messages(sentence=sentence, level=level)
        )
        return response.content.strip()

    def simplify_sentence_two_pass(sentence: str, level: str, max_attempts: int = 3) -> str:
        last_draft = ""
        critique = ""
        for attempt in range(max_attempts):
            if attempt == 0:
                last_draft = llm.invoke(
                    single_pass_prompt.format_messages(sentence=sentence, level=level)
                ).content.strip()
            else:
                last_draft = llm.invoke(
                    retry_prompt.format_messages(
                        sentence=sentence, level=level, critique=critique
                    )
                ).content.strip()

            verdict = llm.invoke(
                judge_prompt.format_messages(sentence=sentence, level=level, draft=last_draft)
            ).content.strip()

            if verdict.upper().startswith("OK"):
                return last_draft
            critique = verdict

        return (
            f"{last_draft} "
            "Note: This simplification may not fully match the target level or may still be too close to the original."
        )

    text = (
        "Although the committee acknowledged the long-term environmental benefits, "
        "it postponed the policy change due to budget constraints and concerns about "
        "short-term economic disruption."
    )

    print("Original: ", text)
    print("Single-pass A2:", simplify_sentence_single_pass(text, "A2"))
    print("Two-pass A2:", simplify_sentence_two_pass(text, "A2"))
    print("Single-pass B1:", simplify_sentence_single_pass(text, "B1"))
    print("Two-pass B1:", simplify_sentence_two_pass(text, "B1"))
    return simplify_sentence_single_pass, simplify_sentence_two_pass


@app.cell
def _(simplify_sentence_single_pass):
    import json
    from pathlib import Path

    in_path_single = Path(
        r"/__modal/volumes/vo-stCe2zDTIgG66Cs3VLKXbH/tsar2025_test.jsonl"
    )
    out_path_single = Path(
        "results_llm_single_pass_reduced.jsonl"
    )

    outputs_single = []

    with in_path_single.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            row_l = json.loads(line)
            text_id = row_l["text_id"]
            original = row_l["original"]
            level = str(row_l["target_cefr"]).strip().upper()

            print(original)
            print(level)

            try:
                simplified = simplify_sentence_single_pass(original, level)
            except Exception as e:
                simplified = original
                print(f"[warn] line={line_no} text_id={text_id} level={level}: {e}")

            outputs_single.append({"text_id": text_id, "simplified": simplified})

    with out_path_single.open("w", encoding="utf-8") as f:
        for obj in outputs_single:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    out_path_single
    return Path, json


@app.cell
def _(Path, json, simplify_sentence_two_pass):

    in_path_two_pass_test = Path(
        r"/__modal/volumes/vo-stCe2zDTIgG66Cs3VLKXbH/tsar2025_test.jsonl"
    )
    out_path_two_pass_test = Path(
        "results_llm_two_pass_reduced.jsonl"
    )

    outputs_two_pass_test = []

    with in_path_two_pass_test.open("r", encoding="utf-8") as file_tp:
        for line_no_tp, line_tp in enumerate(file_tp, start=1):
            line_tp = line_tp.strip()
            if not line_tp:
                continue

            row_tp = json.loads(line_tp)
            text_id_tp = row_tp["text_id"]
            original_tp = row_tp["original"]
            level_tp = str(row_tp["target_cefr"]).strip().upper()

            print(original_tp)
            print(level_tp)

            try:
                simplified_tp = simplify_sentence_two_pass(original_tp, level_tp)
            except Exception as err_tp:
                simplified_tp = original_tp
                print(f"[warn] line={line_no_tp} text_id={text_id_tp} level={level_tp}: {err_tp}")

            outputs_two_pass_test.append({"text_id": text_id_tp, "simplified": simplified_tp})

    with out_path_two_pass_test.open("w", encoding="utf-8") as file_out_tp:
        for obj_tp in outputs_two_pass_test:
            file_out_tp.write(json.dumps(obj_tp, ensure_ascii=False) + "\n")

    out_path_two_pass_test
    return


@app.cell
def _():
    import random, os, json
    import numpy as np
    import pandas as pd
    from sklearn.metrics import f1_score, root_mean_squared_error
    from transformers import pipeline # IMPORTANT: Please ensure your transformers version is v4.55
    import evaluate

    # ---------------- Config ----------------
    GOLD_FILE = "/__modal/volumes/vo-stCe2zDTIgG66Cs3VLKXbH/content/tsar2025_test.jsonl"   # gold file next to this script
    SUBMISSIONS_DIR = "/__modal/volumes/vo-stCe2zDTIgG66Cs3VLKXbH/content/submission"     # folder with team subfolders
    SEED = 42                           # for reproducibility
    BATCH_SIZE = 32                     # adjust for your GPU

    # ---------------- Seed ------------------
    random.seed(SEED)
    np.random.seed(SEED)
    try:
        import torch
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    # ---------------- IO --------------------
    def read_jsonl(path: str):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def read_gold(path: str):
        data = read_jsonl(path)
        if not data:
            raise ValueError(f"Gold file is empty: {path}")
        try:
            original = [e["original"] for e in data]
            reference = [e["reference"] for e in data]
            target   = [e["target_cefr"] for e in data]   # case handled later
            text_ids = [e["text_id"] for e in data]
        except KeyError as ke:
            raise KeyError(f"Gold file missing key {ke}. First item keys: {list(data[0].keys())}")
        return original, reference, target, text_ids

    def read_submission(path: str):
        data = read_jsonl(path)
        if not data:
            raise ValueError(f"Submission is empty: {path}")
        first_keys = list(data[0].keys())
        if "simplified" not in data[0]:
            raise KeyError(f"{path} must contain 'simplified'. Found keys: {first_keys}")
        if "text_id" not in data[0]:
            raise KeyError(f"{path} must contain 'text_id'. Found keys: {first_keys}")
        return [e["simplified"] for e in data], [e["text_id"] for e in data], len(data)

    # Align system outputs to ANY overlapping gold ids (supports partial submissions)
    def align_intersection(hyps, sys_ids, gold_ids, gold_orig, gold_ref, gold_tgt):
        gid2idx = {g:i for i,g in enumerate(gold_ids)}
        pairs = [(gid2idx[sid], hyp) for hyp, sid in zip(hyps, sys_ids) if sid in gid2idx]
        if not pairs:
            return None
        pairs.sort(key=lambda x: x[0])
        sel_idx = [i for i,_ in pairs]
        aligned_hyps = [h for _,h in pairs]
        aligned_orig = [gold_orig[i] for i in sel_idx]
        aligned_ref  = [gold_ref[i]  for i in sel_idx]
        aligned_tgt  = [gold_tgt[i]  for i in sel_idx]
        coverage_n   = len(sel_idx)
        coverage_pct = round(100.0 * coverage_n / len(gold_ids), 2)
        missing_ids  = [g for g in gold_ids if g not in set(sys_ids)]
        extra_ids    = [s for s in sys_ids if s not in set(gold_ids)]
        return {
            "hyps": aligned_hyps,
            "orig": aligned_orig,
            "ref":  aligned_ref,
            "tgt":  aligned_tgt,
            "coverage_n": coverage_n,
            "coverage_pct": coverage_pct,
            "missing_ids": missing_ids,
            "extra_ids": extra_ids
        }

    # ------------- Models/Metrics -----------
    cefr_labeler1 = pipeline("text-classification",
        model="AbdullahBarayan/ModernBERT-base-doc_en-Cefr", device=0, torch_dtype="auto")
    cefr_labeler2 = pipeline("text-classification",
        model="AbdullahBarayan/ModernBERT-base-doc_sent_en-Cefr", device=0, torch_dtype="auto")
    cefr_labeler3 = pipeline("text-classification",
        model="AbdullahBarayan/ModernBERT-base-reference_AllLang2-Cefr2", device=0, torch_dtype="auto")

    meaning_bert = evaluate.load("davebulaval/meaningbert")
    bertscore    = evaluate.load("bertscore")

    CEFR = ["A1","A2","B1","B2","C1","C2"]
    L2I  = {l:i for i,l in enumerate(CEFR)}

    def cefr_labels(hyps, models, batch_size=BATCH_SIZE):
        p1 = models[0](hyps, batch_size=batch_size, truncation=True)
        p2 = models[1](hyps, batch_size=batch_size, truncation=True)
        p3 = models[2](hyps, batch_size=batch_size, truncation=True)
        def top1(x):
            if isinstance(x, dict): return x
            if isinstance(x, list) and x: return max(x, key=lambda d: d["score"])
        outs = []
        for d1, d2, d3 in zip(p1, p2, p3):
            best = max((top1(d1), top1(d2), top1(d3)), key=lambda d: d["score"])
            outs.append(best["label"].strip().upper())
        return outs

    def score_cefr(hyps, ref_lvls, models):
        gold  = [str(l).strip().upper() for l in ref_lvls]
        preds = [str(l).strip().upper() for l in cefr_labels(hyps, models, batch_size=BATCH_SIZE)]
        f1 = f1_score(gold, preds, average="weighted")
        t  = np.array([L2I[l] for l in gold])
        p  = np.array([L2I[l] for l in preds])
        adj  = (np.abs(t - p) <= 1).mean()
        rmse = root_mean_squared_error(t, p)
        return {"weighted_f1": round(float(f1),4),
                "adj_accuracy": round(float(adj),4),
                "rmse": round(float(rmse),4)}

    def score_meaningbert(hyps, refs):
        res = meaning_bert.compute(predictions=hyps, references=refs)
        return round(float(np.mean(res["scores"])) / 100.0, 4)

    def score_bertscore(hyps, refs, scoretype="f1"):
        res = bertscore.compute(references=refs, predictions=hyps, lang="en")
        return round(float(np.mean(res[scoretype])), 4)

    # ------------- Main ---------------------
    if not os.path.isfile(GOLD_FILE):
        raise FileNotFoundError(f"Gold file not found: {GOLD_FILE}")
    gold_orig, gold_ref, gold_tgt, gold_ids = read_gold(GOLD_FILE)

    if not os.path.isdir(SUBMISSIONS_DIR):
        raise FileNotFoundError(f"Submissions folder not found: {SUBMISSIONS_DIR}")

    team_dirs = sorted([d for d in os.listdir(SUBMISSIONS_DIR)
                        if os.path.isdir(os.path.join(SUBMISSIONS_DIR, d)) and not d.startswith(".")])

    results = []
    for team in team_dirs:
        team_path = os.path.join(SUBMISSIONS_DIR, team)
        run_files = sorted([f for f in os.listdir(team_path) if f.endswith(".jsonl")])
        if not run_files:
            print(f"[warn] No .jsonl files in {team_path}")
            continue
        for run in run_files:
            run_path = os.path.join(team_path, run)
            print(f"Evaluating {team}/{run} ...")
            hyps, sys_ids, num_instances = read_submission(run_path)

            aligned = align_intersection(hyps, sys_ids, gold_ids, gold_orig, gold_ref, gold_tgt)
            if aligned is None:
                print(f"[{team}/{run}] no overlap with gold; skipping.")
                row = {"modelname": run, "teamname": team,
                       "num_instances": num_instances,
                       "coverage_n": 0, "coverage_pct": 0.0,
                       "weighted_f1": "n/a", "adj_accuracy": "n/a", "rmse": "n/a",
                       "meaningbert-orig": "n/a", "bertscore-orig": "n/a",
                       "meaningbert-ref": "n/a", "bertscore-ref": "n/a"}
            else:
                if aligned["missing_ids"]:
                    print(f"[{team}/{run}] missing {len(aligned['missing_ids'])} ids.")
                if aligned["extra_ids"]:
                    print(f"[{team}/{run}] extra {len(aligned['extra_ids'])} ids (ignored).")
                hyps_i, orig_i, ref_i, tgt_i = aligned["hyps"], aligned["orig"], aligned["ref"], aligned["tgt"]
                cefr = score_cefr(hyps_i, tgt_i, [cefr_labeler1, cefr_labeler2, cefr_labeler3])
                mb_o = score_meaningbert(hyps_i, orig_i)
                bs_o = score_bertscore(hyps_i, orig_i, "f1")
                mb_r = score_meaningbert(hyps_i, ref_i)
                bs_r = score_bertscore(hyps_i, ref_i, "f1")
                row = {"modelname": run, "teamname": team,
                       "num_instances": num_instances,
                       "coverage_n": aligned["coverage_n"],
                       "coverage_pct": aligned["coverage_pct"],
                       "weighted_f1": cefr["weighted_f1"], "adj_accuracy": cefr["adj_accuracy"], "rmse": cefr["rmse"],
                       "meaningbert-orig": mb_o, "bertscore-orig": bs_o,
                       "meaningbert-ref": mb_r, "bertscore-ref": bs_r}
            results.append(row)

    df = pd.DataFrame(results)
    print("\n=== Results ===")
    print(df.to_string(index=False))
    df.to_excel("results.xlsx", index=False)
    print("\nSaved: results.xlsx")
    return (json,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Results Interpretation

    ### Coverage
    Both models: **74 instances, 100% coverage** — all IDs matched.

    ### CEFR Control (Difficulty Targeting)

    **Single-pass:**
    - weighted_f1 = 0.4917
    - adj_accuracy = 0.9595
    - rmse = 0.8542

    **Two-pass:**
    - weighted_f1 = 0.5001
    - adj_accuracy = 0.9595
    - rmse = 0.8463

    **Interpretation:**
    - Two-pass is slightly better on weighted_f1 (+0.0034) and rmse (-0.0079).
    - Adj_accuracy is identical (0.9595) — about 96% within ±1 CEFR level.
    - Both models are close on CEFR control; two-pass has a small edge.

    ### Meaning Preservation

    **Single-pass:**
    - meaningbert-orig = 0.8482
    - bertscore-orig = 0.9505
    - meaningbert-ref = 0.8275
    - bertscore-ref = 0.9485

    **Two-pass:**
    - meaningbert-orig = 0.8432
    - bertscore-orig = 0.9489
    - meaningbert-ref = 0.8280
    - bertscore-ref = 0.9486

    **Interpretation:**
    - Single-pass is slightly higher on meaningbert-orig (+0.005) and bertscore-orig (+0.0016).
    - Two-pass is slightly higher on meaningbert-ref (+0.0005) and bertscore-ref (+0.0001).
    - Differences are very small; both preserve meaning well.

    ### Overall Summary
    - Two-pass shows a small improvement in CEFR control (weighted_f1 and rmse).
    - Meaning preservation is similar; single-pass is slightly higher on original similarity.
    - The two-pass judge loop appears to help with level targeting without meaning loss.

    ### Practical Takeaway
    The two-pass approach with hallucination checking slightly improves CEFR targeting while maintaining meaning. The gains are modest, suggesting the single-pass model is already strong, and the judge loop provides a small refinement.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    df.to_csv("results.csv", index=False)
    """)
    return


if __name__ == "__main__":
    app.run()
