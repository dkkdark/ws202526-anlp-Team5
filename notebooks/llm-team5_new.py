# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "bert-score==0.3.13",
#     "evaluate==0.4.6",
#     "ipython==9.8.0",
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

__generated_with = "0.19.2"
app = marimo.App(
    width="medium",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell
def _():
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    from pathlib import Path
    import re
    import marimo as mo
    from typing import Tuple, List
    return ChatOpenAI, Path, PromptTemplate, mo


@app.cell
def _(mo):
    api_key = mo.ui.text_area(
        label="Enter openai api key:",
        value="",
        rows=3,
    )
    api_key
    return (api_key,)


@app.cell
def _(mo):
    mo.md(r"""
    ### A simple prompt without instructions
    """)
    return


@app.cell
def _(ChatOpenAI, PromptTemplate, api_key):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        api_key=api_key.value
    )


    rewrite_prompt = PromptTemplate.from_template("""
    Your task:
    Rewrite the sentence for CEFR level {level}.
    Keep the original meaning.

    Original:
    {sent}

    Output format: ONLY the simplified sentence
    """)

    rewrite_chain = rewrite_prompt | llm

    def simplify_sentence_with_simple_prompt(sentence: str, level: str) -> str:

        result = rewrite_chain.invoke({
            "sent": sentence,
            "level": level
        }).content.strip()

        return result

    text = (
        "Despite the fact that the weather was inclement, "
        "the terrestrial journey proceeded as originally scheduled."
    )

    print("Original: ", text)
    print("A2:", simplify_sentence_with_simple_prompt(text, "A2"))
    print("B1:", simplify_sentence_with_simple_prompt(text, "B1"))
    return (simplify_sentence_with_simple_prompt,)


@app.cell
def _(mo):
    mo.md(r"""
    ### The usage of llm with instruction prompt + ask to generate simplified sentences for A2-B2 levels at once + examples for each level
    """)
    return


@app.cell
def _(ChatOpenAI, Path, PromptTemplate, api_key):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        api_key=api_key.value
    )

    def load_vocab(path):
        return set(
            w.strip().lower()
            for w in Path(path).read_text().splitlines()
            if w.strip()
        )

    A2_VOCAB = load_vocab("a2_vocab.txt")
    B1_VOCAB = load_vocab("b1_vocab.txt")


    rewrite_prompt = PromptTemplate.from_template("""
    Your task:
    Rewrite the sentence for MULTIPLE CEFR levels.
    Keep the original meaning.

    Examples:

    Original (C1–C2):
    In light of the unforeseen logistical constraints, the organization elected to defer the implementation of the initiative indefinitely.

    A1:
    The group decided to start the plan later.

    A2:
    The organization decided to delay the plan because of problems.

    B1:
    The organization decided to delay the plan because of unexpected problems.

    B2:
    Because of unexpected logistical problems, the organization decided to delay the plan for some time.

    ---

    Original (C1–C2):
    Notwithstanding the comprehensive guidelines provided, a significant proportion of participants failed to adhere to the prescribed procedures.

    A1:
    Many people did not follow the rules.

    A2:
    Many people did not follow the rules that were given.

    B1:
    Many people did not follow the given rules, even though they were clear.

    B2:
    Even though clear rules were given, many participants did not follow the procedures.

    ---

    Original (C1–C2):
    The policy was formulated with the intention of enhancing operational efficiency while simultaneously mitigating redundant administrative processes.

    A1:
    The rule helps people work better.

    A2:
    The policy helps people work better and reduces extra work.

    B1:
    The policy was made to help people work more efficiently and reduce extra work.

    B2:
    The policy was designed to improve efficiency while reducing unnecessary administrative work.


    ---

    Original:
    {sent}

    Output format (STRICT):
    A2: ...
    B1: ...
    """)

    rewrite_chain = rewrite_prompt | llm

    def simplify_sentence(sentence: str, level: str) -> str:

        result = rewrite_chain.invoke({
            "sent": sentence
        }).content.strip()

        outputs = {}
        for line in result.splitlines():
            if ":" in line:
                lvl, txt = line.split(":", 1)
                outputs[lvl.strip()] = txt.strip()

        selected = outputs.get(level)
        if not selected:
            raise ValueError(f"Level {level} not found in model output")

        vocab = A2_VOCAB if level == "A2" else B1_VOCAB

        return selected

    text = (
        "Despite the fact that the weather was inclement, "
        "the terrestrial journey proceeded as originally scheduled."
    )

    print("Original: ", text)
    print("A2:", simplify_sentence(text, "A2"))
    print("B1:", simplify_sentence(text, "B1"))
    return (simplify_sentence,)


@app.cell
def _(Path, json, simplify_sentence):
    in_path = Path("tsar2025_test.jsonl")
    out_path = Path("results_top10.jsonl")

    outputs = []
    seen = 0

    with in_path.open("r", encoding="utf-8") as f:
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
                simplified = simplify_sentence(original, level)
            except Exception as e:
                simplified = original
                print(f"[warn] line={line_no} text_id={text_id} level={level}: {e}")

            outputs.append({"text_id": text_id, "simplified": simplified})

    with out_path.open("w", encoding="utf-8") as f:
        for obj in outputs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    out_path
    return


@app.cell
def _(mo):
    mo.md(r"""
    <b>Original:</b> Your dreams might show you all kinds of insights into things that you didn't realise you were thinking about. Common dreams like being able to fly or falling, your teeth falling out or having no clothes on in a public place probably mean something similar in most people. But the key to understanding exactly what they mean to you is to connect them to the events and feelings in your daily life

    <b>Simplified simple:</b> Your dreams can give you new ideas about things you didn't know you were thinking about. Common dreams, like flying, falling, losing your teeth, or being naked in public, usually have similar meanings for many people. To understand what they mean for you, it's important to link them to your daily experiences and feelings.

    <b>Simplified improved:</b> Your dreams can give you ideas about things you didn't know you were thinking about. Common dreams, like flying, falling, losing your teeth, or being naked in public, usually mean similar things for many people. But to really understand what they mean for you, you need to link them to what happens and how you feel in your daily life.


    <b>Original:</b> One of the most interesting cases of wild animals living in a city are the wild dogs of Moscow. In Moscow, there are approximately 35,000 wild dogs living on the streets. Some of the dogs were born wild, while others are pets that have been abandoned by their owners. Some dogs live alone and others live in packs. In 2010, scientists studied the dogs and found that the dogs have adapted remarkably successfully to urban life. They have learned that it is safer to cross the street with people and some dogs appear to understand traffic lights.

    <b>Simplified simple:</b> One interesting example of wild animals in a city is the wild dogs of Moscow. About 35,000 wild dogs live on the streets of Moscow. Some were born wild, and others were pets that were left behind by their owners. Some dogs live alone, while others live in groups. In 2010, scientists studied these dogs and found that they have adapted very well to city life. They have learned it is safer to cross the street with people, and some dogs seem to understand traffic lights.

    <b>Simplified improved:</b> One interesting case of wild animals living in a city is the wild dogs of Moscow. There are about 35,000 wild dogs on the streets of Moscow. Some dogs were born wild, and some were pets that people left behind. Some dogs live alone, and others live in groups. In 2010, scientists studied these dogs and found that they have adapted very well to city life. They have learned it is safer to cross the street with people, and some dogs seem to understand traffic lights.
    """)
    return


@app.cell
def _(Path, json, simplify_sentence_with_simple_prompt):
    in_path = Path("tsar2025_test.jsonl")
    out_path = Path("results_simple_prompt.jsonl")

    outputs = []
    seen = 0

    with in_path.open("r", encoding="utf-8") as f:
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
                simplified = simplify_sentence_with_simple_prompt(original, level)
            except Exception as e:
                simplified = original
                print(f"[warn] line={line_no} text_id={text_id} level={level}: {e}")

            outputs.append({"text_id": text_id, "simplified": simplified})

    with out_path.open("w", encoding="utf-8") as f:
        for obj in outputs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    out_path
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
    GOLD_FILE = "/__modal/volumes/vo-GRayTm3kOHyZhbFgF12rO9/tsar2025_test.jsonl"   # gold file next to this script
    SUBMISSIONS_DIR = "/__modal/volumes/vo-GRayTm3kOHyZhbFgF12rO9/submission"     # folder with team subfolders
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
    return df, json


@app.cell
def _(df):
    df.to_csv("results.csv", index=False)
    return


@app.cell
def _(mo):
    mo.md(r"""
    | modelname                                   | teamname  | num_instances | coverage_n | coverage_pct | weighted_f1 | adj_accuracy | rmse  | meaningbert-orig | bertscore-orig | meaningbert-ref | bertscore-ref |
    |--------------------------------------------|-----------|---------------|------------|---------------|-------------|--------------|-------|------------------|----------------|-----------------|---------------|
    | results_simple_prompt.jsonl                | team_test | 200           | 200        | 100.0         | 0.5776      | 0.975        | 0.7176 | 0.8452           | 0.9483         | 0.8447          | 0.9502        |
    | results_llm_without_vocab_coverage.jsonl   | team_test | 200           | 200        | 100.0         | 0.4770      | 0.935        | 0.9247 | 0.8682           | 0.9539         | 0.8383          | 0.9463        |
    """)
    return


if __name__ == "__main__":
    app.run()
