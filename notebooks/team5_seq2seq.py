# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "accelerate==1.12.0",
#     "bert-score==0.3.13",
#     "datasets==4.4.1",
#     "evaluate==0.4.6",
#     "huggingface-hub==0.36.0",
#     "marimo",
#     "matplotlib==3.10.8",
#     "nltk==3.9.2",
#     "numpy==2.2.6",
#     "pandas==2.3.3",
#     "sacrebleu==2.5.1",
#     "sacremoses==0.1.1",
#     "scikit-learn==1.8.0",
#     "torch==2.9.1",
#     "transformers[torch]==4.57.3",
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
    import marimo as mo
    import torch
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrainerCallback,
        DataCollatorForSeq2Seq,
    )
    from datasets import Dataset
    import matplotlib.pyplot as plt
    from pathlib import Path
    import warnings

    warnings.filterwarnings("ignore")
    return (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Dataset,
        Path,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrainerCallback,
        mo,
        plt,
        torch,
    )


@app.cell
def _(Dataset):
    def load_asset_data(asset_folder_path, split="valid"):
        """Load ASSET dataset from folder path and return HuggingFace Dataset"""
        import os

        src_sentences = []
        tgt_sentences = []


        for file_name in os.listdir(asset_folder_path):
            if file_name.endswith(f".{split}.orig"):
                base_name = file_name[: -(len(split) + 6)]  # Remove .split.orig extension
                orig_path = os.path.join(asset_folder_path, file_name)

                with open(orig_path, "r", encoding="utf-8") as f:
                    orig_sentences = [line.strip() for line in f if line.strip()]

                simp_files = [
                    os.path.join(asset_folder_path, simp_file_name)
                    for simp_file_name in os.listdir(asset_folder_path)
                    if simp_file_name.startswith(base_name) and f".{split}.simp." in simp_file_name
                ]

                simp_sentences_list = []
                for simp_file in simp_files:
                    with open(simp_file, "r", encoding="utf-8") as f:
                        simp_sentences = [line.strip() for line in f if line.strip()]
                        simp_sentences_list.append(simp_sentences)


                for i, orig_sentence in enumerate(orig_sentences):
                    for simp_sentences in simp_sentences_list:
                        if i < len(simp_sentences):
                            src_sentences.append("Rewrite the text to be much simpler. Shorten the sentence and reduce complexity. Do NOT keep the same structure: " + orig_sentence)
                            tgt_sentences.append(simp_sentences[i])

        if not src_sentences or not tgt_sentences:
            print("Warning: No data loaded. Check the folder structure and file naming conventions.")

        # Create HuggingFace Dataset
        data_dict = {
            "source": src_sentences,
            "target": tgt_sentences,
        }

        return Dataset.from_dict(data_dict)
    return (load_asset_data,)


@app.cell
def _(AutoTokenizer, load_asset_data, mo):
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", legacy=False)

    train_dataset_t5 = load_asset_data("asset", split="valid")
    test_dataset_t5 = load_asset_data("asset", split="test")

    mo.md(f"""
    ### Dataset Loaded!

    - **Training samples**: {len(train_dataset_t5):,}
    - **Test samples**: {len(test_dataset_t5):,}

    **Example pair:**
    - Complex: {train_dataset_t5[0]['source']}
    - Simple: {train_dataset_t5[0]['target']}
    """)
    return test_dataset_t5, tokenizer, train_dataset_t5


@app.cell
def _(tokenizer):
    def preprocess_function(examples, max_length=128):
        # Tokenize the source
        model_inputs = tokenizer(
            examples["source"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors=None, 
        )

        # Tokenize the targets
        labels = tokenizer(
            examples["target"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )

        cleaned_labels = []
        for label_seq in labels["input_ids"]:
            cleaned_labels.append([
                int(token_id) if token_id != tokenizer.pad_token_id else -100 
                for token_id in label_seq
            ])

        model_inputs["labels"] = cleaned_labels
        return model_inputs
    return (preprocess_function,)


@app.cell
def _(preprocess_function, test_dataset_t5, train_dataset_t5):
    tokenized_train = train_dataset_t5.map(
        preprocess_function, batched=True, remove_columns=["source", "target"]
    )
    tokenized_test = test_dataset_t5.map(
        preprocess_function, batched=True, remove_columns=["source", "target"]
    )

    print(f"Tokenized training samples: {len(tokenized_train):,}")
    print(f"Tokenized test samples: {len(tokenized_test):,}")
    return tokenized_test, tokenized_train


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="Start Training")
    train_button
    return (train_button,)


@app.cell
def _(test_dataset_t5, tokenizer):
    import evaluate
    import numpy as np

    sari_metric = evaluate.load("sari")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        sources = test_dataset_t5["source"] 

        formatted_refs = [[label] for label in decoded_labels]

        sari_score = sari_metric.compute(
            sources=sources[:len(decoded_preds)], # Match length if batching differs
            predictions=decoded_preds,
            references=formatted_refs
        )

        # Calculate length ratio for monitoring
        length_ratios = [len(p.split()) / max(len(l.split()), 1) for p, l in zip(decoded_preds, decoded_labels)]

        return {
            "sari": sari_score["sari"],
            "length_ratio": float(np.mean(length_ratios))
        }
    return


@app.cell
def _(
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    plt,
    tokenized_test,
    tokenized_train,
    tokenizer,
    torch,
    train_button,
):
    if torch.cuda.is_available():
        device_t5 = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device_t5 = torch.device("mps")
    else:
        device_t5 = torch.device("cpu")

    # New parameters
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_ratio=0.05,
        learning_rate=2e-5,
        fp16=False,
        predict_with_generate=True,
        logging_dir="./logs",
        logging_steps=200,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
    )

    model_base = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    model_base = model_base.to(device_t5)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model_base,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )

    class LossCallback(TrainerCallback):
        def __init__(self):
            self.train_losses = []
            self.eval_losses = []
            self.train_steps = []
            self.eval_steps = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                if "loss" in logs:
                    self.train_losses.append(logs["loss"])
                    self.train_steps.append(state.global_step)
                if "eval_loss" in logs:
                    self.eval_losses.append(logs["eval_loss"])
                    self.eval_steps.append(state.global_step)

    loss_callback = LossCallback()

    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model_base,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )

    trainer = Seq2SeqTrainer(
        model=model_base,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=collator,
        callbacks=[loss_callback],
    )

    if train_button.value:
        fig, ax = plt.subplots(figsize=(10, 6))

        train_result = trainer.train()

        eval_result = trainer.evaluate()

        ax.clear()
        if loss_callback.train_losses:
            ax.plot(loss_callback.train_steps, loss_callback.train_losses, 'b-', label='Train Loss', linewidth=2, alpha=0.7)
        if loss_callback.eval_losses:
            ax.plot(loss_callback.eval_steps, loss_callback.eval_losses, 'r-', label='Eval Loss', linewidth=2, marker='o', markersize=6)

        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Loss")
        ax.set_title("T5 Training Progress")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        print(f"""
        ### Training Complete!

        **Final Training Loss**: {train_result.training_loss:.4f}
        **Final Eval Loss**: {eval_result['eval_loss']:.4f}

        """)
    return device_t5, trainer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The flan-t5-base model was teached for 10 epochs. Bellow we can see that evaluation loss stopped to reduce after 6 epoch, that's why I saved that model into Hugging face.

    Epoch	Training Loss	Validation Loss

    1	0.840300	0.921890

    2	0.781500	0.906228

    3	0.798300	0.900891

    4	0.782200	0.897823

    5	0.765800	0.897621

    6	0.726000	0.900363

    7	0.686800	0.900462

    8	0.700500	0.901818

    9	0.702600	0.901202

    10	0.698900	0.901644
    """)
    return


@app.cell
def _(mo):
    mo.image(src="image.png")
    return


@app.cell
def _(Path, tokenizer, trainer):
    save_path = Path("./fine_tuned_t5_simplification_large12")
    save_path.mkdir(exist_ok=True)

    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    return


@app.cell
def _(mo):
    test_input = mo.ui.text_area(
        label="Enter a complex sentence to simplify:",
        value="A Georgian inscription around the drum attests his name.",
        rows=3,
    )
    test_input
    return (test_input,)


@app.cell
def _(AutoModelForSeq2SeqLM, device_t5, mo, test_input, tokenizer, torch):
    def simplify_with_t5(text, model, tokenizer, device, max_length=128):
        """Simplify text using T5 model"""
        model.eval()

        input_text = "Simplify: " + text

        input_ids = tokenizer(
            input_text,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids, max_length=max_length, num_beams=4, early_stopping=True
            )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    model_base = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    model_base.to(device_t5)

    input_text_test = test_input.value.strip()

    # Our fine-tuned model
    simplified_ours = simplify_with_t5(
        input_text_test, model_base, tokenizer, device_t5
    )
    mo.md(f"""
    ### Model Comparison

    **Original (Complex):**
    > {input_text_test}

    ---

    | Model | Simplified Output |
    |-------|-------------------|
    | **Our Fine-tuned T5-base** | {simplified_ours} |
    """)
    return


@app.cell
def _(AutoModelForSeq2SeqLM, AutoTokenizer, torch):
    if torch.cuda.is_available():
        device2 = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device2 = torch.device("mps")
    else:
        device2 = torch.device("cpu")

    checkpoint_path = "dkkdark/t5_base_simlification"  
    model2 = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
    tokenizer_2 = AutoTokenizer.from_pretrained(checkpoint_path)
    model2.to(device2)
    model2.eval()

    def simplify_with_t5_finetuned(text, model2, tokenizer_2, device2, max_length=128):
        """Simplify text using T5 model"""
        model2.eval()

        input_text = "Simplify: " + text

        input_ids = tokenizer_2(
            input_text,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device2)

        with torch.no_grad():
            outputs = model2.generate(
                input_ids, 
                max_length=max_length, 
                num_beams=4, 
                early_stopping=True
            )

        return tokenizer_2.decode(outputs[0], skip_special_tokens=True)
    return device2, model2, simplify_with_t5_finetuned, tokenizer_2


@app.cell
def _(device2, model2, simplify_with_t5_finetuned, tokenizer_2):
    texts = [
        "The photosynthetic process involves the conversion of light energy into chemical energy, primarily through the absorption of photons by chlorophyll pigments.",
        "The nocturnal habits of the feline species allow them to navigate low-light environments with significant dexterity.",
        "Despite the fact that the weather was inclement, the terrestrial journey proceeded as originally scheduled.",
        "The implementation of the new fiscal policy elicited a variety of reactions from the local constituency."
    ]

    for txt in texts:
        simplified_txt = simplify_with_t5_finetuned(txt, model2, tokenizer_2, device2)
        print(f"---")
        print(f"Original:   {txt}")
        print(f"Simplified: {simplified_txt}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###This examples show that the model simplifies text, however it doesn't work ideally



    Original:   The photosynthetic process involves the conversion of light energy into chemical energy, primarily through the absorption of photons by chlorophyll pigments.

    Simplified: The photosynthetic process involves the conversion of light energy into chemical energy, mainly through the absorption of photons by chlorophyll pigments.

    --

    Original:   The nocturnal habits of the feline species allow them to navigate low-light environments with significant dexterity.

    Simplified: The nocturnal habits of the feline species allow them to navigate low-light environments with great dexterity.

    --

    Original:   Despite the fact that the weather was inclement, the terrestrial journey proceeded as originally scheduled.

    Simplified: Although the weather was inclement, the terrestrial journey continued as planned.

    --

    Original:   The implementation of the new fiscal policy elicited a variety of reactions from the local constituency.

    Simplified: The new fiscal policy elicited a variety of reactions from the local constituency.
    """)
    return


@app.cell
def _():
    # Below we evaluate the finetuned and base model
    return


@app.cell
def _(Path, device2, model2, simplify_with_t5_finetuned, tokenizer_2):
    import json

    in_path = Path("tsar2025_test.jsonl")
    out_path = Path("results_finetuned_seqtoseq.jsonl")

    outputs = []

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
                simplified = simplify_with_t5_finetuned(original, model2, tokenizer_2, device2)
            except Exception as e:
                simplified = original
                print(f"[warn] line={line_no} text_id={text_id} level={level}: {e}")

            outputs.append({"text_id": text_id, "simplified": simplified})

    with out_path.open("w", encoding="utf-8") as f:
        for obj in outputs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    out_path
    return (json,)


@app.cell
def _(json, torch):
    import os, random
    import numpy as np
    import pandas as pd
    from sklearn.metrics import f1_score, root_mean_squared_error
    from transformers import pipeline # IMPORTANT: Please ensure your transformers version is v4.55
    import evaluate
    GOLD_FILE = "/__modal/volumes/vo-KdoTRbHGUpf2m21hP3ZZ22/content/tsar2025_test.jsonl"   # gold file next to this script
    SUBMISSIONS_DIR = "/__modal/volumes/vo-KdoTRbHGUpf2m21hP3ZZ22/content/submission"     # folder with team subfolders"
    SEED = 42                           # for reproducibility
    BATCH_SIZE = 32                      # adjust for your GPU

    # ---------------- Seed ------------------
    random.seed(SEED)
    np.random.seed(SEED)
    try:

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
    return (df,)


@app.cell
def _(df):
    df.to_csv("results.csv", index=False)
    return


@app.cell
def _(AutoModelForSeq2SeqLM, AutoTokenizer, torch):
    if torch.cuda.is_available():
        device_large = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device_large = torch.device("mps")
    else:
        device_large = torch.device("cpu")

    checkpoint_large = "google/flan-t5-large"
    model_large = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_large)
    tokenizer_large = AutoTokenizer.from_pretrained(checkpoint_large)
    model_large.to(device_large)
    model_large.eval()

    def simplify_with_t5_large(text, model_large, tokenizer_large, device_large, max_length=128):
        """Simplify text using T5-large (non-finetuned)"""
        model_large.eval()

        input_text = "Simplify: " + text

        input_ids = tokenizer_large(
            input_text,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device_large)

        with torch.no_grad():
            outputs = model_large.generate(
                input_ids,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )

        return tokenizer_large.decode(outputs[0], skip_special_tokens=True)
    return device_large, model_large, simplify_with_t5_large, tokenizer_large


@app.cell
def _(
    Path,
    device_large,
    model_large,
    simplify_with_t5_large,
    tokenizer_large,
):
    import json


    in_path = Path("tsar2025_test.jsonl")
    out_path = Path("results_base_seqtoseq.jsonl")

    outputs = []

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
                simplified = simplify_with_t5_large(original, model_large, tokenizer_large, device_large)
            except Exception as e:
                simplified = original
                print(f"[warn] line={line_no} text_id={text_id} level={level}: {e}")

            outputs.append({"text_id": text_id, "simplified": simplified})

    with out_path.open("w", encoding="utf-8") as f:
        for obj in outputs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    out_path
    return (json,)


@app.cell
def _(mo):
    mo.md(r"""
    | modelname                         | teamname | num_instances | coverage_n | coverage_pct | weighted_f1 | adj_accuracy | rmse   | meaningbert-orig | bertscore-orig | meaningbert-ref | bertscore-ref |
    |----------------------------------|----------|---------------|------------|--------------|-------------|--------------|--------|------------------|----------------|------------------|---------------|
    | results_base_seqtoseq.jsonl       | team5    | 200           | 200        | 100.0        | 0.4451      | 0.91         | 0.9274 | 0.6302           | 0.8946         | 0.5774           | 0.8807        |
    | results_finetuned_seqtoseq.jsonl  | team5    | 200           | 200        | 100.0        | 0.3159      | 0.735        | 1.3115 | 0.9417           | 0.9867         | 0.7967           | 0.9349        |
    """)
    return


if __name__ == "__main__":
    app.run()
