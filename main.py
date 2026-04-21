import warnings
warnings.filterwarnings("ignore")

from configs.config import cfg

import os, gc, time, numpy as np, pandas as pd, torch, evaluate, nltk
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model
from transformers import TrainerCallback, default_data_collator

try:
    import gudhi
    from gudhi.representations import DiagramSelector, DiagramScaler, Clamping, Landscape
    from gensim.models import FastText
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from nltk.tokenize import word_tokenize
except ImportError:
    pass

from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()


# ==================== CONFIGURATION ====================
MODEL_CHOICE = cfg.MODEL_CHOICE
USE_TDA = cfg.USE_TDA

DATA_PATH = cfg.DATA_PATH
TEXT_COLS = cfg.TEXT_COLS

EPOCHS = cfg.EPOCHS
BATCH_SIZE = cfg.BATCH_SIZE
GRAD_ACCUM = cfg.GRAD_ACCUM
LR = cfg.LR_QWEN if MODEL_CHOICE == "Qwen" else cfg.LR_OTHER

MAX_LEN = cfg.MAX_LEN_QWEN if MODEL_CHOICE == "Qwen" else cfg.MAX_LEN_OTHER

PCA_COMPONENTS = cfg.PCA_COMPONENTS
TDA_RESOLUTION = cfg.TDA_RESOLUTION
FASTTEXT_DIM = cfg.FASTTEXT_DIM

MAX_SAMPLES = cfg.MAX_SAMPLES

SEED = cfg.SEED
EARLY_STOPPING_PATIENCE = cfg.EARLY_STOPPING_PATIENCE

torch.manual_seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


# ==================== MODEL REGISTRY ====================
models = {
    "distilgpt2": "distilgpt2",
    "gpt2-medium": "gpt2-medium",
    "TinyLlama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen": "Qwen/Qwen1.5-0.5B-Chat"
}

model_name = models[MODEL_CHOICE]

print(f"\n Fine-tuning: {MODEL_CHOICE} -> {model_name} (TDA = {USE_TDA})")


# ==================== LOAD DATA ====================
df = pd.read_csv(DATA_PATH)
assert all(c in df.columns for c in TEXT_COLS), f"Dataset must contain {TEXT_COLS}"

# Apply sample limit for quick CPU testing
if MAX_SAMPLES is not None:
    df = df.iloc[:MAX_SAMPLES]
    print(f"[DEBUG] Limited dataset to {MAX_SAMPLES} samples for fast iteration")

# Split into train/test -- DON'T create HF Datasets yet (TDA adds columns first)
train_df, test_df = train_test_split(df, test_size=0.1, random_state=SEED)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"Loaded data | Train: {len(train_df)} | Test: {len(test_df)}")


# ==================== TDA PIPELINE (FIXED: fit on train only) ====================
if USE_TDA:
    nltk.download("punkt_tab", quiet=True)
    print("Computing TDA features...")
    tda_start = time.time()

    def compute_tda_features(target_df, ft_model=None, pca_obj=None,
                             scaler_mean=None, scaler_tda=None, fit=False):
        """
        Compute TDA features for a dataframe.
        fit=True: fits FastText/PCA/Scalers on this data, returns fitted objects.
        fit=False: uses provided fitted objects to only transform (no leakage).
        """
        patterns = target_df[TEXT_COLS[0]].astype(str) + " " + target_df[TEXT_COLS[1]].astype(str)
        tokenized = patterns.apply(word_tokenize).tolist()

        # Train or reuse FastText
        if fit:
            ft_model = FastText(sentences=tokenized, vector_size=FASTTEXT_DIM,
                                window=5, min_count=1, workers=4)

        def sentence_embedding_and_diagram(sentence):
            words = sentence.split()
            embeddings = [ft_model.wv[w] for w in words if w in ft_model.wv]
            if len(embeddings) == 0:
                embeddings = [np.zeros(FASTTEXT_DIM)]
            if len(embeddings) == 1:
                embeddings.append(np.zeros(FASTTEXT_DIM))

            rips = gudhi.RipsComplex(points=embeddings, max_edge_length=10)
            st = rips.create_simplex_tree(max_dimension=2)
            diag = st.persistence()
            diagram_points = [p[1] for p in diag]
            mean_emb = np.mean(embeddings, axis=0)
            return diagram_points, mean_emb

        diagrams, means = [], []
        for sentence in patterns:
            diag, mean_emb = sentence_embedding_and_diagram(sentence)
            diagrams.append(diag)
            means.append(mean_emb)
        means = np.array(means)

        vectors = []
        for diag in diagrams:
            if not diag:
                vectors.append(np.zeros(TDA_RESOLUTION))
                continue
            try:
                D = np.array(diag, dtype=float)
                proc1 = DiagramSelector(use=True, point_type="finite")
                proc2 = DiagramScaler(use=True, scalers=[([0, 1], StandardScaler())])
                proc3 = DiagramScaler(use=True, scalers=[([1], Clamping(maximum=0.9))])
                Dp = proc3(proc2(proc1(D)))
                LS = Landscape(resolution=TDA_RESOLUTION)
                L = np.array(LS(Dp), dtype=float).flatten()
                L = np.pad(L, (0, max(0, TDA_RESOLUTION - len(L))))[:TDA_RESOLUTION]
                vectors.append(L)
            except:
                vectors.append(np.zeros(TDA_RESOLUTION))

        tda_matrix = np.vstack(vectors)

        # Fit or transform PCA and scalers
        if fit:
            n_comp = min(PCA_COMPONENTS, tda_matrix.shape[1])
            pca_obj = PCA(n_components=n_comp)
            tda_pca = pca_obj.fit_transform(tda_matrix)
            scaler_mean = StandardScaler().fit(means)
            scaler_tda = StandardScaler().fit(tda_pca)
        else:
            tda_pca = pca_obj.transform(tda_matrix)

        means_scaled = scaler_mean.transform(means)
        tda_scaled = scaler_tda.transform(tda_pca)

        combined = np.hstack((means_scaled, tda_scaled))
        result_df = target_df.copy()
        result_df['tda_compact'] = list(combined[:, :min(10, combined.shape[1])])

        return result_df, ft_model, pca_obj, scaler_mean, scaler_tda

    # FIT on train data only, then TRANSFORM test data (no data leakage)
    print("  Fitting TDA pipeline on TRAIN set...")
    train_df, ft_model, pca_obj, sc_mean, sc_tda = compute_tda_features(train_df, fit=True)
    print("  Transforming TEST set with fitted objects...")
    test_df, _, _, _, _ = compute_tda_features(
        test_df, ft_model=ft_model, pca_obj=pca_obj,
        scaler_mean=sc_mean, scaler_tda=sc_tda, fit=False
    )
    print(f"TDA features computed in {time.time() - tda_start:.1f}s")


# ==================== CREATE HF DATASETS (AFTER TDA columns exist) ====================
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)


# ==================== TOKENIZER ====================
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

if "qwen" in model_name.lower():
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|extra_pad|>'})
    model_resize_needed = True
else:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_resize_needed = False


def tda_to_text(vec):
    return " ".join([f"<tda{i}:{float(v):.3f}>" for i, v in enumerate(vec)])


if "qwen" in model_name.lower():

    def format_prompt(example):
        user_content = str(example[TEXT_COLS[0]])
        if USE_TDA:
            tda_vec = example.get("tda_compact", [0.0] * 10)
            user_content = f"{user_content} {tda_to_text(tda_vec)}"
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": str(example[TEXT_COLS[1]])},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

else:

    def format_prompt(example):
        if USE_TDA:
            tda_vec = example.get("tda_compact", [0.0] * 10)
            tda_str = tda_to_text(tda_vec)
            text = f"<|user|>: {example[TEXT_COLS[0]]} {tda_str}\n<|assistant|>: {example[TEXT_COLS[1]]}"
        else:
            text = f"<|user|>: {example[TEXT_COLS[0]]}\n<|assistant|>: {example[TEXT_COLS[1]]}"

        return {"text": text}


train_tok = train_dataset.map(format_prompt)
test_tok = test_dataset.map(format_prompt)


def tokenize_fn(batch):
    tokens = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)
    tokens["labels"] = tokens["input_ids"].copy()

    tokens["labels"] = [
        [(tid if tid != tokenizer.pad_token_id else -100) for tid in ids]
        for ids in tokens["labels"]
    ]
    return tokens


train_tok = train_tok.map(tokenize_fn, batched=True, remove_columns=train_tok.column_names)
test_tok = test_tok.map(tokenize_fn, batched=True, remove_columns=test_tok.column_names)

train_tok.set_format("torch")
test_tok.set_format("torch")

print("Tokenization ready.")


# ==================== MODEL ====================
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

if model_resize_needed:
    model.resize_token_embeddings(len(tokenizer))

if MODEL_CHOICE in ["Mistral", "TinyLlama"]:
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)
    print("Applied LoRA")


use_fp16 = torch.cuda.is_available()

# ==================== TRAINING WITH TIMING ====================

class TimedEpochPrinter(TrainerCallback):
    """Prints epoch number and elapsed time per epoch for progress monitoring."""
    def __init__(self):
        self.epoch_start = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start = time.time()
        if state.epoch is not None:
            print(f"\n========== Epoch {int(state.epoch) + 1}/{int(args.num_train_epochs)} ==========")

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.epoch_start is not None:
            elapsed = time.time() - self.epoch_start
            print(f"  Epoch completed in {elapsed:.1f}s")


training_args = TrainingArguments(
    output_dir=f"./{MODEL_CHOICE}-TDA-{USE_TDA}",
    eval_strategy="epoch",
    logging_strategy="epoch",
    disable_tqdm=True,
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=1,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    fp16=use_fp16,
    learning_rate=LR,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=test_tok,
    data_collator=default_data_collator,
    callbacks=[TimedEpochPrinter(), EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
)

print(f"\nStarting training... ({len(train_tok)} samples, batch_size={BATCH_SIZE}, epochs={EPOCHS})")
print(f"Estimated steps per epoch: {len(train_tok) // BATCH_SIZE // GRAD_ACCUM}")
train_start = time.time()
trainer.train()
print(f"\nTraining completed in {(time.time() - train_start) / 60:.1f} minutes")


# ==================== EVALUATION ====================
bertscore = evaluate.load("bertscore")
rouge = evaluate.load("rouge")


def compute_metrics(model, tokenizer, df):
    preds, refs = [], []
    model.eval()

    samples = df.sample(min(10, len(df))).to_dict(orient="records")

    for example in samples:
        if "qwen" in model_name.lower():
            messages = [{"role": "user", "content": example[TEXT_COLS[0]]}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"<|user|>: {example[TEXT_COLS[0]]}\n<|assistant|>:"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                top_p=0.9,
                temperature=0.8
            )

        gen_tokens = output[0][inputs["input_ids"].shape[-1]:]
        pred = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip() or " "
        preds.append(pred)
        refs.append(str(example[TEXT_COLS[1]]).strip() or " ")

    b_results = bertscore.compute(predictions=preds, references=refs, lang="en")
    bert_f1 = float(np.mean(b_results["f1"]))
    rouge_score = rouge.compute(predictions=preds, references=refs)["rougeL"]

    return bert_f1, rouge_score


eval_loss = trainer.evaluate()["eval_loss"]
ppl = np.exp(eval_loss)

bert_s, rouge_s = compute_metrics(model, tokenizer, test_df)

print(f"\n {MODEL_CHOICE} (TDA={USE_TDA}) Evaluation:")
print(f"Perplexity: {ppl:.2f}")
print(f"BERTScore: {bert_s:.3f} | ROUGE-L: {rouge_s:.3f}")
