import warnings
warnings.filterwarnings("ignore")

from configs.config import cfg

import os, gc, numpy as np, pandas as pd, torch, evaluate, nltk
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model
from transformers import TrainerCallback

from transformers import DataCollatorWithPadding
# Optional imports (TDA)
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

hf_logging.set_verbosity_error()        # or set_verbosity_warning()
hf_logging.disable_progress_bar()



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

SEED = cfg.SEED
EARLY_STOPPING_PATIENCE = cfg.EARLY_STOPPING_PATIENCE

torch.manual_seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)



models = {
    "distilgpt2": "distilgpt2",
    "gpt2-medium": "gpt2-medium",
    "TinyLlama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen": "Qwen/Qwen1.5-0.5B-Chat"
}

model_name = models[MODEL_CHOICE]

print(f"\n Fine-tuning: {MODEL_CHOICE} -> {model_name} (TDA = {USE_TDA})")



df = pd.read_csv(DATA_PATH)
assert all(c in df.columns for c in TEXT_COLS), f"Dataset must contain {TEXT_COLS}"

train_df, test_df = train_test_split(df, test_size=0.1, random_state=SEED)
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

print(f"Loaded data | Train: {len(train_dataset)} | Test: {len(test_dataset)}")



if USE_TDA:
    nltk.download("punkt_tab", quiet=True)

    df['patterns'] = df[TEXT_COLS[0]].astype(str) + " " + df[TEXT_COLS[1]].astype(str)
    df['tokenized_patterns'] = df['patterns'].apply(word_tokenize)

    print("Training FastText model...")
    ft_model = FastText(sentences=df['tokenized_patterns'].tolist(),
                        vector_size=FASTTEXT_DIM, window=5, min_count=1, workers=4)

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
    for sentence in df['patterns']:
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

    pca = PCA(n_components=min(PCA_COMPONENTS, tda_matrix.shape[1]))
    tda_pca = pca.fit_transform(tda_matrix)

    means_scaled = StandardScaler().fit_transform(means)
    tda_scaled = StandardScaler().fit_transform(tda_pca)

    combined = np.hstack((means_scaled, tda_scaled))
    df['tda_compact'] = list(combined[:, :min(10, combined.shape[1])])

    print("TDA features computed:", combined.shape)



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
        messages = [
            {"role": "user", "content": example[TEXT_COLS[0]]},
            {"role": "assistant", "content": example[TEXT_COLS[1]]},
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



model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32, device_map="auto")

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

training_args = TrainingArguments(
    output_dir=f"./{MODEL_CHOICE}-TDA-{USE_TDA}",
    eval_strategy="epoch",
    logging_strategy="epoch",      # Changed from "epoch" to "steps"
    disable_tqdm=True,             # THIS STOPS THE SCROLLING PROGRESS BAR
    save_strategy="epoch",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    fp16=use_fp16,
    learning_rate=LR,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,
    report_to="none"
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


class EpochBeginPrinter(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        # state.epoch is a float; cast to int and +1 for human‑friendly display
        if state.epoch is not None:
            print(f"\n========== Epoch {int(state.epoch) + 1} ==========\n")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=test_tok,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE),
               EpochBeginPrinter()]
)

trainer.train()


bleu = evaluate.load("bleu")
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
                max_length=256,
                do_sample=True,
                top_p=0.9,
                temperature=0.8
            )

        pred = tokenizer.decode(output[0], skip_special_tokens=True)
        preds.append(pred)
        refs.append(example[TEXT_COLS[1]])

    bleu_score = bleu.compute(predictions=preds, references=refs)["bleu"]
    rouge_score = rouge.compute(predictions=preds, references=refs)["rougeL"]

    return bleu_score, rouge_score


eval_loss = trainer.evaluate()["eval_loss"]
ppl = np.exp(eval_loss)

bleu_s, rouge_s = compute_metrics(model, tokenizer, test_df)

print(f"\n {MODEL_CHOICE} (TDA={USE_TDA}) Evaluation:")
print(f"Perplexity: {ppl:.2f}")
print(f"BLEU: {bleu_s:.3f} | ROUGE-L: {rouge_s:.3f}")