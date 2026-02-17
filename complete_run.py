import warnings
warnings.filterwarnings("ignore")
import os, gc, numpy as np, pandas as pd, torch, evaluate, nltk
from datasets import Dataset
from sklearn.model_selection import train_test_split
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback
)
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

try:
    import gudhi
    from gudhi.representations import DiagramScaler, Landscape
    from gensim.models import FastText
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from nltk.tokenize import word_tokenize
except ImportError:
    pass

class ResearchLogger(TrainerCallback):
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs

    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        if logs is None: return 
        
        epoch = int(state.epoch)
        train_loss = logs.get("loss", "N/A")
        eval_loss = logs.get("eval_loss", "N/A")
        print(f"\nEpoch {epoch}/{self.total_epochs}")

        t_str = f"{train_loss:.4f}" if isinstance(train_loss, (float, int)) else train_loss
        v_str = f"{eval_loss:.4f}" if isinstance(eval_loss, (float, int)) else eval_loss
        
        print(f"\n[SUMMARY] Epoch {epoch}/{self.total_epochs} | Train Loss: {t_str} | Eval Loss: {v_str}")

def apply_tda_pipeline(train_df, test_df, text_cols):
    nltk.download("punkt_tab", quiet=True)
    
    def get_features(target_df, fit_objs=None):
        df_copy = target_df.copy()
        patterns = df_copy[text_cols[0]].astype(str) + " " + df_copy[text_cols[1]].astype(str)
        tokenized = patterns.apply(word_tokenize).tolist()
        
        if fit_objs is None:
            ft = FastText(sentences=tokenized, vector_size=100, window=5, min_count=1, workers=4)
        else:
            ft = fit_objs['ft']

        def diag_mean(s):
            words = s.split()
            embs = [ft.wv[w] for w in words if w in ft.wv]
            if len(embs) < 2: embs = [np.zeros(100), np.zeros(100)]
            rips = gudhi.RipsComplex(points=embs, max_edge_length=10)
            st = rips.create_simplex_tree(max_dimension=2)
            return [p[1] for p in st.persistence()], np.mean(embs, axis=0)

        res = [diag_mean(s) for s in patterns]
        diags, means = zip(*res)
        
        vectors = []
        for d in diags:
            try:
                D = np.array(d, dtype=float)
                proc = DiagramScaler(use=True, scalers=[([0, 1], StandardScaler())])
                LS = Landscape(resolution=50)
                L = np.array(LS(proc(D)), dtype=float).flatten()
                vectors.append(np.pad(L, (0, max(0, 50 - len(L))))[:50])
            except:
                vectors.append(np.zeros(50))
        
        tda_mat = np.vstack(vectors)
        if fit_objs is None:
            pca = PCA(n_components=10).fit(tda_mat)
            sc_m = StandardScaler().fit(np.array(means))
            sc_t = StandardScaler().fit(pca.transform(tda_mat))
            fit_objs = {'ft': ft, 'pca': pca, 'sc_m': sc_m, 'sc_t': sc_t}
        
        comb = np.hstack((fit_objs['sc_m'].transform(np.array(means)), 
                          fit_objs['sc_t'].transform(fit_objs['pca'].transform(tda_mat))))
        df_copy['tda_compact'] = list(comb[:, :10])
        return df_copy, fit_objs

    train_out, objs = get_features(train_df)
    test_out, _ = get_features(test_df, fit_objs=objs)
    return train_out, test_out

# --- 3. Configuration ---
MODELS_TO_RUN = ["distilgpt2", "gpt2-medium", "TinyLlama", "Qwen"]
TDA_MODES = [True, False]
DATASETS = [
    {"path": "data/preprocessed_counselchat_data_df.csv", "cols": ("questionText", "answerText"), "name": "CounselChat"},
    {"path": "data/preprocessed_mental_health_chatbot.csv", "cols": ("human_input", "assistant_output"), "name": "MentalHealthChatbot"},
    {"path": "data/preprocessed_MentalChat_df.csv", "cols": ("input", "output"), "name": "MentalChat"},
    {"path": "data/preprocessed_nlp_mental_health_df.csv", "cols": ("Context", "Response"), "name": "NLPMentalHealth"}
]

SEED = 42
EPOCHS = 2
BATCH_SIZE = 1
GRAD_ACCUM = 2

def compute_metrics_final(model, tokenizer, df, text_cols, use_tda):
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    samples = df.sample(min(10, len(df))).to_dict(orient="records")
    preds, refs = [], []
    model.eval()
    for ex in samples:
        tda_s = " ".join([f"<tda{i}:{v:.3f}>" for i, v in enumerate(ex.get('tda_compact', [0]*10))]) if use_tda else ""
        prompt = f"<|user|>: {ex[text_cols[0]]} {tda_s}\n<|assistant|>:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        preds.append(tokenizer.decode(out[0], skip_special_tokens=True))
        refs.append(ex[text_cols[1]])
    return bleu.compute(predictions=preds, references=refs)['bleu'], rouge.compute(predictions=preds, references=refs)['rougeL']

# --- 4. Main Loop ---
for ds_info in DATASETS:
    for m_key in MODELS_TO_RUN:
        for tda_on in TDA_MODES:
            print("\n" + "="*70)
            print(f"RUNNING: {m_key} | Dataset: {ds_info['name']} | TDA: {tda_on}")
            print("="*70)

            m_id = {"distilgpt2":"distilgpt2", "gpt2-medium":"gpt2-medium", 
                    "TinyLlama":"TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                    "Qwen":"Qwen/Qwen1.5-0.5B-Chat"}[m_key]
            
            df_raw = pd.read_csv(ds_info['path']).iloc[:100]
            train_df, test_df = train_test_split(df_raw, test_size=0.1, random_state=SEED)
            
            if tda_on:
                train_df, test_df = apply_tda_pipeline(train_df, test_df, ds_info['cols'])

            tokenizer = AutoTokenizer.from_pretrained(m_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token if "Qwen" not in m_key else "<|extra_pad|>"

            def format_fn(ex):
                tda_s = " ".join([f"<tda{i}:{v:.3f}>" for i, v in enumerate(ex.get('tda_compact', [0]*10))]) if tda_on else ""
                u_text = f"{ex[ds_info['cols'][0]]} {tda_s}".strip()
                if "Chat" in m_id or "Qwen" in m_id:
                    msgs = [{"role":"user","content":u_text}, {"role":"assistant","content":ex[ds_info['cols'][1]]}]
                    return {"text": tokenizer.apply_chat_template(msgs, tokenize=False)}
                return {"text": f"<|user|>: {u_text}\n<|assistant|>: {ex[ds_info['cols'][1]]}"}

            train_tok = Dataset.from_pandas(train_df).map(format_fn)
            test_tok = Dataset.from_pandas(test_df).map(format_fn)

            def tok_fn(batch):
                t = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)
                t["labels"] = [[(i if i != tokenizer.pad_token_id else -100) for i in s] for s in t["input_ids"]]
                return t

            train_tok = train_tok.map(tok_fn, batched=True, remove_columns=train_tok.column_names)
            test_tok = test_tok.map(tok_fn, batched=True, remove_columns=test_tok.column_names)

            model = AutoModelForCausalLM.from_pretrained(m_id, dtype=torch.float32, device_map="auto")
            
            if "Qwen" in m_id: 
                model.resize_token_embeddings(len(tokenizer))

            if m_key in ["TinyLlama", "Qwen"]:
                lora_cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, 
                                      target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
                model = get_peft_model(model, lora_cfg)
                print("Applied LoRA")

            args = TrainingArguments(
                output_dir=f"./results/{m_key}_{tda_on}",
                eval_strategy="epoch", save_strategy="epoch", logging_strategy="epoch",
                num_train_epochs=EPOCHS, per_device_train_batch_size=BATCH_SIZE,
                gradient_accumulation_steps=GRAD_ACCUM, fp16=torch.cuda.is_available(),
                load_best_model_at_end=True, report_to="none", disable_tqdm=True
            )

            trainer = Trainer(
                model=model, args=args, train_dataset=train_tok, eval_dataset=test_tok,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3), ResearchLogger(EPOCHS)]
            )
            
            trainer.train()

            b_val, r_val = compute_metrics_final(model, tokenizer, test_df, ds_info['cols'], tda_on)
            print(f"Results | BLEU: {b_val:.4f} | ROUGE-L: {r_val:.4f}\n" + "="*70)

            del model, trainer, tokenizer
            gc.collect()
            torch.cuda.empty_cache()