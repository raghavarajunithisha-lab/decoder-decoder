class Config:
    # --- Model & Data ---
    MODEL_CHOICE = "TinyLlama"        # "distilgpt2", "gpt2-medium", "TinyLlama", "Qwen"
    USE_TDA = True                    # True = enable TDA, False = disable
    DATA_PATH = "data/preprocessed_counselchat_data_df.csv"
    TEXT_COLS = ("questionText", "answerText")

    # --- Training ---
    EPOCHS = 100                      # Early stopping (patience 3) handles convergence
    BATCH_SIZE = 4                    # Increased from 1 — fewer steps per epoch, much faster
    GRAD_ACCUM = 2
    LR_QWEN = 1e-5
    LR_OTHER = 2e-5
    MAX_LEN_QWEN = 512               # Reduced from 1024 — saves memory on CPU
    MAX_LEN_OTHER = 256
    EARLY_STOPPING_PATIENCE = 3

    # --- Data Limiting ---
    MAX_SAMPLES = None                # Set to e.g. 200 for quick testing on CPU, None = use all

    # --- TDA Settings ---
    PCA_COMPONENTS = 250
    TDA_RESOLUTION = 50
    FASTTEXT_DIM = 100

    # --- Misc ---
    SEED = 42


cfg = Config()