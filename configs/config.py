class Config:
    # --- Model & Data ---
    MODEL_CHOICE = "TinyLlama"        # "distilgpt2", "gpt2-medium", "TinyLlama", "Qwen"
    USE_TDA = True                    # True = enable TDA, False = disable
    DATA_PATH = "data/preprocessed_counselchat_data_df.csv"
    TEXT_COLS = ("questionText", "answerText")

    # --- Training ---
    EPOCHS = 100
    BATCH_SIZE = 1
    GRAD_ACCUM = 2
    LR_QWEN = 1e-5
    LR_OTHER = 2e-5
    MAX_LEN_QWEN = 1024
    MAX_LEN_OTHER = 256
    EARLY_STOPPING_PATIENCE = 3

    # --- TDA Settings ---
    PCA_COMPONENTS = 200
    TDA_RESOLUTION = 50
    FASTTEXT_DIM = 100

    # --- Misc ---
    SEED = 42


cfg = Config()
