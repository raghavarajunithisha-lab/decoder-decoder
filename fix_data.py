"""
Fix data quality issues in Decoder-Decoder datasets:
1. CounselChat: Drop 4 rows with null answerText
2. MentalChat: Drop 79 duplicate rows
3. NLPMentalHealth: Drop 6 null Response rows + 1,483 duplicate rows
4. MentalHealthChatbot: No changes needed (clean)

Creates backups before modifying.
"""
import pandas as pd
import os
import shutil

data_dir = r"e:\TDA\code\Encoder\Decoder-Decoder\data"

def fix_dataset(filename, text_cols, label):
    path = os.path.join(data_dir, filename)
    backup = path + ".bak"

    df = pd.read_csv(path)
    original_len = len(df)
    print(f"\n{'='*60}")
    print(f"  {label} ({filename})")
    print(f"{'='*60}")
    print(f"  Original rows: {original_len}")

    # 1. Drop rows where any text column is null
    before = len(df)
    df = df.dropna(subset=list(text_cols))
    nulls_dropped = before - len(df)
    if nulls_dropped:
        print(f"  Dropped {nulls_dropped} rows with null values")

    # 2. Drop rows where text columns are empty or whitespace-only
    before = len(df)
    for col in text_cols:
        df = df[df[col].astype(str).str.strip().str.len() > 0]
    empty_dropped = before - len(df)
    if empty_dropped:
        print(f"  Dropped {empty_dropped} rows with empty/whitespace text")

    # 3. Drop exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    dupes_dropped = before - len(df)
    if dupes_dropped:
        print(f"  Dropped {dupes_dropped} exact duplicate rows")

    # 4. Reset index
    df = df.reset_index(drop=True)

    total_dropped = original_len - len(df)
    print(f"  Final rows: {len(df)} (removed {total_dropped} total)")

    if total_dropped > 0:
        # Create backup
        shutil.copy2(path, backup)
        print(f"  Backup saved: {backup}")
        # Save cleaned data
        df.to_csv(path, index=False)
        print(f"  Cleaned file saved!")
    else:
        print(f"  No changes needed.")

    return df


# Fix all 4 datasets
fix_dataset(
    "preprocessed_counselchat_data_df.csv",
    ("questionText", "answerText"),
    "CounselChat"
)

fix_dataset(
    "preprocessed_MentalChat_df.csv",
    ("input", "output"),
    "MentalChat"
)

fix_dataset(
    "preprocessed_mental_health_chatbot.csv",
    ("human_input", "assistant_output"),
    "MentalHealthChatbot"
)

fix_dataset(
    "preprocessed_nlp_mental_health_df.csv",
    ("Context", "Response"),
    "NLPMentalHealth"
)

print("\n\nDone! All datasets cleaned.")
