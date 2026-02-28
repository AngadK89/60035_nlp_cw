import torch
import pandas as pd
import nlpaug.augmenter.word as naw
from transformers import BertTokenizer


# Patch missing method to handle versioning issue
BertTokenizer._convert_token_to_id = BertTokenizer.convert_tokens_to_ids # type: ignore

device = "cuda" if torch.cuda.is_available() else "cpu"


# Backtranslation (EN -> DE -> EN)
def backtranslate(texts, labels, filepath):
    print("\nRunning backtranslation (EN → DE → EN)")

    bt_aug = naw.BackTranslationAug(
        from_model_name="Helsinki-NLP/opus-mt-en-de",
        to_model_name="Helsinki-NLP/opus-mt-de-en",
        device=device,
    )

    bt_augmented_texts = bt_aug.augment(texts)

    # Flatten
    bt_augmented = [t[0] if isinstance(t, list) else t for t in bt_augmented_texts]


    df_bt = pd.DataFrame({"text": bt_augmented, "label": labels})
    df_bt.to_csv(filepath, index=False)
    print(f"Saved {len(df_bt)} backtranslated samples to train_backtranslate.csv")

    return df_bt

# Using Contextual Word Embeddings with BERT to substitute words
def substitute_words(texts, labels, filepath, top_k=5, aug_p=0.1):
    print("\nRunning contextual word embeddings (BERT substitute)")

    substitute_aug = naw.ContextualWordEmbsAug(
        model_path="bert-base-uncased",
        action="substitute",
        top_k=top_k,
        aug_p=aug_p,
        device=device,
    )

    cwe_augmented_texts = substitute_aug.augment(texts)

    # Flatten
    cwe_augmented = [t[0] if isinstance(t, list) else t for t in cwe_augmented_texts]


    df_cwe = pd.DataFrame({"text": cwe_augmented, "label": labels})
    df_cwe.to_csv(filepath, index=False)
    print(f"Saved {len(df_cwe)} CWE-augmented samples to train_cwe_bert.csv")

    return df_cwe

if __name__ == "__main__":
    df = pd.read_csv("data/train_data.csv")
    df_label1 = df[df["label"] == 1].reset_index(drop=True)
    texts = df_label1["text"].tolist()
    labels = df_label1["label"].tolist()

    print(f"Found {len(df_label1)} samples with label == 1.")

    backtranslate(texts, labels, "data/train_backtranslate.csv")
    substitute_words(texts, labels, "data/train_cwe_bert.csv")

    print("\nDone!")
