import torch
import polars as pl
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os
import gc

TRANSCRIPT_FILE = "src/ingestion/data/backup/temp_transcripts.parquet"
OUTPUT_DIR = "src/ingestion/data/finbert_tx"
INFERENCE_BATCH_SIZE = 64

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def process_symbol(symbol, tx_lz, tokenizer, model, device):
    print(f"Processing {symbol}")

    symbol_data = tx_lz.filter(pl.col("symbol") == symbol).collect()

    if symbol_data.is_empty():
        print(f"  No data for {symbol}")
        return False

    texts = symbol_data["text"].to_list()
    num_rows = len(texts)
    print(f"  Rows to process: {num_rows}")

    all_predictions = []

    with torch.no_grad():
        for i in tqdm(range(0, num_rows, INFERENCE_BATCH_SIZE), desc=f"  Inference for {symbol}", unit="batch"):
            batch_texts = texts[i : i + INFERENCE_BATCH_SIZE]

            encoding = tokenizer(
                batch_texts,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )

            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            all_predictions.extend(probs.cpu().numpy())

    symbol_data = symbol_data.with_columns(
        pl.Series("sentiment_probs", all_predictions)
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    symbol_data.write_parquet(os.path.join(OUTPUT_DIR, f"{symbol}.parquet"))

    del symbol_data, all_predictions
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return True

def main():
    device = get_device()
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.to(device)
    if device.type != "cpu":
        model.half()
    model.eval()

    tx_lz = (
        pl.scan_parquet(TRANSCRIPT_FILE)
        .explode("transcript")
        .unnest("transcript")
        .select([
            "symbol",
            "reportedDate",
            pl.col("speaker"),
            pl.col("content").alias("text")
        ])
        .filter(pl.col("speaker") != "Operator")
    )

    process_symbol("NVDA", tx_lz, tokenizer, model, device)

if __name__ == "__main__":
    main()
