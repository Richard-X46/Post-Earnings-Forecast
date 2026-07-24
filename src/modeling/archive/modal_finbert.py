import modal
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Define Volume
output_volume = modal.Volume.from_name("finbert-output", create_if_missing=True)

# Define App
app = modal.App("finbert-inference")

# Define Image
image = modal.Image.debian_slim().pip_install(
    "torch", "transformers", "polars", "pyarrow", "tqdm"
)

class TranscriptDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

@app.function(
    image=image,
    gpu="T4",
    timeout=600,
    volumes={"/data": output_volume},
)
def run_finbert(symbol: str):
    # Load model
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    device = torch.device("cuda")
    model.to(device)
    model.half()
    model.eval()

    # Read data from volume
    input_path = "/data/temp_transcripts.parquet"
    df = pl.read_parquet(input_path).filter(pl.col("symbol") == symbol)
    
    # Process
    ds = TranscriptDataset(df["content"].to_list(), tokenizer)
    loader = DataLoader(ds, batch_size=64, num_workers=0)
    
    all_predictions = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            all_predictions.extend(probs.cpu().numpy())
            
    df = df.with_columns(pl.Series("sentiment_probs", all_predictions))
    
    # Save to Volume
    output_path = f"/data/{symbol}.parquet"
    df.write_parquet(output_path)
    output_volume.commit()
    return f"Processed {symbol}"

@app.local_entrypoint()
def main():
    # Ensure file exists in volume locally before running
    # modal volume put finbert-output src/ingestion/data/backup/temp_transcripts.parquet /
    
    # Scan the file in the volume
    input_path = "/data/temp_transcripts.parquet"
    
    # NOTE: You must have uploaded the file to the volume first!
    # If this fails, run: `modal volume put finbert-output src/ingestion/data/backup/temp_transcripts.parquet /`
    
    all_symbols = pl.scan_parquet(input_path).select("symbol").unique().collect()["symbol"].to_list()
    
    print(f"Mapping {len(all_symbols)} symbols to Modal...")
    
    results = run_finbert.map(all_symbols)
    
    for result in results:
        print(result)
