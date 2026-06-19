import pysentiment2 as ps
import polars as pl

lm = ps.LM()

def get_lm_polarity(text: str) -> float:
    if text is None:
        return 0.0
    tokens = lm.tokenize(text)
    score = lm.get_score(tokens)
    return score

TRANSCRIPT_FILE = "src/ingestion/data/backup/temp_transcripts.parquet"

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

symbol = "NVDA"
symbol_data = tx_lz.filter(pl.col("symbol") == symbol).collect()

symbol_data = symbol_data.with_columns(
    pl.col("text")
    .map_elements(get_lm_polarity, return_dtype=pl.Float64)
    .alias("lm_sentiment")
)

symbol_data['text'][1]

text = """ally, we revealed plans to build Earth-2, the world's most powerful AI supercomputer dedicated to confronting climate change. The system would be the climate change counterpart to Cambridge-1, the U.K.'s most powerful AI supercomputer that we built for corporate research. Earth-2 furnishes all the technologies we've invented up to this moment. Let me discuss Arm. I'll provide you a brief update on our proposed acquisition of Arm. Arm with NVIDIA is a great opportunity for the industry and customers with NVIDIA's scale, capabilities, and robust understanding of data center computing, acceleration, and AI. We assessed Arm in expanding their reach into data centers, IoT, and PCs, and advanced Arm's IP for decades to come. The combination of our companies can enhance competition in the industry as we work together on further building the world of AI. Regulators at the U.S. FTC have expressed concerns regarding the transaction and we are engaged in discussions with them regarding remedies to address those concerns. The transaction has been under review by the China Antitrust Authority, pending the formal case initiation. Regulators in the U.K. and the EU have declined to approve the transaction in Phase 1 of their reviews on competition concerns. In the U.K., they have also voiced national security concerns. We have begun the Phase 2 process in the EU and U.K. jurisdictions. Despite these concerns and those raised by some Arm licensees, we continue to believe in the merits and the benefits of the acquisition for Arm, its licensees, and the industry. We believe these concerns raised by some Arm licensees do not invalidate the merits of the ongoing acquisition. Moving to the rest of the P&L. GAAP gross margin for the third quarter was up 260 basis points from a year earlier, primarily due to higher end mix within desktop, notebook, and GeForce GPUs. The year-on-year increase also benefited from a reduced impact of acquisition-related costs. GAAP gross margin was u... (line truncated to 2000 chars)
"""

get_lm_polarity(text)
