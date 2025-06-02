import pandas as pd
import numpy as np
import spacy
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from openpyxl import load_workbook
from collections import defaultdict

# Load NLP models
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")
smoothie = SmoothingFunction().method4

# File paths
input_excel = "AQM_TCG_Metrics_OriginalADO 1.xlsx"
output_excel = "AQM_TCG_Metrics_OriginalADO_Scored.xlsx"

# Column names
user_story_col = 'User Story Title'
ado_action_col = 'Unnamed: 6'
generated_action_col = 'Unnamed: 11'

# Read the Excel into a DataFrame (preserve as-is)
df = pd.read_excel(input_excel, dtype=str)
df[ado_action_col] = df[ado_action_col].fillna("").str.strip()
df[generated_action_col] = df[generated_action_col].fillna("").str.strip()
df[user_story_col] = df[user_story_col].fillna("").str.strip()

# Result storage
tfidf_scores, bleu_scores, contextual_scores = [], [], []

# Grouped scoring by User Story Title
for story, group in tqdm(df.groupby(user_story_col), desc="Processing User Stories"):
    ado_actions = group[ado_action_col].tolist()
    generated_actions = group[generated_action_col].tolist()
    generated_actions = [x for x in generated_actions if x]

    if not generated_actions:
        tfidf_scores.extend([None] * len(group))
        bleu_scores.extend([None] * len(group))
        contextual_scores.extend([None] * len(group))
        continue

    gen_bleu_tokens = [list(token.text for token in nlp(step)) for step in generated_actions]
    gen_tfidf = TfidfVectorizer().fit(generated_actions)
    gen_vectors = gen_tfidf.transform(generated_actions)
    gen_embeddings = model.encode(generated_actions, convert_to_tensor=True)

    for ado in ado_actions:
        if not ado:
            tfidf_scores.append(None)
            bleu_scores.append(None)
            contextual_scores.append(None)
            continue

        ado_vec = gen_tfidf.transform([ado])
        tfidf_sim = cosine_similarity(ado_vec, gen_vectors)[0]
        tfidf_best = np.max(tfidf_sim)

        ado_tokens = [token.text for token in nlp(ado)]
        bleu_vals = [sentence_bleu([gen], ado_tokens, smoothing_function=smoothie) for gen in gen_bleu_tokens]
        bleu_best = max(bleu_vals)

        ado_emb = model.encode(ado, convert_to_tensor=True)
        contextual_sim = cosine_similarity([ado_emb], gen_embeddings)[0]
        contextual_best = np.max(contextual_sim)

        tfidf_scores.append(int(round(tfidf_best * 100)))
        bleu_scores.append(int(round(bleu_best * 100)))
        contextual_scores.append(int(round(contextual_best * 100)))

# Load workbook without disturbing formatting
wb = load_workbook(input_excel)
ws = wb.active

# Determine new column positions
start_col = ws.max_column + 1

# Score headers
score_headers = ["TFIDF Cosine Similarity (%)", "BLEU Score (%)", "Contextual Precision (%)"]
for i, header in enumerate(score_headers):
    ws.cell(row=2, column=start_col + i, value=header)

# Write individual scores starting from row 3
for idx, row in enumerate(range(3, len(df) + 3)):
    ws.cell(row=row, column=start_col, value=tfidf_scores[idx])
    ws.cell(row=row, column=start_col + 1, value=bleu_scores[idx])
    ws.cell(row=row, column=start_col + 2, value=contextual_scores[idx])

# === Compute Average Scores per User Story ===
grouped = df.groupby(user_story_col)
group_averages = {}

for story, group in grouped:
    indices = group.index.tolist()
    # Extract scores for this group by indices
    group_tfidf = [tfidf_scores[i] for i in indices if tfidf_scores[i] is not None]
    group_bleu = [bleu_scores[i] for i in indices if bleu_scores[i] is not None]
    group_contextual = [contextual_scores[i] for i in indices if contextual_scores[i] is not None]

    avg_tfidf = int(round(np.mean(group_tfidf))) if group_tfidf else None
    avg_bleu = int(round(np.mean(group_bleu))) if group_bleu else None
    avg_contextual = int(round(np.mean(group_contextual))) if group_contextual else None

    group_averages[story] = (avg_tfidf, avg_bleu, avg_contextual)

# Add average score headers (3 columns after individual scores)
avg_headers = ["Avg TFIDF Score (%)", "Avg BLEU Score (%)", "Avg Contextual Precision (%)"]
for i, header in enumerate(avg_headers):
    ws.cell(row=2, column=start_col + 3 + i, value=header)

# Write averages only once per group at the first row of the group
row_offset = 3  # Because data starts at Excel row 3
for story, group in grouped:
    avg_tfidf, avg_bleu, avg_contextual = group_averages[story]
    first_row = group.index.min() + row_offset  # Excel row number for the first row of this group

    ws.cell(row=first_row, column=start_col + 3, value=avg_tfidf)
    ws.cell(row=first_row, column=start_col + 4, value=avg_bleu)
    ws.cell(row=first_row, column=start_col + 5, value=avg_contextual)

# Save final Excel
wb.save(output_excel)
print(f"✅ Final Excel with scores and grouped averages saved to: {output_excel}")
