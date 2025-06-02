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

# Write scores starting from row 3
for idx, row in enumerate(range(3, len(df) + 3)):
    ws.cell(row=row, column=start_col, value=tfidf_scores[idx])
    ws.cell(row=row, column=start_col + 1, value=bleu_scores[idx])
    ws.cell(row=row, column=start_col + 2, value=contextual_scores[idx])

# === Compute Average Scores per User Story ===
user_story_list = df[user_story_col].tolist()
grouped_scores = defaultdict(lambda: {"tfidf": [], "bleu": [], "contextual": []})

for idx, story in enumerate(user_story_list):
    if tfidf_scores[idx] is not None:
        grouped_scores[story]["tfidf"].append(tfidf_scores[idx])
    if bleu_scores[idx] is not None:
        grouped_scores[story]["bleu"].append(bleu_scores[idx])
    if contextual_scores[idx] is not None:
        grouped_scores[story]["contextual"].append(contextual_scores[idx])

# Compute group averages
avg_tfidf, avg_bleu, avg_contextual = [], [], []
for story in user_story_list:
    tfidf_avg = np.mean(grouped_scores[story]["tfidf"]) if grouped_scores[story]["tfidf"] else None
    bleu_avg = np.mean(grouped_scores[story]["bleu"]) if grouped_scores[story]["bleu"] else None
    contextual_avg = np.mean(grouped_scores[story]["contextual"]) if grouped_scores[story]["contextual"] else None
    avg_tfidf.append(int(round(tfidf_avg)) if tfidf_avg is not None else None)
    avg_bleu.append(int(round(bleu_avg)) if bleu_avg is not None else None)
    avg_contextual.append(int(round(contextual_avg)) if contextual_avg is not None else None)

# Add average score headers
avg_headers = ["Avg TFIDF Score (%)", "Avg BLEU Score (%)", "Avg Contextual Precision (%)"]
for i, header in enumerate(avg_headers):
    ws.cell(row=2, column=start_col + 3 + i, value=header)

# Write averages starting from row 3
for idx, row in enumerate(range(3, len(df) + 3)):
    ws.cell(row=row, column=start_col + 3, value=avg_tfidf[idx])
    ws.cell(row=row, column=start_col + 4, value=avg_bleu[idx])
    ws.cell(row=row, column=start_col + 5, value=avg_contextual[idx])

# Save final Excel
wb.save(output_excel)
print(f"✅ Final Excel with scores and averages saved to: {output_excel}")
