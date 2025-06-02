import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import TreebankWordTokenizer
import openpyxl
from collections import defaultdict

# Input/Output paths
input_excel = "your_file.xlsx"
output_excel = "your_file_updated.xlsx"

# Column names
user_story_col = "User Story Title"
ado_col = "ADO Testcase_Action"
generated_col = "Generated Testcase_Action"

# Load Excel with formatting preserved
df = pd.read_excel(input_excel)
wb = openpyxl.load_workbook(input_excel)
ws = wb.active

# Initialize models
tokenizer = TreebankWordTokenizer()
model = SentenceTransformer("all-MiniLM-L6-v2")

# Score storage
tfidf_scores = []
bleu_scores = []
contextual_scores = []

# Process row-wise
for i, row in df.iterrows():
    ado_action = row.get(ado_col)
    gen_actions = row.get(generated_col)

    if pd.isna(ado_action) or not isinstance(ado_action, str) or not ado_action.strip():
        tfidf_scores.append(None)
        bleu_scores.append(None)
        contextual_scores.append(None)
        continue

    # Split multiline actions
    ado_steps = [s.strip() for s in ado_action.strip().split("\n") if s.strip()]
    gen_steps = [s.strip() for s in str(gen_actions).strip().split("\n") if isinstance(gen_actions, str) and s.strip()]

    tfidf_match_scores = []
    bleu_match_scores = []
    contextual_match_scores = []

    for ado_step in ado_steps:
        # BLEU
        bleu_vals = []
        reference = tokenizer.tokenize(ado_step)
        for gen_step in gen_steps:
            candidate = tokenizer.tokenize(gen_step)
            bleu = sentence_bleu([reference], candidate, smoothing_function=SmoothingFunction().method1)
            bleu_vals.append(bleu * 100)

        # TF-IDF Cosine
        tfidf = TfidfVectorizer().fit_transform([ado_step] + gen_steps)
        cosine_similarities = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten() * 100

        # Contextual Similarity
        emb = model.encode([ado_step] + gen_steps)
        contextual_sim = cosine_similarity([emb[0]], emb[1:]).flatten() * 100

        tfidf_match_scores.append(np.max(cosine_similarities) if len(cosine_similarities) > 0 else 0)
        bleu_match_scores.append(np.max(bleu_vals) if bleu_vals else 0)
        contextual_match_scores.append(np.max(contextual_sim) if len(contextual_sim) > 0 else 0)

    tfidf_scores.append(round(np.mean(tfidf_match_scores)))
    bleu_scores.append(round(np.mean(bleu_match_scores)))
    contextual_scores.append(round(np.mean(contextual_match_scores)))

# Find starting column index
start_col = len(df.columns) + 1
ws.cell(row=1, column=start_col, value="TFIDF Score (%)")
ws.cell(row=1, column=start_col + 1, value="BLEU Score (%)")
ws.cell(row=1, column=start_col + 2, value="Contextual Score (%)")

# Write scores row-wise
for i, (tfidf, bleu, contextual) in enumerate(zip(tfidf_scores, bleu_scores, contextual_scores), start=2):
    if tfidf is not None:
        ws.cell(row=i, column=start_col, value=tfidf)
    if bleu is not None:
        ws.cell(row=i, column=start_col + 1, value=bleu)
    if contextual is not None:
        ws.cell(row=i, column=start_col + 2, value=contextual)

# Step 2: Compute average scores by User Story Title
grouped_scores = defaultdict(lambda: {"tfidf": [], "bleu": [], "contextual": [], "start_row": None})

for i, row in enumerate(range(2, len(df) + 2)):
    title = df.loc[i, user_story_col]
    if not title:
        continue
    if grouped_scores[title]["start_row"] is None:
        grouped_scores[title]["start_row"] = row
    if tfidf_scores[i] is not None:
        grouped_scores[title]["tfidf"].append(tfidf_scores[i])
    if bleu_scores[i] is not None:
        grouped_scores[title]["bleu"].append(bleu_scores[i])
    if contextual_scores[i] is not None:
        grouped_scores[title]["contextual"].append(contextual_scores[i])

# Step 3: Write average values
avg_col_start = start_col + 3
ws.cell(row=1, column=avg_col_start, value="Avg TFIDF Score (%)")
ws.cell(row=1, column=avg_col_start + 1, value="Avg BLEU Score (%)")
ws.cell(row=1, column=avg_col_start + 2, value="Avg Contextual Score (%)")

for title, data in grouped_scores.items():
    row = data["start_row"]
    avg_tfidf = round(np.mean(data["tfidf"])) if data["tfidf"] else None
    avg_bleu = round(np.mean(data["bleu"])) if data["bleu"] else None
    avg_contextual = round(np.mean(data["contextual"])) if data["contextual"] else None

    ws.cell(row=row, column=avg_col_start, value=avg_tfidf)
    ws.cell(row=row, column=avg_col_start + 1, value=avg_bleu)
    ws.cell(row=row, column=avg_col_start + 2, value=avg_contextual)

# Save final Excel
wb.save(output_excel)
print(f"✅ Final Excel file saved with individual and average scores: {output_excel}")
