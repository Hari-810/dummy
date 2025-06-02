import pandas as pd
import numpy as np
import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from openpyxl import load_workbook
from collections import defaultdict

# Initialize NLP models
nlp = spacy.load("en_core_web_md")
nltk.download('punkt')

# File and column names
input_excel = "your_input_file.xlsx"
output_excel = "your_input_file.xlsx"  # Same file
user_story_col = "User Story Title"
ado_action_col = "ADO Testcase_Action"
generated_action_col = "Generated Testcase_Action"

# Load Excel with pandas
df = pd.read_excel(input_excel)

# Initialize score lists
tfidf_scores = []
bleu_scores = []
contextual_scores = []

# Precompute all generated test actions
generated_actions_all = df[generated_action_col].dropna().astype(str).tolist()

# TF-IDF model over all text
tfidf_vectorizer = TfidfVectorizer().fit(generated_actions_all)

# Compute scores
for idx, row in df.iterrows():
    ado_action = str(row[ado_action_col]).strip()
    if not ado_action or ado_action.lower() == "nan":
        tfidf_scores.append(None)
        bleu_scores.append(None)
        contextual_scores.append(None)
        continue

    max_tfidf, max_bleu, max_contextual = 0, 0, 0

    for gen_action in generated_actions_all:
        # TF-IDF Cosine Similarity
        tfidf_matrix = tfidf_vectorizer.transform([ado_action, gen_action])
        cos_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0] * 100
        max_tfidf = max(max_tfidf, cos_sim)

        # BLEU Score
        ref_tokens = nltk.word_tokenize(gen_action.lower())
        cand_tokens = nltk.word_tokenize(ado_action.lower())
        bleu = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=SmoothingFunction().method1) * 100
        max_bleu = max(max_bleu, bleu)

        # Contextual similarity using spaCy
        doc1 = nlp(ado_action)
        doc2 = nlp(gen_action)
        sim = doc1.similarity(doc2) * 100
        max_contextual = max(max_contextual, sim)

    tfidf_scores.append(round(max_tfidf))
    bleu_scores.append(round(max_bleu))
    contextual_scores.append(round(max_contextual))

# Load workbook for writing results
wb = load_workbook(input_excel)
ws = wb.active

# Find starting column for new scores
existing_cols = len(df.columns)
start_col = existing_cols + 1

# Write headers
ws.cell(row=1, column=start_col, value="TFIDF Score (%)")
ws.cell(row=1, column=start_col + 1, value="BLEU Score (%)")
ws.cell(row=1, column=start_col + 2, value="Contextual Score (%)")

# Write individual scores
for i in range(len(df)):
    if tfidf_scores[i] is not None:
        ws.cell(row=i + 2, column=start_col, value=tfidf_scores[i])
        ws.cell(row=i + 2, column=start_col + 1, value=bleu_scores[i])
        ws.cell(row=i + 2, column=start_col + 2, value=contextual_scores[i])

# Grouped averages by User Story Title
grouped_scores = defaultdict(lambda: {"tfidf": [], "bleu": [], "contextual": [], "start_row": None})

for i in range(len(df)):
    title = df.loc[i, user_story_col]
    if not title:
        continue
    if grouped_scores[title]["start_row"] is None:
        grouped_scores[title]["start_row"] = i + 2
    if tfidf_scores[i] is not None:
        grouped_scores[title]["tfidf"].append(tfidf_scores[i])
    if bleu_scores[i] is not None:
        grouped_scores[title]["bleu"].append(bleu_scores[i])
    if contextual_scores[i] is not None:
        grouped_scores[title]["contextual"].append(contextual_scores[i])

# Column start for average scores
avg_col_start = start_col + 3

# Write headers
ws.cell(row=1, column=avg_col_start, value="Avg TFIDF Score (%)")
ws.cell(row=1, column=avg_col_start + 1, value="Avg BLEU Score (%)")
ws.cell(row=1, column=avg_col_start + 2, value="Avg Contextual Score (%)")

# Write average scores only in the first row per group
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
print(f"✅ All scores and averages saved to: {output_excel}")
