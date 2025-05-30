import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

# Load CSV file
df = pd.read_csv("AQM_Feature_4122287_testcase.csv")

# Define key columns based on flat CSV headers
col_id = 'User Story ID'
userstory_fields = [
    'User Story Title',
    'User Story Description',
    'Acceptance Criteria',
    'Test Scenario'  # This holds ADO test steps
]
testcase_fields = [
    'Generated Testcase_Test Case Title',
    'Generated Testcase_Action',
    'Generated Testcase_Excepted Output'
]

def clean_text(text):
    if pd.isna(text):
        return ""
    return str(text).replace('\n', ' ').replace('\r', ' ').strip().lower()

# Grouping logic
grouped = df.groupby(col_id)
model = SentenceTransformer('all-MiniLM-L6-v2')
smoothie = SmoothingFunction().method4
results = []

for user_story_id, group in grouped:
    # Combine user story metadata (excluding 'Test Scenario')
    userstory_text = " ".join([clean_text(group[field].iloc[0]) for field in userstory_fields if field != 'Test Scenario'])

    # Collect ADO Test Actions and split into steps
    ado_actions_raw = clean_text(group['Test Scenario'].iloc[0])
    ado_steps = [step.strip() for step in sent_tokenize(ado_actions_raw) if step.strip()]

    # Combine all generated testcase actions into one string
    generated_actions = " ".join([clean_text(tc) for tc in group['Generated Testcase_Action']])

    # Evaluate similarity metrics for each ADO step
    step_results = []
    for step in ado_steps:
        # TF-IDF Cosine Similarity
        tfidf = TfidfVectorizer().fit([step, generated_actions])
        vecs = tfidf.transform([step, generated_actions])
        cosine = cosine_similarity(vecs[0], vecs[1])[0][0]

        # BLEU Score
        bleu = sentence_bleu(
            [nltk.word_tokenize(generated_actions)],
            nltk.word_tokenize(step),
            smoothing_function=smoothie
        )

        # Sentence Transformer Cosine
        emb_step = model.encode(step, convert_to_tensor=True)
        emb_gen = model.encode(generated_actions, convert_to_tensor=True)
        contextual_sim = cosine_similarity([emb_step], [emb_gen])[0][0]
        contextual_precision = contextual_sim
        contextual_recall = contextual_sim
        contextual_f1 = 2 * (contextual_precision * contextual_recall) / (contextual_precision + contextual_recall + 1e-8)

        step_results.append({
            'Cosine Similarity': cosine,
            'BLEU Score': bleu,
            'Contextual Precision': contextual_precision,
            'Contextual Recall': contextual_recall,
            'Contextual F1': contextual_f1
        })

    # Aggregate stepwise results (e.g., mean)
    if step_results:
        avg_scores = {
            'Cosine Similarity': np.mean([r['Cosine Similarity'] for r in step_results]),
            'BLEU Score': np.mean([r['BLEU Score'] for r in step_results]),
            'Contextual Precision': np.mean([r['Contextual Precision'] for r in step_results]),
            'Contextual Recall': np.mean([r['Contextual Recall'] for r in step_results]),
            'Contextual F1': np.mean([r['Contextual F1'] for r in step_results]),
        }
    else:
        avg_scores = {
            'Cosine Similarity': np.nan,
            'BLEU Score': np.nan,
            'Contextual Precision': np.nan,
            'Contextual Recall': np.nan,
            'Contextual F1': np.nan,
        }

    results.append({
        col_id: user_story_id,
        **avg_scores
    })

# Map results back to the original DataFrame
score_dict = {row[col_id]: row for row in results}
df['Cosine Similarity'] = np.nan
df['BLEU Score'] = np.nan
df['Contextual Precision'] = np.nan
df['Contextual Recall'] = np.nan
df['Contextual F1'] = np.nan

for idx, row in df.iterrows():
    user_story_id = row[col_id]
    if pd.notna(user_story_id) and user_story_id in score_dict:
        df.at[idx, 'Cosine Similarity'] = round(score_dict[user_story_id]['Cosine Similarity'], 4)
        df.at[idx, 'BLEU Score'] = round(score_dict[user_story_id]['BLEU Score'], 4)
        df.at[idx, 'Contextual Precision'] = round(score_dict[user_story_id]['Contextual Precision'], 4)
        df.at[idx, 'Contextual Recall'] = round(score_dict[user_story_id]['Contextual Recall'], 4)
        df.at[idx, 'Contextual F1'] = round(score_dict[user_story_id]['Contextual F1'], 4)

# Save result to CSV
df.to_csv("UserStory_Testcase_Scored_Final.csv", index=False)
print("✅ File saved: UserStory_Testcase_Scored_Final.csv")
