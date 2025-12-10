"""
Analyse du sentiment avec FinBERT
Remplit finbert_sentiment et finbert_confidence dans la base SQLite
"""

import sqlite3
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import numpy as np

DB_PATH = "stock_news.db"


class FinBertSentimentAnalyzer:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path

        print("📦 Initialisation de FinBERT…")

        # Charger FinBERT
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.model.eval()

        # Mapping id → label
        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}

        print("✅ FinBERT est prêt !\n")

    def analyze_text(self, text):
        """Analyse un texte avec FinBERT"""
        if not text or pd.isna(text):
            return "neutral", 0.33

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).numpy()[0]

        sentiment_id = np.argmax(probs)
        sentiment = self.id2label[sentiment_id]
        confidence = float(probs[sentiment_id])

        return sentiment, confidence

    def get_articles_without_finbert(self):
        """Récupère uniquement les articles sans résultat FinBERT"""
        conn = sqlite3.connect(self.db_path)

        df = pd.read_sql_query("""
            SELECT id, ticker, title, description
            FROM news_articles
            WHERE finbert_sentiment IS NULL
               OR finbert_sentiment = ''
               OR finbert_sentiment = 'None'
               OR finbert_confidence IS NULL
        """, conn)

        conn.close()
        print(f"🔍 Articles à analyser: {len(df)}")
        return df

    def update_finbert(self, article_id, sentiment, confidence):
        """Met à jour la base SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE news_articles
            SET finbert_sentiment = ?,
                finbert_confidence = ?
            WHERE id = ?
        """, (sentiment, confidence, article_id))

        conn.commit()
        conn.close()

    def analyze_all(self):
        """Analyse toutes les news non encore analysées"""
        print("📊 Chargement des articles non analysés…\n")
        df = self.get_articles_without_finbert()

        if len(df) == 0:
            print("🎉 Tous les articles ont déjà un sentiment FinBERT.")
            return

        print(f"🚀 Début de l'analyse FinBERT sur {len(df)} articles\n")

        results = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="FinBERT"):
            text = f"{row['title']}. {row['description']}"
            sentiment, confidence = self.analyze_text(text)

            self.update_finbert(row["id"], sentiment, confidence)

            results.append({
                "ticker": row["ticker"],
                "sentiment": sentiment,
                "confidence": confidence
            })

        # Résumé
        results_df = pd.DataFrame(results)
        print("\n📈 Distribution des sentiments FinBERT :")
        print(results_df["sentiment"].value_counts())

        print(f"\nConfiance moyenne : {results_df['confidence'].mean():.2%}")

        self.get_stats()

    def get_stats(self):
        """Affiche les stats générales"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM news_articles")
        total = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*)
            FROM news_articles
            WHERE finbert_sentiment IS NOT NULL
              AND finbert_sentiment != ''
        """)
        done = cursor.fetchone()[0]

        conn.close()

        print("\n" + "="*60)
        print("📊 STATISTIQUES FINALES FINBERT")
        print("="*60)
        print(f"Articles totaux : {total}")
        print(f"Articles avec FinBERT : {done} ({done/total*100:.1f}%)")
        print("="*60)


# 🚀 LANCEMENT
if __name__ == "__main__":
    analyzer = FinBertSentimentAnalyzer()

    print("\n🚀 Lancement de l'analyse FinBERT…")
    analyzer.analyze_all()

    print("\n✅ FinBERT terminé !")
