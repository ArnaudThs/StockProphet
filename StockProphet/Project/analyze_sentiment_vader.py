"""
Version RAPIDE avec VADER (30 secondes pour 7000 articles)
Moins précis que FinBERT mais parfait pour tester
"""

import sqlite3
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

class VaderSentimentAnalyzer:
    def __init__(self, db_path='stock_news.db'):
        self.db_path = db_path
        print("📦 Initialisation de VADER...")
        self.analyzer = SentimentIntensityAnalyzer()
        print("✅ Prêt!\n")

    def analyze_text(self, text):
        """Analyse le sentiment avec VADER"""
        if not text or pd.isna(text):
            return 'neutral', 0.33

        # Score VADER (compound entre -1 et 1)
        scores = self.analyzer.polarity_scores(text)
        compound = scores['compound']

        # Classification
        if compound >= 0.05:
            sentiment = 'positive'
        elif compound <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        # Confidence (valeur absolue du compound)
        confidence = abs(compound)

        return sentiment, confidence

    def get_articles_without_sentiment(self):
        """Récupère les articles sans sentiment"""
        conn = sqlite3.connect(self.db_path)

        df = pd.read_sql_query("""
            SELECT id, ticker, title, description
            FROM news_articles
            WHERE sentiment = '' OR sentiment IS NULL
        """, conn)

        conn.close()
        return df

    def update_sentiment(self, article_id, sentiment, confidence):
        """Met à jour le sentiment dans la DB"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE news_articles
            SET sentiment = ?,
                sentiment_reasoning = ?
            WHERE id = ?
        """, (sentiment, f"VADER confidence: {confidence:.2%}", article_id))

        conn.commit()
        conn.close()

    def analyze_all(self):
        """Analyse tous les articles"""
        print("📊 Récupération des articles...\n")
        df = self.get_articles_without_sentiment()

        if len(df) == 0:
            print("✅ Tous les articles ont déjà un sentiment!")
            return

        print(f"🔍 {len(df)} articles à analyser\n")

        results = []

        # Analyser avec barre de progression
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Analyse"):
            # Combiner titre + description
            text = f"{row['title']}. {row['description']}"

            # Analyser
            sentiment, confidence = self.analyze_text(text)

            # Sauvegarder
            self.update_sentiment(row['id'], sentiment, confidence)

            results.append({
                'ticker': row['ticker'],
                'sentiment': sentiment,
                'confidence': confidence
            })

        # Statistiques
        results_df = pd.DataFrame(results)

        print("\n" + "="*60)
        print("📊 RÉSULTATS DE L'ANALYSE")
        print("="*60)
        print(f"\nTotal analysé: {len(results_df)} articles")
        print(f"\nDistribution des sentiments:")
        print(results_df['sentiment'].value_counts())
        print(f"\nConfiance moyenne: {results_df['confidence'].mean():.2%}")

        print(f"\nPar ticker:")
        ticker_stats = results_df.groupby('ticker')['sentiment'].value_counts().unstack(fill_value=0)
        print(ticker_stats)

        print("\n✅ Analyse terminée!")

        # Stats finales
        self.get_stats()

    def get_stats(self):
        """Stats finales de la DB"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM news_articles")
        total = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*) FROM news_articles
            WHERE sentiment != '' AND sentiment IS NOT NULL
        """)
        with_sentiment = cursor.fetchone()[0]

        cursor.execute("""
            SELECT sentiment, COUNT(*) as count
            FROM news_articles
            WHERE sentiment != '' AND sentiment IS NOT NULL
            GROUP BY sentiment
        """)
        sentiment_dist = cursor.fetchall()

        conn.close()

        print("\n" + "="*60)
        print("📈 STATISTIQUES FINALES DE LA BASE DE DONNÉES")
        print("="*60)
        print(f"Total d'articles: {total}")
        print(f"Articles avec sentiment: {with_sentiment} ({with_sentiment/total*100:.1f}%)")

        if sentiment_dist:
            print(f"\nDistribution globale:")
            for sentiment, count in sentiment_dist:
                print(f"   {sentiment}: {count} articles ({count/with_sentiment*100:.1f}%)")

        print("="*60)


# UTILISATION
if __name__ == "__main__":
    analyzer = VaderSentimentAnalyzer(db_path='stock_news.db')

    print("🚀 Lancement de l'analyse VADER (rapide)")
    print("⏱️  Estimation: ~30 secondes pour 7000 articles\n")

    analyzer.analyze_all()

    print("\n✅ Base de données prête pour le LSTM + RL!")
