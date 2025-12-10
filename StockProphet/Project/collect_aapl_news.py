import requests
import sqlite3
import time
from datetime import datetime
import json

class PolygonNewsCollector:
    def __init__(self, api_key, db_path='AAPL_stock_news.db'):
        self.api_key = api_key
        self.db_path = db_path
        self.base_url = "https://api.polygon.io/v2/reference/news"
        self.init_database()

    def init_database(self):
        """Initialise la base de données avec les tables nécessaires"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Table pour les articles
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_articles (
                id TEXT PRIMARY KEY,
                ticker TEXT NOT NULL,
                published_utc TEXT,
                title TEXT,
                author TEXT,
                article_url TEXT,
                description TEXT,
                keywords TEXT,
                image_url TEXT,
                sentiment TEXT,
                sentiment_reasoning TEXT,
                raw_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Table pour tracker les périodes collectées
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS collection_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                start_date TEXT,
                end_date TEXT,
                articles_collected INTEGER,
                collected_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Index pour améliorer les performances
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_ticker_date
            ON news_articles(ticker, published_utc)
        ''')

        conn.commit()
        conn.close()
        print(f"✅ Base de données initialisée: {self.db_path}")

    def fetch_news(self, ticker, start_date, end_date, total_limit=None):
        """
        Récupère les news pour un ticker sur une période donnée.
        total_limit = None -> récupère toutes les pages disponibles.
        """
        articles = []
        params = {
            'ticker': ticker,
            'published_utc.gte': start_date,
            'published_utc.lte': end_date,
            'limit': 1000,  # max par requête
            'apiKey': self.api_key,
            'sort': 'published_utc',
            'order': 'asc'
        }

        next_url = None
        total_fetched = 0

        while True:
            try:
                if next_url:
                    url = next_url
                    response = requests.get(url)
                else:
                    response = requests.get(self.base_url, params=params)

                if response.status_code == 429:
                    print("⚠️  Rate limit atteint, attente de 60 secondes...")
                    time.sleep(60)
                    continue

                response.raise_for_status()
                data = response.json()

                results = data.get('results', [])
                if not results:
                    print("   Aucune donnée supplémentaire, arrêt.")
                    break

                articles.extend(results)
                total_fetched += len(results)

                print(f"   Récupéré {len(results)} articles (total: {total_fetched})")

                # Vérifier s'il y a une page suivante
                next_url = data.get('next_url')

                # Ajouter l'API key au next_url si besoin
                if next_url:
                    if '?' in next_url:
                        next_url += f'&apiKey={self.api_key}'
                    else:
                        next_url += f'?apiKey={self.api_key}'

                # Condition d'arrêt : plus de page OU total_limit atteint
                if not next_url:
                    print("   Plus de page suivante, arrêt de la collecte.")
                    break

                if total_limit is not None and total_fetched >= total_limit:
                    print("   Limite totale atteinte, arrêt de la collecte.")
                    break

                # Respecter le rate limit (5 requêtes/minute free tier)
                time.sleep(12)

            except requests.exceptions.RequestException as e:
                print(f"❌ Erreur lors de la requête: {e}")
                break

        return articles

    def save_articles(self, articles, ticker):
        """Sauvegarde les articles dans la base de données"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        new_articles = 0
        duplicates = 0

        for article in articles:
            try:
                # Extraire les données avec gestion des valeurs manquantes
                article_id = article.get('id', '')
                published_utc = article.get('published_utc', '')
                title = article.get('title', '')
                author = article.get('author', '')
                article_url = article.get('article_url', '')
                description = article.get('description', '')

                # Gérer les keywords (liste -> string JSON)
                keywords = json.dumps(article.get('keywords', []))

                # Gérer l'image
                image_url = article.get('image_url', '')

                # Extraire les données de sentiment
                insights = article.get('insights', [{}])
                sentiment = ''
                sentiment_reasoning = ''

                if insights and len(insights) > 0:
                    sentiment = insights[0].get('sentiment', '')
                    sentiment_reasoning = insights[0].get('sentiment_reasoning', '')

                # Sauvegarder le JSON complet
                raw_json = json.dumps(article)

                cursor.execute('''
                    INSERT OR IGNORE INTO news_articles
                    (id, ticker, published_utc, title, author, article_url,
                     description, keywords, image_url, sentiment,
                     sentiment_reasoning, raw_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (article_id, ticker, published_utc, title, author,
                      article_url, description, keywords, image_url,
                      sentiment, sentiment_reasoning, raw_json))

                if cursor.rowcount > 0:
                    new_articles += 1
                else:
                    duplicates += 1

            except Exception as e:
                print(f"⚠️  Erreur lors de la sauvegarde d'un article: {e}")
                continue

        conn.commit()
        conn.close()

        print(f"   💾 {new_articles} nouveaux articles sauvegardés, {duplicates} doublons ignorés")
        return new_articles

    def log_collection(self, ticker, start_date, end_date, count):
        """Enregistre la collection dans le log"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO collection_log (ticker, start_date, end_date, articles_collected)
            VALUES (?, ?, ?, ?)
        ''', (ticker, start_date, end_date, count))

        conn.commit()
        conn.close()

    def collect_ticker_range(self, ticker, start_date, end_date, total_limit=None):
        """Collecte les news pour un ticker sur une plage de dates précise."""
        print(f"\n📊 Collecte des données pour {ticker}")
        print(f"   Période: {start_date} à {end_date}")

        articles = self.fetch_news(ticker, start_date, end_date, total_limit=total_limit)

        if articles:
            count = self.save_articles(articles, ticker)
            self.log_collection(ticker, start_date, end_date, count)
            return count

        print("   Aucun article à sauvegarder.")
        return 0

    def get_stats(self):
        """Affiche les statistiques de la base de données"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total d'articles
        cursor.execute("SELECT COUNT(*) FROM news_articles")
        total = cursor.fetchone()[0]

        # Par ticker
        cursor.execute("""
            SELECT ticker, COUNT(*) as count
            FROM news_articles
            GROUP BY ticker
            ORDER BY count DESC
        """)
        by_ticker = cursor.fetchall()

        # Articles avec sentiment
        cursor.execute("""
            SELECT COUNT(*)
            FROM news_articles
            WHERE sentiment != '' AND sentiment IS NOT NULL
        """)
        with_sentiment = cursor.fetchone()[0]

        conn.close()

        print("\n📊 STATISTIQUES DE LA BASE DE DONNÉES")
        print("=" * 60)
        print(f"Total d'articles: {total}")
        if total > 0:
            print(f"Articles avec sentiment: {with_sentiment} ({with_sentiment/total*100:.1f}%)")
        else:
            print("Articles avec sentiment: 0")

        print("\nPar ticker:")
        for ticker, count in by_ticker:
            print(f"   {ticker}: {count} articles")
        print("=" * 60)


# EXÉCUTION STANDALONE POUR AAPL 2020–2025
if __name__ == "__main__":
    # 🔑 Mets ta clé API Polygon ici
    API_KEY = "SiV7GQdKTF2ZtrAr1xNSrnNYP11dKCAC"

    # Créer le collecteur avec une DB dédiée à AAPL
    collector = PolygonNewsCollector(api_key=API_KEY, db_path="AAPL_stock_news.db")

    # Période fixe : 1er janvier 2020 -> 31 décembre 2025
    START_DATE = "2020-01-01"
    END_DATE = "2025-12-10"

    print("\n🚀 Début de la collecte pour AAPL (2020–2025)")
    count = collector.collect_ticker_range("AAPL", START_DATE, END_DATE, total_limit=None)

    print(f"\n✅ AAPL: {count} articles collectés entre {START_DATE} et {END_DATE}")

    # Statistiques
    collector.get_stats()

    print("\n✅ Collecte terminée! Base de données prête pour l'entraînement.")
    print(f"📁 Fichier: AAPL_stock_news.db")
