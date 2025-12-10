import requests
import sqlite3
import time
from datetime import datetime, timedelta
import json

class PolygonNewsCollector:
    def __init__(self, api_key, db_path='stock_news.db'):
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

        # Table pour tracker les tickers et périodes collectées
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

    def fetch_news(self, ticker, start_date, end_date, limit=1000):
        """Récupère les news pour un ticker sur une période donnée"""
        articles = []
        params = {
            'ticker': ticker,
            'published_utc.gte': start_date,
            'published_utc.lte': end_date,
            'limit': min(limit, 1000),  # Max 1000 par requête
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
                articles.extend(results)
                total_fetched += len(results)

                print(f"   Récupéré {len(results)} articles (total: {total_fetched})")

                # Vérifier s'il y a une page suivante
                next_url = data.get('next_url')
                if not next_url or total_fetched >= limit:
                    break

                # Ajouter l'API key au next_url
                if '?' in next_url:
                    next_url += f'&apiKey={self.api_key}'
                else:
                    next_url += f'?apiKey={self.api_key}'

                # Respecter le rate limit (5 requêtes/minute pour free tier)
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

                # Gérer les keywords (liste -> string)
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

    def collect_ticker_history(self, ticker, years=5):
        """Collecte l'historique complet pour un ticker"""
        print(f"\n📊 Collecte des données pour {ticker} sur {years} ans")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)

        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        print(f"   Période: {start_str} à {end_str}")

        articles = self.fetch_news(ticker, start_str, end_str)

        if articles:
            count = self.save_articles(articles, ticker)
            self.log_collection(ticker, start_str, end_str, count)
            return count

        return 0

    def collect_multiple_tickers(self, tickers, years=5):
        """Collecte les données pour plusieurs tickers"""
        print(f"\n🚀 Début de la collecte pour {len(tickers)} tickers")
        print("=" * 60)

        results = {}

        for i, ticker in enumerate(tickers, 1):
            print(f"\n[{i}/{len(tickers)}] Traitement de {ticker}")
            try:
                count = self.collect_ticker_history(ticker, years)
                results[ticker] = count
                print(f"✅ {ticker}: {count} articles collectés")
            except Exception as e:
                print(f"❌ Erreur pour {ticker}: {e}")
                results[ticker] = 0

            # Pause entre les tickers pour éviter le rate limit
            if i < len(tickers):
                print(f"   ⏳ Pause de 15 secondes avant le prochain ticker...")
                time.sleep(15)

        print("\n" + "=" * 60)
        print("📈 RÉSUMÉ DE LA COLLECTE")
        print("=" * 60)
        for ticker, count in results.items():
            print(f"   {ticker}: {count} articles")
        print(f"\n   TOTAL: {sum(results.values())} articles collectés")

        return results

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
        print(f"Articles avec sentiment: {with_sentiment} ({with_sentiment/total*100:.1f}%)" if total > 0 else "Articles avec sentiment: 0")
        print("\nPar ticker:")
        for ticker, count in by_ticker:
            print(f"   {ticker}: {count} articles")
        print("=" * 60)


# EXEMPLE D'UTILISATION STANDALONE
if __name__ == "__main__":
    # Configuration
    API_KEY = "SiV7GQdKTF2ZtrAr1xNSrnNYP11dKCAC"

    # Liste de tickers à collecter
    TICKERS = [
        'AAPL',   # Apple
        'MSFT',   # Microsoft
        'GOOGL',  # Alphabet
        'AMZN',   # Amazon
        'TSLA',   # Tesla
        'META',   # Meta
        'NVDA',   # Nvidia
        'JPM',    # JPMorgan
        'V',      # Visa
        'PPH'     # PPH
    ]

    # Créer le collecteur
    collector = PolygonNewsCollector(api_key=API_KEY)

    # Collecter les données pour tous les tickers (5 ans par défaut)
    results = collector.collect_multiple_tickers(TICKERS, years=5)

    # Afficher les statistiques
    collector.get_stats()

    print("\n✅ Collecte terminée! Base de données prête pour l'entraînement.")
    print(f"📁 Fichier: stock_news.db")
