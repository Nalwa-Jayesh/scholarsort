from time import sleep

import feedparser
import pandas as pd


def fetch_arxiv_abstracts_by_category(
    categories, per_category=300, batch_size=100, delay=3
):
    """
    Fetch balanced abstracts from arXiv for multiple categories.

    Args:
        categories (List[str]): e.g., ['cs.LG', 'math.ST', 'physics.gen-ph']
        per_category (int): How many to fetch per category
        batch_size (int): Batch size per API call
        delay (int): Seconds to wait between calls (avoid rate limit)

    Returns:
        pd.DataFrame with columns: title, abstract, category, published
    """
    base_url = "http://export.arxiv.org/api/query?"
    all_results = []

    for cat in categories:
        print(f"[INFO] Fetching {per_category} entries from arXiv category: {cat}")
        results = []
        for start in range(0, per_category, batch_size):
            query = f"search_query=cat:{cat}&start={start}&max_results={batch_size}"
            url = base_url + query
            feed = feedparser.parse(url)
            for entry in feed.entries:
                results.append(
                    {
                        "title": entry.title,
                        "abstract": entry.summary.replace("\n", " ").strip(),
                        "category": cat,
                        "published": entry.published,
                    }
                )
            sleep(delay)
        all_results.extend(results[:per_category])  # just in case overfetched
    return pd.DataFrame(all_results)


def load_arxiv_data(
    from_api=True, categories=None, per_category=300, local_path=None, limit=None
):
    """
    Load data either from the arXiv API or from a local CSV.

    Args:
        from_api (bool): Whether to fetch via API or use local CSV
        categories (list): List of arXiv categories
        per_category (int): Abstracts to pull per category
        local_path (str): Optional fallback CSV path
        limit (int): Max rows to return (after processing)

    Returns:
        pd.DataFrame
    """
    if from_api:
        if not categories:
            # Balanced defaults
            categories = [
                "cs.LG",
                "math.ST",
                "stat.ML",
                "eess.SP",
                "q-fin.EC",
                "q-bio.BM",
                "physics.gen-ph",
                "cs.MA",
                "cs.AI",
            ]

        df = fetch_arxiv_abstracts_by_category(categories, per_category=per_category)
    else:
        if not local_path:
            raise ValueError("local_path must be specified if from_api=False")
        print(f"[INFO] Loading dataset from: {local_path}")
        df = pd.read_csv(local_path)

    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=42).reset_index(drop=True)

    print(f"[INFO] Loaded {len(df)} records.")
    return df
