"""
Production Data Ingestion & Streaming Module (Method 3 Production).
Uses Polars for low-memory, high-throughput TSV ingestion.
Dynamically partitions by country to process US, France, and India in isolated streams.
"""

import os
import polars as pl

COUNTRY_STOPWORDS = {
    "US": {'inc', 'corp', 'llc', 'ltd', 'limited', 'private', 'incorporated', 'the', 'co', 'company', 'and', 'of', 'in', 'at', 'a', 'an', 'services', 'group', 'enterprises'},
    "France": {'sa', 'sarl', 'sas', 'eurl', 'sci', 'ste', 'societe', 'france', 'paris', 'pvt', 'ltd', 'inc', 'corp', 'llc', 'the', 'co', 'and', 'de', 'la', 'le', 'les', 'du', 'des', 'en', 'a'},
    "India": {'pvt', 'ltd', 'private', 'limited', 'enterprises', 'enterprise', 'company', 'co', 'the', 'and', 'of', 'in', 'at', 'services', 'india', 'solutions', 'industries'}
}

GLOBAL_STOPWORDS = {
    'pvt', 'ltd', 'inc', 'corp', 'llc', 'limited', 'private', 'incorporated',
    'the', 'co', 'company', 'and', 'of', 'in', 'at', 'a', 'an', 'services',
    'group', 'enterprises', 'sa', 'sarl', 'sas', 'eurl', 'sci', 'ste'
}


def load_partitioned_test_data(test_dir: str, country: str):
    """
    Loads test_source1, test_source2, and test_source3 strictly filtered by country.
    Uses Polars for fast loading and low memory overhead.
    """
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    print(f"\n[Loader] Streaming test data for country partition: '{country}'...")

    s1_df = pl.read_csv(
        s1_path,
        separator="\t",
        schema_overrides={
            "entity_id": pl.String,
            "business_name": pl.String,
            "business_address": pl.String,
            "country": pl.String
        }
    ).filter(pl.col("country") == country)

    s2_df = pl.read_csv(
        s2_path,
        separator="\t",
        schema_overrides={
            "entity_id": pl.String,
            "business_name": pl.String,
            "business_address": pl.String,
            "country": pl.String
        }
    ).filter(pl.col("country") == country)

    s3_df = pl.read_csv(
        s3_path,
        separator="\t",
        schema_overrides={
            "entity_id": pl.String,
            "business_name": pl.String,
            "business_address": pl.String,
            "country": pl.String
        }
    ).filter(pl.col("country") == country)

    print(f"  Loaded partition '{country}': S1={s1_df.height:,}, S2={s2_df.height:,}, S3={s3_df.height:,}")
    return s1_df, s2_df, s3_df


def get_all_test_countries(test_dir: str) -> list:
    """Detects unique countries present in test_source1.tsv."""
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    df = pl.read_csv(s1_path, separator="\t", columns=["country"])
    countries = df["country"].unique().to_list()
    # Order smallest country first for fast warming
    order = {"France": 0, "US": 1, "India": 2}
    return sorted(countries, key=lambda c: order.get(c, 99))
