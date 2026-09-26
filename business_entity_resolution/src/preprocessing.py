import re
import pandas as pd

# Legal suffixes / honorifics that vary between sources and carry no identity signal
NAME_STOPWORDS = [
    'ltd', 'limited', 'pvt', 'private', 'inc', 'incorporated', 'corp', 'corporation',
    'llc', 'l l c', 'llp', 'co', 'company', 'plc', 'sa', 'sas', 'sarl', 'eurl', 'gmbh',
    'dr', 'mr', 'mrs', 'ms', 'smt', 'shri', 'sri', 'm s', 'the', 'dba', 'and',
]
NAME_STOP_RE = re.compile(r'\b(?:' + '|'.join(re.escape(w) for w in NAME_STOPWORDS) + r')\b')
DOMAIN_RE = re.compile(r'^(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.(?:com|net|org|in|co\.in|fr|us|biz|info|io)\b.*$')

ADDR_ABBREV = {
    'rd': 'road', 'st': 'street', 'str': 'street', 'ave': 'avenue', 'av': 'avenue',
    'ct': 'court', 'cir': 'circle', 'dr': 'drive', 'ln': 'lane', 'blvd': 'boulevard',
    'hwy': 'highway', 'pkwy': 'parkway', 'pl': 'place', 'sq': 'square', 'ste': 'suite',
    'fl': 'floor', 'flr': 'floor', 'apt': 'apartment', 'bldg': 'building', 'opp': 'opposite',
    'nr': 'near', 'mg': 'marg', 'n': 'north', 's': 'south', 'e': 'east', 'w': 'west',
}
ADDR_ABBREV_RE = re.compile(r'\b(' + '|'.join(ADDR_ABBREV) + r')\b')
LEADING_ZERO_RE = re.compile(r'\b0+(\d)')


def _basic_clean(s: pd.Series) -> pd.Series:
    """Vectorized: strip accents, drop non-latin script, lowercase, keep [a-z0-9 ]."""
    s = s.fillna('').str.normalize('NFKD').str.lower()
    s = s.str.replace('&', ' and ', regex=False)
    s = s.str.replace(r'[^a-z0-9]+', ' ', regex=True)
    return s.str.strip()


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a compact frame: name, addr (cleaned)."""
    raw_name = df['business_name'].fillna('').str.lower().str.strip()
    domain = raw_name.str.extract(DOMAIN_RE, expand=False)

    name = _basic_clean(df['business_name'])
    name = name.where(domain.isna(), domain.fillna('').str.replace('-', ' ', regex=False))
    name = name.str.replace(NAME_STOP_RE, ' ', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()

    addr = _basic_clean(df['business_address'])
    addr = addr.str.replace(LEADING_ZERO_RE, r'\1', regex=True)
    addr = addr.str.replace(ADDR_ABBREV_RE, lambda m: ADDR_ABBREV[m.group(1)], regex=True)

    return pd.DataFrame({'name': name.values, 'addr': addr.values})
