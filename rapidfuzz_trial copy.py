
import pandas as pd
import re
from rapidfuzz import process, fuzz
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

file1 = "..."
file2 = "..."
data1 = pd.read_csv(file1)
data2 = pd.read_csv(file2)

data1['norm_adr'] = data1['address'].str.lower().str.strip()
data2['norm_adr'] = data2['address'].str.lower().str.strip()

def get_kanton(string):
    match = re.findall(r'[A-Z]{2}', string)
    return match[0] if match else None

def del_kanton(string, up_char):
    if up_char:
        return string.replace(up_char, '', 1).strip()
    return string

data1['kanton_code'] = data1['address'].apply(get_kanton)
data2['kanton_code'] = data2['address'].apply(get_kanton)

kanton_dict = data2.groupby('kanton_code')['norm_adr'].apply(list).to_dict()

def match_address(row):
    addr1 = row['norm_adr']
    kcode1 = row['kanton_code']
    
    addr1_norm = del_kanton(addr1, kcode1)

    pot_match = kanton_dict.get(kcode1, [])

    if pot_match:
        matches = process.extract(
            addr1_norm,
            pot_match,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=80
        )
        if matches:
            best_match_idx = pot_match.index(matches[0][0])
            match_adr_id2 = data2.iloc[best_match_idx]['adr_id2']
            return {
                'adr_id1': row['adr_id1'],
                'adr_id2': match_adr_id2,
                'Address1': row['address'],
                'Address2': matches[0][0],
                'Score': matches[0][1]
            }
    
    return {
        'adr_id1': row['adr_id1'],
        'adr_id2': None,
        'Address1': row['address'],
        'Address2': None,
        'Score': None
    }

if __name__ == "__main__":
    num_cores = cpu_count()

    rows = data1.to_dict('records')

    res = list(tqdm(Pool(num_cores).imap(match_address, rows), total=len(rows), desc="Matching addresses"))

    res_df = pd.DataFrame(res)

    num_matches = res_df['Address2'].notnull().sum()
    tot_adr = len(res_df)
    match_perc = (num_matches / tot_adr) * 100

    res_df[['adr_id1', 'adr_id2']].to_csv('matched_adr.csv', index=False)

    print(res_df[['Address1', 'Address2']].head(30))

    print(f"\nTotal # of matches found: {num_matches}")
    print(f"% of matches: {match_perc:.2f}%")