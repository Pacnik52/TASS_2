import pandas as pd
from thefuzz import process, fuzz
import os

FILES_MATURA = ['EM2023 - 09.2023.csv', 'EM2023 - 09.2024.csv', 'EM2023 - 09.2025.csv']
FILE_RSPO = '../results/analysis/rspo_2025_fixed.csv'
FILE_THRESHOLDS = '../results/scraping/school_thresholds_otouczelnie_clean.csv'
OUTPUT_FILE = '../results/analysis/joined_data.csv'

def load_matura_data(files):
    dfs = []
    for f in files:
        if os.path.exists(f):
            print(f"Loading {f}...")
            df = pd.read_csv(f, sep=';', encoding='utf-8-sig', low_memory=False)
            cols = ['RSPO', 'nazwa_szkoły', 'miejscowosc', 'PPP_mean', 'O_zdawalnosc']
            available_cols = [c for c in cols if c in df.columns]
            df = df[available_cols].copy()
            for col in ['PPP_mean', 'O_zdawalnosc']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.').str.replace('%', ''), errors='coerce')
            dfs.append(df)
    
    if not dfs:
        return pd.DataFrame()
    
    combined = pd.concat(dfs)
    # Aggregate by RSPO - mean across years
    # We also keep nazwa_szkoły and miejscowosc for matching
    return combined.groupby('RSPO').agg({
        'nazwa_szkoły': 'first',
        'miejscowosc': 'first',
        'PPP_mean': 'mean',
        'O_zdawalnosc': 'mean'
    }).reset_index()

def load_rspo_data(file):
    if not os.path.exists(file):
        print(f"RSPO file {file} not found.")
        return pd.DataFrame()
    print(f"Loading {file}...")
    df = pd.read_csv(file, sep=';', encoding='utf-8-sig', low_memory=False)
    
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.replace('^="', '', regex=True).str.replace('"$', '', regex=True)

    cols_map = {
        'Numer RSPO': 'RSPO',
        'Typ organu prowadzącego': 'organ_prowadzacy',
        'Województwo': 'wojewodztwo',
        'Liczba uczniów': 'liczba_uczniow',
        'Miejscowość': 'miejscowosc_rspo',
        'Nazwa': 'nazwa_rspo',
        'Oddziały podstawowe wg specyfiki': 'specyfika'
    }
    df = df.rename(columns=cols_map)
    df['RSPO'] = pd.to_numeric(df['RSPO'], errors='coerce')
    available_cols = [c for c in cols_map.values() if c in df.columns]
    return df[available_cols]

def load_thresholds(file):
    if not os.path.exists(file):
        print(f"Thresholds file {file} not found.")
        return pd.DataFrame()
    print(f"Loading {file}...")
    df = pd.read_csv(file)
    df['threshold'] = pd.to_numeric(df['threshold'], errors='coerce')

    school_avg = df.groupby(['school', 'city']).agg({
        'threshold': 'mean',
        'address': 'first'
    }).reset_index()
    return school_avg

def clean_city_name(city):
    # Replace city names "Białymstoku" -> "Białystok"
    replacements = {
        'Białymstoku': 'Białystok',
        'Warszawie': 'Warszawa',
        'Krakowie': 'Kraków',
        'Wrocławiu': 'Wrocław',
        'Poznaniu': 'Poznań',
        'Gdańsku': 'Gdańsk',
        'Szczecinie': 'Szczecin',
        'Bydgoszczy': 'Bydgoszcz',
        'Lublinie': 'Lublin',
        'Katowicach': 'Katowice',
        'Łodzi': 'Łódź',
        'Rzeszowie': 'Rzeszów',
        'Opolu': 'Opole',
        'Kielcach': 'Kielce',
        'Olsztynie': 'Olsztyn',
        'Toruniu': 'Toruń',
        'Gorzowie Wielkopolskim': 'Gorzów Wielkopolski',
        'Zielonej Górze': 'Zielona Góra',
        'Wroclaw': 'Wrocław'
    }
    return replacements.get(city, city)

def fuzzy_merge(df_matura, df_thresholds):
    print("Starting fuzzy merge...")
    df_matura['match_key'] = df_matura['nazwa_szkoły'].str.lower()
    df_thresholds['city_clean'] = df_thresholds['city'].apply(clean_city_name)
    
    results = []
    
    total = len(df_thresholds)
    for idx, row in df_thresholds.iterrows():
        if idx % 50 == 0:
            print(f"Processing {idx}/{total}...")
        
        city = row['city_clean']
        school_name = row['school'].lower()
        
        # Filter by city
        possible_matches = df_matura[df_matura['miejscowosc'].str.contains(city, case=False, na=False)].copy()
        
        if possible_matches.empty:
            # Try a broader search if city name is problematic
            possible_matches = df_matura.copy()

        choices = possible_matches['match_key'].tolist()
        if not choices:
            continue
            
        best_match, score = process.extractOne(school_name, choices, scorer=fuzz.token_sort_ratio)
        
        if score > 70:
            matched_row = possible_matches[possible_matches['match_key'] == best_match].iloc[0]
            results.append({
                'threshold_idx': idx,
                'RSPO': matched_row['RSPO'],
                'match_score': score,
                'matched_nazwa': matched_row['nazwa_szkoły']
            })
    
    match_df = pd.DataFrame(results)
    if match_df.empty:
        return pd.DataFrame()
    
    merged = pd.merge(df_thresholds.reset_index(), match_df, left_on='index', right_on='threshold_idx')
    final_merged = pd.merge(merged, df_matura, on='RSPO', suffixes=('_thresh', '_matura'))
    return final_merged

def main():
    df_matura = load_matura_data(FILES_MATURA)
    df_thresholds = load_thresholds(FILE_THRESHOLDS)
    df_rspo = load_rspo_data(FILE_RSPO)
    
    if df_matura.empty or df_thresholds.empty:
        print("Required data missing.")
        return

    df_joined = fuzzy_merge(df_matura, df_thresholds)
    
    if df_joined.empty:
        print("No matches found.")
        return
        
    if not df_rspo.empty:
        print("Adding RSPO extra data...")
        df_joined = pd.merge(df_joined, df_rspo, on='RSPO', how='left')
    
    print(f"Saving {len(df_joined)} joined records to {OUTPUT_FILE}")
    df_joined.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print("Done.")

if __name__ == "__main__":
    main()
