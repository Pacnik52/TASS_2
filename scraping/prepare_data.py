import pandas as pd
import numpy as np
import re
import unicodedata
from pathlib import Path
from typing import Dict
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz

# ============================================================================
# FUNKCJE DO NORMALIZACJI TEKSTU
# ============================================================================

def remove_polish(text):
    if pd.isna(text):
        return text
    text = str(text)
    polish_map = {
        'ą': 'a', 'ć': 'c', 'ę': 'e', 'ł': 'l', 'ń': 'n',
        'ó': 'o', 'ś': 's', 'ź': 'z', 'ż': 'z',
        'Ą': 'a', 'Ć': 'c', 'Ę': 'e', 'Ł': 'l', 'Ń': 'n',
        'Ó': 'o', 'Ś': 's', 'Ź': 'z', 'Ż': 'z'
    }
    trans_table = str.maketrans(polish_map)
    text = text.translate(trans_table)
    return text

def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.lower() # małe litery
    text = remove_polish(text) # brak polskich znaków
    text = re.sub(r'[^\w\s]', ' ', text)  # zamiana interpunkcji na spacje
    text = re.sub(r'\s+', ' ', text).strip()  # redukcja spacji
    return text

def normalize_school_name(text):
    """Specjalna normalizacja dla nazw szkół do łączenia zbiorów."""
    norm = normalize_text(text)
    
    # Standaryzacja skrótów i nazw typów szkół
    norm = norm.replace('liceum ogolnoksztalcace', 'lo')
    norm = norm.replace('technikum', 'tech')
    
    # Usuwanie słów zakłócających
    words_to_remove = ['im', 'nr', 'z oddzialami', 'integracyjnymi', 'dwujezycznymi', 'sportowymi', 'mistrzostwa', 'sportowego', 'dla doroslych', 'zespol szkol']
    for word in words_to_remove:
        norm = re.sub(r'\b' + word + r'\b', '', norm)
    
    # Usuwanie wszystkich spacji
    norm = re.sub(r'\s+', '', norm)
    return norm

def fix_city_names(city):
    if pd.isna(city):
        return city
    city = str(city).strip()
    
    city_mapping = {
        'Białymstoku': 'Białystok',
        'Bielsku-Białej': 'Bielsko-Biała',
        'Bydgoszczy': 'Bydgoszcz',
        'Częstochowie': 'Częstochowa',
        'Gdańsku': 'Gdańsk',
        'Gdyni': 'Gdynia',
        'Katowicach': 'Katowice',
        'Kielcach': 'Kielce',
        'Krakowie': 'Kraków',
        'Lublinie': 'Lublin',
        'Łodzi': 'Łódź',
        'Olsztynie': 'Olsztyn',
        'Opolu': 'Opole',
        'Poznaniu': 'Poznań',
        'Rzeszowie': 'Rzeszów',
        'Szczecinie': 'Szczecin',
        'Toruniu': 'Toruń',
        'Warszawie': 'Warszawa',
        'Wroclaw': 'Wrocław',
        'Zielonej': 'Zielona Góra',
    }
    
    return city_mapping.get(city, city)

def extract_street(address):
    """Wyodrębnia ulicę z pełnego adresu"""
    if pd.isna(address):
        return None
    address = str(address)
    
    # Szukaj wzorca: ul./al./pl. + nazwa ulicy
    match = re.search(r'(ul\.|al\.|pl\.)\s+([A-Za-ząćęłńóśźż\s-]+?)(?:\s+\d|$)', address)
    if match:
        return match.group(1) + ' ' + match.group(2).strip()
    
    # Jeśli nie znaleźli - zwróć None
    return None

# ============================================================================
# ETAP 1: PRZETWORZENIE WYNIKÓW MATURALNYCH
# ============================================================================

def process_matura_files():
    print("\n=== ETAP 1: Przetworzenie wyników maturalnych ===")
    
    matura_files = [
        'EM2023 - 09.2023.csv',
        'EM2023 - 09.2024.csv',
        'EM2023 - 09.2025.csv'
    ]
    
    dfs = []
    for file in matura_files:
        filepath = Path(file)
        if not filepath.exists():
            print(f"Plik nie znaleziony: {file}")
            continue
        
        # Wydobycie roczmika z nazwy pliku
        year_match = re.search(r'09\.(\d{4})', file)
        year = int(year_match.group(1)) if year_match else None
        
        print(f"Wczytanie: {file} (rok: {year})")
        df = pd.read_csv(filepath, sep=';', encoding='utf-8')
        
        # Dodaj kolumnę roku
        df['rok_wyniku'] = year

        # Normalizacja danych w trakcie wczytywania
        df['nazwa_szkoly'] = df['nazwa_szkoly'].apply(normalize_school_name)
        df['nazwa_szkoly'] = df['nazwa_szkoly'].apply(normalize_text)
        df['miejscowosc'] = df['miejscowosc'].apply(normalize_text)
        
        df['ulica_nr'] = df['ulica_nr'].apply(normalize_text)
        
        dfs.append(df)
    
    matura_df = pd.concat(dfs, ignore_index=True)
    print(f"Połączone: {len(matura_df)} wierszy")

    columns_to_keep = [
        'rok_wyniku', 'wojewodztwo', 'powiat', 'gmina', 'typ_gminy',
        'RSPO', 'nazwa_szkoly', 'miejscowosc', 'ulica_nr',
        'typ_placowki', 'czy_publiczna',
        'O_otrzymali_swiadectwo_dojrzałosci', 'O_liczba_zdających', 'O_zdawalnosc',
        'PPP_liczba_zdajacych', 'PPP_liczba_laureatow', 'PPP_zdawalnosc',
        'PPP_mean', 'PPP_median'
    ]
    
    matura_df = matura_df[columns_to_keep]
    
    # Filtrowanie szkół nielicealnych
    matura_df = matura_df[matura_df['typ_placowki'].str.contains('liceum', case=False, na=False)]
    print(f"Po filtracji tylko licea: {len(matura_df)} wierszy")

    print(f"Matura head:\n {matura_df.head(10)}")
    
    return matura_df

# ============================================================================
# ETAP 2: PRZETWORZENIE RSPO
# ============================================================================

def process_rspo():
    print("\n=== ETAP 2: Przetworzenie RSPO ===")
    
    filepath = Path('rspo_2025.csv')
    if not filepath.exists():
        print(f"Plik nie znaleziony: {filepath}")
        return None
    
    print("Wczytanie: rspo_2025.csv")
    rspo_df = pd.read_csv(filepath, sep=';', encoding='utf-8', on_bad_lines='skip', engine='python')
    
    print(f"Wczytane: {len(rspo_df)} wierszy")
    
    columns_to_keep = [
        'RSPO', 'typ_placowki', 'nazwa','wojewodztwo', 'powiat', 'gmina', 'miejscowosc',
        'rodzaj_miejscowości', 'ulica', 'numer_budynku',
        'kod_pocztowy', 'publicznosc_status', 'kategoria_uczniow',
        'data_zalozenia', 'data_rozpoczecia_dzialalnosci', 'miejsce_w_strukturze', 
        'typ_podmiotu_nadrzednego','liczba_uczniow','tereny_sportowe','jezyki_nauczane',
        'czy_zatrudnia_logopede','czy_zatrudnia_psychologa','czy_zatrudnia_pedagoga',
        'oddzialy_podstawowe_wg_specyfiki','oddzialy_dodatkowe'
    ]
    rspo_df = rspo_df[columns_to_keep]
    
    # Filtrowanie szkół nielicealnych
    rspo_df = rspo_df[rspo_df['typ_placowki'].str.contains('liceum', case=False, na=False)]
    print(f"Po filtracji tylko licea: {len(rspo_df)} wierszy")

    # Normalizacja danych
    rspo_df['nazwa'] = rspo_df['nazwa'].apply(normalize_text)
    rspo_df['nazwa'] = rspo_df['nazwa'].apply(normalize_school_name)
    rspo_df['miejscowosc'] = rspo_df['miejscowosc'].apply(normalize_text)
    rspo_df['ulica'] = rspo_df['ulica'].apply(normalize_text)

    print(f"RSPO head:\n {rspo_df.head(10)}")
    
    return rspo_df

# ============================================================================
# ETAP 3: PRZETWORZENIE PROGÓW REKRUTACYJNYCH
# ============================================================================

def process_thresholds():
    print("\n=== ETAP 3: Przetworzenie progów rekrutacyjnych ===")
    
    filepath = Path('school_thresholds_otouczelnie_clean.csv')
    if not filepath.exists():
        print(f"Plik nie znaleziony: {filepath}")
        return None
    
    print(f"Wczytanie: {filepath}")
    thresholds_df = pd.read_csv(filepath, encoding='utf-8')
    print(f"Wczytane: {len(thresholds_df)} wierszy")
    
    # Normalizacja danych
    thresholds_df['school'] = thresholds_df['school'].apply(normalize_school_name)
    thresholds_df['city'] = thresholds_df['city'].apply(fix_city_names)
    thresholds_df['city'] = thresholds_df['city'].apply(normalize_text)
    
    # Ekstrakcja ulicy
    thresholds_df['ulica'] = thresholds_df['address'].apply(extract_street)
    thresholds_df['ulica'] = thresholds_df['ulica'].apply(normalize_text)
    
    # Ekstrakcja kodu pocztowego
    thresholds_df['kod_pocztowy'] = thresholds_df['address'].apply(lambda x: re.search(r'\d{2}-\d{3}', str(x)).group(0) if re.search(r'\d{2}-\d{3}', str(x)) else None)
    
    # Konwersja threshold na float
    thresholds_df['threshold'] = pd.to_numeric(thresholds_df['threshold'], errors='coerce')
    
    # Wyznaczenie średnich i median progów dla każdej szkoły w danym roku
    thresholds_agg = thresholds_df.groupby(['year', 'school', 'city', 'ulica', 'kod_pocztowy', 'address'])['threshold'].agg(['mean', 'median', 'count']).reset_index() 
    thresholds_agg.rename(columns={
        'mean': 'avg_threshold',
        'median': 'median_threshold',
        'count': 'count_courses'
    }, inplace=True)

    # Konwersja formatu rocznika
    thresholds_agg['year'] = thresholds_agg['year'].str[:4].astype(int)
    
    print(f"Po agregacji: {len(thresholds_agg)} szkół")

    print(f"thresholds head:\n {thresholds_agg.head(10)}")
    return thresholds_agg

# ============================================================================
# ETAP 4: ZŁĄCZANIE DANYCH
# ============================================================================

def merge_datasets(matura_df, rspo_df, thresholds_agg):
    print("\n=== ETAP 4: Złączenie danych===")
    print("a) Złączenie matura + RSPO")
    
    # Merge z RSPO 
    matura_rspo = rspo_df.merge(
        matura_df,
        on='RSPO',
        how='left',
        suffixes=('_matura', '_rspo')
    )

    print(f"    Po złączeniu: {len(matura_rspo)} wierszy")
    # print(f"Kolumny po fuzji: {matura_rspo.columns.tolist()}")
    # print(f"Matura_RSPO head:\n {matura_rspo.head(10)}")
    print(f"   matura_rspo rows: {len(matura_rspo)}")
    print(f"   Unikalne szkół w zbiorze matura_rspo: {len(matura_rspo[['nazwa_szkoly', 'miejscowosc_rspo']].drop_duplicates())}")
    print(f"   thresholds_agg rows: {len(thresholds_agg)}")
    print(f"   Unikalne szkół w zbiorze progi rekrutacyjne: {len(thresholds_agg[['school', 'city']].drop_duplicates())}")
    
    print("b) Złączenie matura_RSPO + progi punktowe")
    final_df = matura_rspo.merge(
        thresholds_agg[['school', 'city', 'year', 'avg_threshold', 'median_threshold', 'count_courses']].drop_duplicates(subset=['school', 'city', 'year']),
        left_on=['nazwa_szkoly', 'miejscowosc_rspo', 'rok_wyniku'],
        right_on=['school', 'city', 'year'],
        how='left'
    )

    print(f"   Po złączeniu: {len(final_df)} wierszy")

    matched = final_df[final_df['avg_threshold'].notna()]
    print(f"   Z informacją o progach rekrutacyjnych: {len(matched)} wierszy ({100*len(matched)/len(final_df):.1f}%)")
    
    # Fuzzy matching dla unmatched szkół
    unmatched = final_df[final_df['avg_threshold'].isna()].copy()
    if not unmatched.empty:
        print(f"   Fuzzy matching dla {len(unmatched)} szkół")
        
        possible_matches = {}
        for (city, year), group in thresholds_agg.groupby(['city', 'year']):
            possible_matches[(city, year)] = dict(zip(group['school'], group[['ulica', 'kod_pocztowy', 'avg_threshold', 'median_threshold', 'count_courses']].to_dict('records')))
        
        def find_fuzzy_match(row):
            city_year = (row['miejscowosc_rspo'], row['rok_wyniku'])
            if city_year not in possible_matches:
                return pd.Series([None, None, None])
            
            school_name = row['nazwa_szkoly']
            best_match = None
            best_score = 0
            for thresh_school in possible_matches[city_year]:
                score = fuzz.ratio(school_name, thresh_school)
                if score > best_score and score > 80:
                    best_score = score
                    best_match = possible_matches[city_year][thresh_school]
            
            if best_match:
                return pd.Series([best_match['avg_threshold'], best_match['median_threshold'], best_match['count_courses']])
            else:
                return pd.Series([None, None, None])
        
        fuzzy_results = unmatched.apply(find_fuzzy_match, axis=1)
        fuzzy_results.columns = ['avg_threshold', 'median_threshold', 'count_courses']
        
        for idx in unmatched.index:
            if not fuzzy_results.loc[idx].isna().all():
                final_df.loc[idx, ['avg_threshold', 'median_threshold', 'count_courses']] = fuzzy_results.loc[idx]
        
        matched_after_fuzzy = final_df[final_df['avg_threshold'].notna()]
        print(f"   Znaleziono: {len(matched_after_fuzzy)} wierszy")
    
    # Filtrowanie szkół bez progów w którymkolwiek roku
    final_df = final_df[final_df.groupby('RSPO')['avg_threshold'].transform('any')]
    print(f"   Po wyfiltowaniu szkół bez danych o progach: {len(final_df)} wierszy")
    unique_schools_with_thresholds = len(final_df[['RSPO']].drop_duplicates())
    print(f"   Znaleziono unikalnych szkół: {unique_schools_with_thresholds}")

    
    # Przywracanie nazw kolumn
    cols_to_restore = ['wojewodztwo_matura', 'powiat_matura', 'gmina_matura', 'miejscowosc_matura', 
                       'typ_gminy_matura', 'rodzaj_placowki_matura',
                       'czy_publiczna_matura', 'nazwa_szkoly_matura']
    for col in cols_to_restore:
        if col in final_df.columns:
            new_col = col.replace('_matura', '')
            final_df[new_col] = final_df[col]
    
    # Usuń zbędne kolumny z fuzji
    cols_to_drop = [col for col in final_df.columns if col.endswith(('_matura', '_rspo'))]
    final_df = final_df.drop(columns=cols_to_drop, errors='ignore')
    
    # print(f"Kolumny po fuzji: {final_df.columns.tolist()}")
    # print(f"final df head:\n {final_df.head(10)}")
    
    print(f"    Po fuzji z progami: {len(final_df)} wierszy")
    
    return final_df

# ============================================================================
# ETAP 5: Analiza danych
# ============================================================================

def clean(df):
    print("\n=== ETAP 5: Analiza danych ===")

    # Sprawdzenie braków w danych
    print("\nAnaliza braków danych:")
    nan_counts = df.isnull().sum()
    if (nan_counts > 0).sum() > 0:
        for col in nan_counts[nan_counts > 0].index:
            if col in ['avg_threshold', 'median_threshold', 'count_courses']:
                print(f"   {col}: {nan_counts[col]} ({100*nan_counts[col]/len(df):.1f}%)")
            else:
                print(f"   {col}: {nan_counts[col]} ({100*nan_counts[col]/len(df):.1f}%)")
    else:
        print("   Brak braków danych")
    
    # Sortowanie po RSPO i roku
    df = df.sort_values(['RSPO', 'rok_wyniku']).reset_index(drop=True)
    
    return df

# ============================================================================
# ETAP 6: FORMATOWANIE DANYCH
# ============================================================================
# łaczenie danych każdej szkoły z wielu roczników w jeden wiersz

def pivot_by_year(df):
    print("\n=== ETAP 6: Formatowanie danych ===")
    
    # Zmiana nazw kolumn PPP_ na polski_
    df = df.rename(columns={col: col.replace('PPP_', 'polski_') for col in df.columns if col.startswith('PPP_')})
    
    # Kolumny identyfikacyjne
    id_cols = ['nazwa_szkoly', 'ulica_nr', 'numer_budynku', 'numer_lokalu', 'kod_pocztowy',
               'wojewodztwo', 'powiat', 'gmina', 'typ_gminy', 'kod_teryt_gminy', 'miejscowosc', 
               'rodzaj_miejscowości', 'typ_placowki', 'rodzaj_placowki', 'czy_publiczna', 'publicznosc_status',
               'kategoria_uczniow', 'specyfika_placowki', 'data_zalozenia', 'data_rozpoczecia_dzialalnosci',
               'data_likwidacji', 'typ_organu_prowadzacego', 'typ_podmiotu_nadrzednego', 'liczba_uczniow',
               'tereny_sportowe', 'jezyki_nauczane', 'czy_zatrudnia_logopede', 'czy_zatrudnia_psychologa',
               'czy_zatrudnia_pedagoga', 'oddzialy_podstawowe_wg_specyfiki', 'oddzialy_dodatkowe',
               'zawod', 'zawod_artystyczny', 'zawod_w_KPSS', 'dziedzina_BCU']
    id_cols = [col for col in id_cols if col in df.columns]
    
    # Kolumny per rok
    year_value_cols = ['O_otrzymali_swiadectwo_dojrzałosci', 'O_liczba_zdających', 'O_zdawalnosc',
                       'polski_liczba_zdajacych', 'polski_liczba_laureatow', 'polski_zdawalnosc',
                       'polski_mean', 'polski_median', 'avg_threshold', 'median_threshold']
    year_value_cols = [col for col in year_value_cols if col in df.columns]
    
    # Zbieranie stałych danych szkoły
    metadata_df = df[['RSPO'] + id_cols].drop_duplicates(subset=['RSPO'], keep='first')
    
    # Teraz pivot wartości per rok
    result_df = metadata_df.copy()
    
    for year in sorted(df['rok_wyniku'].unique()):
        year_str = str(year)[2:]  # 2023 -> 23
        year_df = df[df['rok_wyniku'] == year][['RSPO'] + year_value_cols].copy()
        
        # Dodanie roczników do nazw kolumn
        rename_dict = {col: f'{year_str}_{col}' for col in year_value_cols}
        year_df = year_df.rename(columns=rename_dict)

        result_df = result_df.merge(year_df, on='RSPO', how='left')
    
    print(f"    Unikalnych szkół po formatowaniu: {len(result_df)}")
    
    # Organizacja kolumn
    # 1. metadane
    metadata_cols = ['RSPO', 'nazwa_szkoly', 'publicznosc_status', 'wojewodztwo', 'powiat', 'gmina', 
                     'typ_gminy', 'miejscowosc', 'rodzaj_miejscowości', 'ulica_nr', 'numer_budynku', 
                     'kod_pocztowy']
    metadata_cols = [col for col in metadata_cols if col in result_df.columns]
    # 2. dane o wynikach maturalnych i progach
    year_order_template = [
        'O_otrzymali_swiadectwo_dojrzałosci',
        'O_liczba_zdających',
        'O_zdawalnosc',
        'polski_liczba_zdajacych',
        'polski_liczba_laureatow',
        'polski_zdawalnosc',
        'polski_mean',
        'polski_median',
        'avg_threshold',
        'median_threshold'
    ]
    year_cols = []
    for year in sorted(df['rok_wyniku'].unique()):
        year_str = str(year)[2:]
        for template_col in year_order_template:
            col_name = f'{year_str}_{template_col}'
            if col_name in result_df.columns:
                year_cols.append(col_name)
    # 3. inne 
    other_cols = [col for col in result_df.columns if col not in metadata_cols and col not in year_cols]
    
    final_order = metadata_cols + year_cols + other_cols
    final_order = [col for col in final_order if col in result_df.columns]
    
    result_df = result_df[final_order]
    print(f"    Kolumny zorganizowane: {len(result_df.columns)}")
    print(f"    Kolumny: {result_df.columns.tolist()}")

    print(f"Final df head:\n {result_df.head(10)}")
    
    return result_df

# ============================================================================
# MAIN
# ============================================================================

def main():
    try:
        # Zbieranie i wstepne przetwanianie zbiorów
        matura_df = process_matura_files()
        rspo_df = process_rspo()
        thresholds_agg = process_thresholds()
        
        if matura_df is None or rspo_df is None or thresholds_agg is None:
            print("Nie udało się wczytać wszystkich plików!")
            return
        
        # Łączenie zbiorów
        final_df = merge_datasets(matura_df, rspo_df, thresholds_agg)
        
        # Czyszczenie i analiza danych
        final_df = clean(final_df)
        
        # Formatowanie końcowego zbioru danych
        final_df = pivot_by_year(final_df)
        
        # Zapis
        output_file = '../datasets/data.csv'
        print(f"\nZapis do pliku: {output_file}")
        final_df.to_csv(output_file, index=False, encoding='utf-8', sep=';')
        print(f"    Zapisano {len(final_df)} wierszy")
        return final_df
        
    except Exception as e:
        print(f"\nBŁĄD: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    df = main()
