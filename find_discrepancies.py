import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def find_discrepancies(input_file='joined_data.csv', output_file='school_discrepancies.csv'):
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku {input_file}. Uruchom najpierw join_data.py.")
        return

    # Filtrowanie: tylko szkoły z ponad 10 uczniami
    if 'liczba_uczniow' in df.columns:
        initial_count = len(df)
        df = df[df['liczba_uczniow'] > 10]
        print(f"Odfiltrowano {initial_count - len(df)} szkół z 10 lub mniej uczniami.")

    # Filtrowanie: wykluczenie szkół sportowych
    initial_count = len(df)
    sport_keywords = ['sportow', 'mistrzostwa sportowego']
    
    # Sprawdzamy w nazwie szkoły i w specyfice (jeśli dostępna)
    mask_sport = df['nazwa_szkoły'].str.contains('|'.join(sport_keywords), case=False, na=False)
    if 'specyfika' in df.columns:
        mask_sport = mask_sport | df['specyfika'].str.contains('|'.join(sport_keywords), case=False, na=False)
    
    df = df[~mask_sport]
    print(f"Odfiltrowano {initial_count - len(df)} szkół sportowych.")

    # 1. Rozbieżność: Średni wynik z j. polskiego vs Zdawalność ogólna
    # Obliczamy różnicę (punkty procentowe)
    df['diff_polski_zdawalnosc'] = df['PPP_mean'] - df['O_zdawalnosc']
    df['abs_diff_polski_zdawalnosc'] = df['diff_polski_zdawalnosc'].abs()

    # 2. Rozbieżność: Wynik maturalny vs Próg punktowy
    # Używamy regresji liniowej, aby wyznaczyć oczekiwany wynik na podstawie progu
    X = df[['threshold']].values
    y = df['PPP_mean'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    df['expected_PPP'] = model.predict(X)
    df['threshold_discrepancy'] = df['PPP_mean'] - df['expected_PPP']
    df['abs_threshold_discrepancy'] = df['threshold_discrepancy'].abs()

    # Znalezienie największych rozbieżności dla j. polskiego vs zdawalność
    top20_polish_pos = df.nlargest(20, 'diff_polski_zdawalnosc')
    top20_polish_neg = df.nsmallest(20, 'diff_polski_zdawalnosc')
    
    # Znalezienie największych rozbieżności dla progu (overperforming i underperforming)
    top20_overperforming = df.nlargest(20, 'threshold_discrepancy')
    top20_underperforming = df.nsmallest(20, 'threshold_discrepancy')

    # Wyświetlenie top 10 (dla czytelności w konsoli zostawiamy 10)
    print("\n" + "="*80)
    print("SZKOŁY O NAJWIĘKSZEJ POZYTYWNEJ ROZBIEŻNOŚCI: J. POLSKI VS ZDAWALNOŚĆ OGÓLNA")
    print("="*80)
    print(top20_polish_pos.head(10)[['nazwa_szkoły', 'city', 'PPP_mean', 'O_zdawalnosc', 'diff_polski_zdawalnosc']].to_string(index=False))

    print("\n" + "="*80)
    print("SZKOŁY O NAJWIĘKSZEJ NEGATYWNEJ ROZBIEŻNOŚCI: J. POLSKI VS ZDAWALNOŚĆ OGÓLNA")
    print("="*80)
    print(top20_polish_neg.head(10)[['nazwa_szkoły', 'city', 'PPP_mean', 'O_zdawalnosc', 'diff_polski_zdawalnosc']].to_string(index=False))

    print("\n" + "="*80)
    print("SZKOŁY OVERPERFORMING: WYNIK MATURALNY VS PRÓG PUNKTOWY (RESIDUUM > 0)")
    print("="*80)
    print(top20_overperforming.head(10)[['nazwa_szkoły', 'city', 'threshold', 'PPP_mean', 'threshold_discrepancy']].to_string(index=False))

    print("\n" + "="*80)
    print("SZKOŁY UNDERPERFORMING: WYNIK MATURALNY VS PRÓG PUNKTOWY (RESIDUUM < 0)")
    print("="*80)
    print(top20_underperforming.head(10)[['nazwa_szkoły', 'city', 'threshold', 'PPP_mean', 'threshold_discrepancy']].to_string(index=False))

    # Zapisanie wyników do plików po 20 rekordów
    top20_polish_pos.to_csv('top20_polish_high_vs_zdawalnosc.csv', index=False)
    top20_polish_neg.to_csv('bottom20_polish_low_vs_zdawalnosc.csv', index=False)
    top20_overperforming.to_csv('top20_overperforming_threshold.csv', index=False)
    top20_underperforming.to_csv('bottom20_underperforming_threshold.csv', index=False)

    print("\nZapisano 4 pliki CSV z top 20 szkołami dla każdego przypadku.")

    # Zapisanie wszystkich wyników do pliku
    df.to_csv(output_file, index=False)
    print(f"Pełne wyniki analizy zostały zapisane do pliku: {output_file}")

if __name__ == "__main__":
    find_discrepancies()
