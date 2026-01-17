import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import os

def perform_analysis(input_file='joined_data.csv'):
    if not os.path.exists(input_file):
        print(f"Błąd: Nie znaleziono pliku {input_file}. Uruchom najpierw join_data.py.")
        return

    df = pd.read_csv(input_file)
    
    # 1. Filtrowanie: tylko szkoły z ponad 10 uczniami
    if 'liczba_uczniow' in df.columns:
        initial_count = len(df)
        df = df[df['liczba_uczniow'] > 10]
        print(f"Odfiltrowano {initial_count - len(df)} szkół z 10 lub mniej uczniami.")

    # 2. Filtrowanie: wykluczenie szkół sportowych
    initial_count = len(df)
    sport_keywords = ['sportow', 'mistrzostwa sportowego']
    mask_sport = df['nazwa_szkoły'].str.contains('|'.join(sport_keywords), case=False, na=False)
    if 'specyfika' in df.columns:
        mask_sport = mask_sport | df['specyfika'].str.contains('|'.join(sport_keywords), case=False, na=False)
    
    df = df[~mask_sport]
    print(f"Odfiltrowano {initial_count - len(df)} szkół sportowych.")

    # 3. Regresja Liniowa: Próg vs J. Polski
    X = df[['threshold']].values
    y = df['PPP_mean'].values
    
    model = LinearRegression()
    model.fit(X, y)
    df['expected_PPP'] = model.predict(X)
    df['residuum'] = df['PPP_mean'] - df['expected_PPP']

    # 4. Identyfikacja anomalii
    overperforming = df.nlargest(5, 'residuum')
    underperforming = df.nsmallest(5, 'residuum')

    print("\nTop 5 Overperforming (Wynik wyższy niż sugeruje próg):")
    print(overperforming[['nazwa_szkoły', 'city', 'threshold', 'PPP_mean', 'residuum']].to_string(index=False))

    print("\nTop 5 Underperforming (Wynik niższy niż sugeruje próg):")
    print(underperforming[['nazwa_szkoły', 'city', 'threshold', 'PPP_mean', 'residuum']].to_string(index=False))

    # 5. Analiza cech (Organ prowadzący)
    if 'organ_prowadzacy' in df.columns:
        print("\nŚredni wynik wg organu prowadzącego:")
        print(df.groupby('organ_prowadzacy')['PPP_mean'].mean().sort_values(ascending=False))

    # 6. Wizualizacja
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='threshold', y='PPP_mean', hue='organ_prowadzacy' if 'organ_prowadzacy' in df.columns else None)
    plt.plot(df['threshold'], df['expected_PPP'], color='red', label='Linia trendu')
    plt.title('Zależność między progiem punktowym a wynikiem z j. polskiego')
    plt.xlabel('Średni próg punktowy')
    plt.ylabel('Średni wynik z j. polskiego')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('scatter_threshold_vs_polish.png')
    print("\nZapisano wykres: scatter_threshold_vs_polish.png")

    plt.figure(figsize=(10, 6))
    df['diff_polski_zdawalnosc'] = df['PPP_mean'] - df['O_zdawalnosc']
    sns.histplot(df['diff_polski_zdawalnosc'], kde=True)
    plt.title('Rozkład różnicy: J. Polski vs Zdawalność ogólna')
    plt.xlabel('Różnica (p.p.)')
    plt.savefig('polish_vs_overall_dist.png')
    print("Zapisano wykres: polish_vs_overall_dist.png")

if __name__ == "__main__":
    perform_analysis()
