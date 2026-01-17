# Dokumentacja

## Wykorzystane pakiety python 3.x
- pandas 
- numpy
- fuzzywuzzy
- difflib
- pathlib
- requests
- BeautifulSoup

## scraper.py - scrapowanie danych z otouczelnie.pl
Skrypt pobiera progi punktowe dla liceów w latach 2023-2025
1. Przechodzi przez zdefiniowane lata rekrutacji
2. Dla każdego roku pobiera listę dostępnych miast, a następnie listę szkół w każdym mieście
3. Odwiedza stronę każdej szkoły i wyciąga dane z tabeli progów
4. Pomiędzy kolejnymi wywołaniami stosuje opóźnienie, aby uniknąć błędów
5. Wynikiem jest plik school_thresholds_otouczelnie_raw.csv

## clean_csv.py
Skrypt wczytuje surowy plik z zescrapowanymi danymi i zapisuje go do pliku *clean.csv z usuniętymi zbędnymi spacjami oraz znakami nowej linii

## prepare_data.py - plik przetwarzający dane do analizy
Przed uruchomieniem skryptu należało przygotować pliki CSV z rejestru RSPO oraz OKE. Plik school_thesholds_otouczelnie_clean.csv zawiera dane zescrapowane ze strony otouczelnie.pl

- funkcje do normalizacji tekstu: ze względu na niespójność nazewnictwa w źródłach, dodano funkcje:
1. remove_polish(text): usuwa polskie znaki 
2. normalize_text(text): zamienia tekst na małe litery, usuwa interpunkcję i zbędne spacje
3. normalize_school_name(text): standaryzuje skróty, usuwa słowa utrudniające parowanie 
4. fix_city_names(city): mapuje odmienione nazwy miast na mianownik

# Pipeline:
1. process_matura_files() - przetworzenie wyników maturalnych: wczytanie plików z lat 2023-2025, odfiltrowanie liceów, normalizuje nazwy szkół i miejscowości, wybiera kluczowe kolumny
2. process_rspo() - przetworzenie danych RSPO: wczytanie rejestru szkół, odfiltrowanie liceów, wybranie metadanych, normalizacja danych adresowych
3. process_thresholds() - przetworzenie progów punktowych: wczytanie danych, agregacja różnych klas danej szkoły i wyliczenie średniej oraz mediany progów w danym roku
4. merge_datasets(matura_df, rspo_df, thresholds_agg) - złączenie danych: po unikalnym numerze RSPO, po kluczu [nazwa szkoły] + [miasto] + [rok] (exact match) oraz fuzzy match z wykorzystaniem algorytmu Levenshteina (fuzz.ratio)
5. clean(df) - czyszczenie danych: analiza NaN, sortowanie rekordów
6. pivot_by_year(df) - formatowanie danych, uzyskanie jednego wiersza kla każdej szkoły
7. main() - wynikiem skryptu jest plik data.csv, który zawiera kolumny metadanych z prefiksami lat, każdy wiersz ma unikalny RSPO