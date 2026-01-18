import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# Base URL
base_url = "https://www.otouczelnie.pl/progi-punktowe/licea/ogolnopolskie/"

# Years to scrape
years = ["2023-2024", "2024-2025", "2025-2026"]

# Function to get cities for a year
def get_cities(year):
    url = base_url + year
    response = requests.get(url, verify=False)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find links to cities
    city_links = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        if '/miasto/' in href and year in href:
            city_links.append(href)
    print(f"Found {len(city_links)} city links")
    # city_links = city_links[:2] 
    return city_links

# Function to scrape data for a city and year
def scrape_city_data(city_link, year):
    url = city_link
    response = requests.get(url, verify=False)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract city name from title
    title = soup.find('title').text
    # Title like "Progi punktowe do liceów w Białymstoku 2025/2026"
    city = title.split(' w ')[1].split(' ')[0] if ' w ' in title else city_link.split('/')[-2]
    
    # Parse the page to get school data
    data = []
    
    # Find all school links
    school_links = []
    for a in soup.find_all('a', class_='text-bold utitle hide_on_small_screen'):
        href = a['href']
        if 'progi-punktowe' in href:
            school_links.append('https://www.otouczelnie.pl' + href)
    
    print(f"Found {len(school_links)} schools in {city}")
    # school_links = school_links[:3]
    
    # For each school, scrape the detailed page
    for school_url in school_links:
        school_data = scrape_school_data(school_url, city, year)
        data.extend(school_data)
        time.sleep(0.5)
    
    return data

def scrape_school_data(school_url, city, year):
    response = requests.get(f"{school_url}-{year}", verify=False)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract school name
    h1 = soup.find('h1')
    school_name = h1.text.strip() if h1 else 'Unknown'
    
    # Extract address
    address_div = soup.find('div', class_='address_row')
    address = ''
    if address_div:
        spans = address_div.find_all('span')
        if len(spans) > 1:
            address = spans[1].text.strip()
    
    # Find the table
    table = soup.find('table', class_='tabela-matura')
    data = []
    if table:
        rows = table.find_all('tr')
        for row in rows[1:]:  # skip header
            tds = row.find_all('td')
            if len(tds) == 2:
                course = tds[0].text.strip()
                threshold = tds[1].text.strip()
                data.append({
                    'year': year.replace('-', '/'),
                    'city': city,
                    'school': school_name,
                    'address': address,
                    'course': course,
                    'threshold': threshold
                })
    
    print(f"Scraped {len(data)} courses for {school_name} in {year}")
    
    return data

# Main datasets
all_data = []

for year in years:
    city_links = get_cities(year)
    for link in city_links:
        city_data = scrape_city_data(link, year)
        all_data.extend(city_data)
        time.sleep(0.5)  # Be polite

# Save to CSV
df = pd.DataFrame(all_data)
df.to_csv('school_thresholds_otouczelnie_raw.csv', index=False)
