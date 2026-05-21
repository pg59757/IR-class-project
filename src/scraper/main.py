import json
import src.scraper.scraper as scraper

def main():

    repo_url = f"https://repositorium.uminho.pt/handle"
    collection = "1822/21295" #CCTC - Artigos em atas de conferências internacionais (texto completo)
    base_url = f"{repo_url}/{collection}"

    # Create an instance of the Scraper class
    # The scraper will automatically detect Chrome in default locations
    scraper_instance = scraper.UMinhoDSpace8Scraper(base_url, max_items=20) #20 documentos PDF
    final_results = scraper_instance.scrape()

    print(f"Scraping completed. Total papers scraped: {len(final_results)}")

    # Save results to a JSON file
    with open('scraper_results.json', 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)

    print(f"Done! {len(final_results)} items saved.")

if __name__ == "__main__":
    main()