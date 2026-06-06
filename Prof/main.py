import json
import scraper

def main():
    # Example collection:  https://repositorium.uminho.pt/collections/690f7814-a67b-4f27-8fff-6b33581d1a91/search
    # https://repositorium.uminho.pt/handle/1822/21293
    repo_url = f"https://repositorium.uminho.pt/handle/"
    collection = "1822/21293"
    base_url = f"{repo_url}/{collection}"

    # Create an instance of the Scraper class
    # The scraper will automatically detect Chrome in default locations
    scraper_instance = scraper.UMinhoDSpace8Scraper(base_url, max_items=15)
    final_results = scraper_instance.scrape()

    print(f"Scraping completed. Total papers scraped: {len(final_results)}")

    # Save results to a JSON file
    with open('scraper_results.json', 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)

    print(f"Done! {len(final_results)} items saved.")

if __name__ == "__main__":
    main()
