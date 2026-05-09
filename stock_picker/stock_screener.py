import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from io import StringIO
import time
import logging
import os

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_tickers(url=None, delay=1):
    """
    Scrape FinViz and return list of tickers directly.
    
    Args:
        url: FinViz screener URL (uses default if None)
        delay: Delay between page requests
        
    Returns:
        List of ticker symbols
    """
    if url is None:
        url = "https://finviz.com/screener.ashx?v=121&f=cap_smallover,sh_relvol_o2,ta_perf_d5o&ft=4&o=-marketcap"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'lxml')
        num_pages = get_total_pages(soup)
        logging.info(f"Found {num_pages} pages to scrape")

        all_data = []
        ticker_number = 1
        
        for page in range(num_pages):
            try:
                logging.info(f"Scraping page {page + 1}/{num_pages}")
                page_url = url + f"&r={ticker_number}"
                response = requests.get(page_url, headers=headers)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, 'lxml')
                table = soup.find('table', class_='styled-table-new is-rounded is-tabular-nums w-full screener_table')

                if table is None:
                    logging.warning(f"No table found on page {page + 1}")
                    continue

                table_html = StringIO(str(table))
                pd_data = pd.read_html(table_html)
                all_data.append(pd_data[0])
                ticker_number += 20
                time.sleep(delay)

            except Exception as e:
                logging.error(f"Error processing page {page + 1}: {str(e)}")
                continue

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            tickers = list(dict.fromkeys(combined_df['Ticker'].tolist()))
            logging.info(f"Found {len(tickers)} tickers")
            return tickers
        else:
            logging.warning("No tickers found")
            return []

    except requests.exceptions.RequestException as e:
        logging.error(f"Network error: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise


def append_to_csv(df, csv_file):
    """Append dataframe to CSV file with timestamp"""
    try:
        # add timestamp column
        df['Scraped_At'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # check if file exists and is not empty
        file_exists = os.path.isfile(csv_file) and os.path.getsize(csv_file) > 0
        
        # append to CSV, write header only if file doesn't exist
        df.to_csv(csv_file, mode='w', header=True, index=False)
        
    except Exception as e:
        logging.error(f"Error writing to CSV: {str(e)}")
        raise

def get_total_pages(soup):
    """Get the total number of pages from the pagination"""
    try:
        pagination_tags = soup.find(class_="body-table screener_pagination").find_all('a')
        total_pages = (len(pagination_tags) - 1)  # Excluding the arrow
        return 1 if total_pages == 0 else total_pages
    except AttributeError:
        logging.warning("No pagination found, assuming single page")
        return 1
    except Exception as e:
        logging.error(f"Error getting total pages: {str(e)}")
        return 1

def get_webpage(url, csv_file, delay=1):
    """Scrape data from FinViz and save to CSV"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        # initial connection to get total pages
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # raise an exception for bad status codes

        soup = BeautifulSoup(response.content, 'lxml')
        num_pages = get_total_pages(soup)
        logging.info(f"Found {num_pages} pages to scrape")

        # collect all data before writing
        all_data = []
        
        ticker_number = 1
        for page in range(num_pages):
            try:
                logging.info(f"Scraping page {page + 1}/{num_pages}")

                # visit each page and convert into pandas data
                page_url = url + f"&r={ticker_number}"
                response = requests.get(page_url, headers=headers)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, 'lxml')
                table = soup.find('table', class_='styled-table-new is-rounded is-tabular-nums w-full screener_table')

                if table is None:
                    logging.warning(f"No table found on page {page + 1}")
                    continue

                table_html = StringIO(str(table))
                pd_data = pd.read_html(table_html)
                
                # collect data from this page
                all_data.append(pd_data[0])
                ticker_number += 20

                # add delay between requests to avoid rate limiting
                time.sleep(delay)

            except Exception as e:
                logging.error(f"Error processing page {page + 1}: {str(e)}")
                continue

        # combine all data and write once
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            append_to_csv(combined_df, csv_file)
            logging.info(f"Wrote {len(combined_df)} records to {csv_file}")
        else:
            logging.warning("No data collected to write")

    except requests.exceptions.RequestException as e:
        logging.error(f"Network error: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise

def main():
    """Main function to run the scraper"""
    try:
        # configuration
        csv_file = "saved_data/FinVizData.csv"
        url = "https://finviz.com/screener.ashx?v=121&f=cap_smallover,sh_relvol_o2,ta_perf_d5o&ft=4&o=-marketcap"

        logging.info(f"Starting scraper for {url}")
        logging.info(f"Data will be saved to {csv_file}")

        # run the scraper
        get_webpage(url, csv_file)

        logging.info("Scraping completed successfully")

    except Exception as e:
        logging.error(f"Script failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()