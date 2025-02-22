import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium.webdriver.support.ui import WebDriverWait
class SeleniumPreprocessing:
    def __init__(self):
        self.CSV_PATH = "/mnt/d/PWR/Praca magisterska/Datasets/CLEARED_BASE_DATASET/dataset_mal_cleared_full_1.csv"
        self.CHUNK_SIZE = 5000
        self.max_workers = 64
        self.base_filename_results = "/mnt/d/PWR/Praca magisterska/Datasets/CLEARED_BASE_DATASET/cleared_web/url/dataset_mal_cleared_full"
        self.base_filename_errors = "/mnt/d/PWR/Praca magisterska/Datasets/CLEARED_BASE_DATASET/cleared_web/err/dataset_mal_cleared_full_errors"

    def get_final_url(self,url):
        try:
            response = requests.get(url, allow_redirects=True, timeout=10)
            return response.url
        except Exception as e:
            return None

    def create_driver(self):
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--blink-settings=imagesEnabled=false")
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        )
        options.page_load_strategy = "eager"
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(10)
        return driver
    def process_url(self,row):
        link_orig = row.original_url
        link_new = row.full_url
        link_type = row.type

        final_url = self.get_final_url(link_new)
        if final_url is None:
            driver = self.create_driver()
            try:
                driver.get(link_new)
                initial_url = driver.current_url
                try:
                    WebDriverWait(driver, 10).until(lambda d: d.current_url != initial_url)
                except Exception as e:
                    pass
                final_url = driver.current_url
            except Exception as e:
                final_url = None
            driver.quit()

        return link_orig, final_url, link_type
    def check_urls(self):
        chunk_index = 0
        for chunk_index, chunk in enumerate(pd.read_csv(self.CSV_PATH, chunksize=self.CHUNK_SIZE), start=1):
            if chunk_index < 8:
                continue
            print(f"\nThe block is being processed {chunk_index} (lines: {len(chunk)})")
            success_chunk_orig = []
            success_chunk_full = []
            success_chunk_type = []
            error_chunk_orig = []
            error_chunk_type = []
            futures_dict = {}
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for row in chunk.itertuples():
                    fut = executor.submit(self.process_url, row)
                    futures_dict[fut] = row
                progress_bar = tqdm(as_completed(futures_dict), total=len(futures_dict), desc=f"Chunk {chunk_index}")
                for future in progress_bar:
                    try:
                        link_orig, full_url, link_type = future.result(timeout=45)
                    except Exception as e:
                        row = futures_dict[future]
                        link_orig = row.original_url
                        link_type = row.type
                        full_url = None
                    if full_url is not None:
                        success_chunk_orig.append(link_orig)
                        success_chunk_full.append(full_url)
                        success_chunk_type.append(link_type)
                    else:
                        error_chunk_orig.append(link_orig)
                        error_chunk_type.append(link_type)
                    progress_bar.set_postfix({"errors": len(error_chunk_orig)})
            df_success = pd.DataFrame({
                "original_url": success_chunk_orig,
                "full_url": success_chunk_full,
                "type": success_chunk_type
            })
            df_error = pd.DataFrame({
                "errors": error_chunk_orig,
                "type_err": error_chunk_type
            })
            results_filename = f"{self.base_filename_results}_chunk_{chunk_index}.csv"
            errors_filename = f"{self.base_filename_errors}_chunk_{chunk_index}.csv"
            df_success.to_csv(results_filename, index=False)
            df_error.to_csv(errors_filename, index=False)
            print(f"The block {chunk_index} has been processed : {len(df_success)} successful, {len(df_error)} errors.")
            print(f"Saved as: \n  {results_filename}\n  {errors_filename}")

