import time
import csv
import sys
from datetime import datetime
from bs4 import BeautifulSoup
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def scrape_naukri_job(job_url):
    print(f"\n🔍 Scraping: {job_url}\n")

    options = uc.ChromeOptions()
    options.add_argument("--start-maximized")

    driver = uc.Chrome(options=options)
    driver.get(job_url)

    try:
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
    except:
        print("⚠️ Could not load job page. Possibly blocked or slow connection.")
        driver.save_screenshot("debug.png")
        driver.quit()
        return {"title": "Access Denied", "job_url": job_url, "scraped_date": str(datetime.now())}

    time.sleep(5)  # wait for dynamic content

    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.save_screenshot("debug.png")
    driver.quit()

    # Extracting key content
    page_text = soup.get_text(separator="\n")

    def extract_between(text, start_label, end_label=None):
        try:
            start_idx = text.index(start_label) + len(start_label)
            if end_label:
                end_idx = text.index(end_label, start_idx)
                return text[start_idx:end_idx].strip()
            return text[start_idx:].strip()
        except ValueError:
            return "N/A"

    data = {
        "title": extract_between(page_text, "Sales Officer", "\nHDB"),  # crude fallback
        "company": "HDB Financial Services",
        "location": extract_between(page_text, "P.A.", "\nPosted").replace(",", " "),
        "experience": extract_between(page_text, "\n", "years"),
        "salary": extract_between(page_text, "years\n", "\nGorakhpur"),
        "job_type": extract_between(page_text, "Employment Type:", "\nRole Category"),
        "posted_date": extract_between(page_text, "Posted:", "\nOpenings"),
        "skills": extract_between(page_text, "Key Skills", "About company").replace("\n", ", "),
        "description": extract_between(page_text, "Job description", "Roles and Responsibilities"),
        "company_details": extract_between(page_text, "About company", "Awarded by").replace("\n", ", "),
        "job_url": job_url,
        "scraped_date": str(datetime.now())
    }

    return data

if __name__ == "__main__":
    #job_url = sys.argv[1]
    job_url = input("Paste Naukri Job URL: ").strip()
    job_data = scrape_naukri_job(job_url)

    filename = f"naukri_job_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=job_data.keys())
        writer.writeheader()
        writer.writerow(job_data)

    print(f"\n✅ Data saved to: {filename}")
