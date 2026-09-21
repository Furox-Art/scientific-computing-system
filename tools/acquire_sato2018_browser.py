#!/usr/bin/env python3
import hashlib, json, os, time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

OUT=Path("sato2018_browser_acquisition"); OUT.mkdir(exist_ok=True)
download_dir=str(OUT.resolve())
landing="https://datadryad.org/dataset/doi:10.5061/dryad.6675p"
legacy="https://datadryad.org/downloads/file_stream/61164"

opts=Options()
opts.add_argument("--headless=new")
opts.add_argument("--no-sandbox")
opts.add_argument("--disable-dev-shm-usage")
opts.add_argument("--window-size=1280,1000")
opts.add_argument("--disable-blink-features=AutomationControlled")
opts.add_experimental_option("prefs",{
  "download.default_directory":download_dir,
  "download.prompt_for_download":False,
  "download.directory_upgrade":True,
  "safebrowsing.enabled":True
})
driver=webdriver.Chrome(options=opts)
log={}
try:
    driver.execute_cdp_cmd("Page.setDownloadBehavior",{"behavior":"allow","downloadPath":download_dir})
    driver.get(landing)
    WebDriverWait(driver,30).until(lambda d:"Data from:" in d.page_source or "data_able-bodied.csv" in d.page_source)
    log["landing_title"]=driver.title
    log["landing_url"]=driver.current_url
    # Use the exact public href in the same browser tab. The page normally
    # exposes it with target="_blank"; using the same tab lets us observe the
    # JavaScript validation page and subsequent download.
    els=driver.find_elements(By.CSS_SELECTOR,"a.js-individual-dl[href*='file_stream/61164']")
    href=els[0].get_attribute("href") if els else legacy
    log["resolved_href"]=href
    driver.set_page_load_timeout(30)
    try:
        driver.get(href)
    except Exception as e:
        # A download can abort/replace navigation; retain the browser and
        # continue polling the download directory.
        log["navigation_exception"]=repr(e)
    # Let JS validation run and watch for a download.
    deadline=time.time()+90
    challenge_seen=False
    while time.time()<deadline:
        files=[p for p in OUT.iterdir() if p.is_file() and not p.name.endswith((".crdownload",".tmp"))]
        csvs=[p for p in files if p.suffix.lower()==".csv" or "able-bodied" in p.name.lower()]
        if csvs:
            p=max(csvs,key=lambda x:x.stat().st_mtime)
            b=p.read_bytes()
            log.update({"success":True,"path":p.name,"bytes":len(b),"sha256":hashlib.sha256(b).hexdigest(),"head":b[:500].decode("utf-8","replace")})
            break
        src=driver.page_source
        if "Validating..." in src or "validating" in src.lower(): challenge_seen=True
        time.sleep(1)
    else:
        log.update({"success":False,"challenge_seen":challenge_seen,"final_title":driver.title,"final_url":driver.current_url,"page_prefix":driver.page_source[:1500]})
finally:
    driver.quit()

(OUT/"probe.json").write_text(json.dumps(log,indent=2))
print(json.dumps(log,indent=2))
raise SystemExit(0 if log.get("success") else 2)
