#!/usr/bin/env python3
import json, time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
OUT=Path("sato2018_dom_probe"); OUT.mkdir(exist_ok=True)
opts=Options(); opts.add_argument("--headless=new"); opts.add_argument("--no-sandbox"); opts.add_argument("--disable-dev-shm-usage")
d=webdriver.Chrome(options=opts)
log={}
try:
    d.get("https://datadryad.org/dataset/doi:10.5061/dryad.6675p")
    time.sleep(3)
    items=[]
    for el in d.find_elements(By.XPATH,"//*[contains(text(),'data_able-bodied.csv') or contains(@href,'file_stream') or contains(@href,'61164')]"):
        items.append({
          "tag":el.tag_name,
          "text":el.text,
          "href":el.get_attribute("href"),
          "onclick":el.get_attribute("onclick"),
          "outerHTML":el.get_attribute("outerHTML")
        })
    log["items"]=items
    log["current_url"]=d.current_url
    log["title"]=d.title
    # Also inspect every anchor around the Data files section.
    anchors=[]
    for el in d.find_elements(By.TAG_NAME,"a"):
        href=el.get_attribute("href") or ""
        txt=(el.text or "").strip()
        if "61164" in href or "file_stream" in href or "able-bodied" in txt:
            anchors.append({"text":txt,"href":href,"outerHTML":el.get_attribute("outerHTML")})
    log["anchors"]=anchors
finally:
    d.quit()
(OUT/"dom.json").write_text(json.dumps(log,indent=2))
print(json.dumps(log,indent=2))
