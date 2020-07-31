
import json, requests
from datetime import datetime
from dateutil.rrule import rrule, MONTHLY
import time
from tqdm import tqdm

API_KEY = "kaZZQe4LCfeEEJ222nnJttQYBZQA634s"

def save_archive_month(month, year, raw_dir='data/raw/'):
    # dowload data
    url = "https://api.nytimes.com/svc/archive/v1/" + \
          str(year) + "/" + str(month) + ".json?api-key=" + \
          API_KEY
    resp = requests.get(url)
    data = json.loads(resp.text)

    # save locally
    file_name = str(year) + "_" + format(month, '02') + ".txt"
    with open(raw_dir + file_name, 'w') as outfile:
        json.dump(data, outfile)


def save_history(start_date, end_date):
    # list of months
    dates = [dt for dt in rrule(MONTHLY, dtstart=start_date, until=end_date)]
    months = [(dt.year, dt.month) for dt in dates]

    # save history
    for m in tqdm(months):
        save_archive_month(m[1], m[0])
        time.sleep(6)  # waiting time for API

    print("done.")

if __name__ == "__main__":
    start = datetime(1975,1,1)
    end = datetime(2020,6,1)

    save_history(start, end)
    #save_archive_month(1, 1975)

