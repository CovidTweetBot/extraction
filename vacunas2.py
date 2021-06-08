from datetime import datetime
from urllib.request import Request, urlopen

# url to the dataset (CSV format) containing the nominal vaccination records
CSV_URL = "https://cloud.minsa.gob.pe/s/ZgXoXqK2KLjRLxD/download"

CHANGE_INDICATOR_FILENAME = "./data/change_indicator.txt"


def main():
    print(f"Reading database from link: {CSV_URL}")
    req = Request(CSV_URL, headers={'User-Agent': 'Mozilla/5.0'})

    rsp = urlopen(req)

    content_length = dict(rsp.getheaders())['Content-Length']

    msg = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}; Content-Length={content_length}"
    with open(CHANGE_INDICATOR_FILENAME, "a+") as f:
        f.write(f"{msg}\n")

    print(msg)


if __name__ == "__main__":
    main()
