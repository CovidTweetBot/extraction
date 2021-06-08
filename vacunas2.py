from urllib.request import Request, urlopen

# Requests
csv_url = "https://cloud.minsa.gob.pe/s/ZgXoXqK2KLjRLxD/download"


def main():
    print('Reading database from link')
    req = Request(csv_url, headers={'User-Agent': 'Mozilla/5.0'})

    rsp = urlopen(req)

    content_length = dict(rsp.getheaders())['Content-Length']
    print(f"Content-Length={content_length}")


if __name__ == "__main__":
    main()
