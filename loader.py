from datetime import datetime
from urllib.request import Request, urlopen
import configparser

import pandas as pd


CHANGE_INDICATOR_FILENAME = "./data/change_indicator.txt"


class DataLoader:

    def __init__(self, config_file_path):
        self.config_file_path = config_file_path
        # load config file into self.config variable
        self.config = configparser.ConfigParser()
        self.config.read(self.config_file_path)

    def load_source(self, key_name):
        assert key_name in self.config, f"source {key_name} is not registered"
        config_section = self.config[key_name]
        url = config_section["url"]
        print(f"Reading {key_name} database from url={url}")
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        rsp = urlopen(req)
        # only to keep track of the ContentLength value over time
        content_length = dict(rsp.getheaders())['Content-Length']
        msg = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}; Source={key_name}; Content-Length={content_length}"
        with open(CHANGE_INDICATOR_FILENAME, "a+") as f:
            f.write(f"{msg}\n")

        if config_section["LastContentLength"] != content_length:
            print("Changes detected. Loading data into memory...")
            df = pd.read_csv(rsp)
            # save new ContentLength value into the config file
            config_section["LastContentLength"] = content_length
            with open(self.config_file_path, 'w') as config_file:
                self.config.write(config_file)
            return df
        else:
            return None


def main():
    loader = DataLoader("./data/config.ini")
    data_frame = loader.load_source("vaccinations")
    if data_frame is None:
        print("No changes detected")
    else:
        print("Data loaded into memory successfully. Use data_frame variable to access them")


if __name__ == "__main__":
    main()
