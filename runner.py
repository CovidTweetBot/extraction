from loader import DataLoader
from processors import apply_processing
from visualizations import generate_visualizations


def main():
    loader = DataLoader("./data/config.ini")
    # loading vaccinations
    df_vaccinations = loader.load_source("vaccinations")
    if df_vaccinations is None:
        print("No vaccinations changes detected")
    else:
        print("vaccinations data loaded into memory successfully. Use df_vaccinations variable to access them")

    # loading sinadef
    df_sinadef = loader.load_source("sinadef", sep=';', index_col=1, encoding='latin-1')
    if df_sinadef is None:
        print("No sinadef changes detected")
    else:
        print("sinadef data loaded into memory successfully. Use df_sinadef variable to access them")
        apply_processing(df_sinadef)
        generate_visualizations()


if __name__ == "__main__":
    main()
