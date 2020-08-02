import pandas as pd


class PreProcess:
    def __init__(self, file_name):
        self.file_name = file_name
        self.data = pd.read_csv(file_name)

        # remove empty lines & articles
        self.data = self.data[[i != '\xa0' for i in self.data.abstract]]
        self.data = self.data[[i != '\xa0' for i in self.data.main]]
        self.data = self.data[[i != '-->' for i in self.data.abstract]]
        self.data = self.data[[i != '>' for i in self.data.abstract]]
        self.data = self.data[[i != 'int-facebook' for i in self.data.abstract]]
        self.data = self.data[['-- No Title' not in str(i) for i in self.data['main']]]


        # remove duplicate abstracts and headlines
        self.data = self.data[[isinstance(i, str) for i in self.data['main']]]
        self.data = self.data[[isinstance(i, str) for i in self.data['abstract']]]

        # remove common headline tagline: World Briefing |
        to_remove = "World Briefing"
        self.data['main'] = [i.split(":")[-1] if to_remove in i else i for i in self.data['main']]

        self.data = self.data.drop_duplicates(subset=['main'])
        self.data = self.data.drop_duplicates(subset=['abstract'])


if __name__ == '__main__':
    pp = PreProcess(file_name='data/europe_section.csv')
    df = pp.data
    print(df.head())
