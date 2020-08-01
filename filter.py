import ujson
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.rrule import rrule, MONTHLY
from tqdm import tqdm  # for progress bar

def get_raw_json(month, year, raw_dir='data/raw/'):
    file_name = str(year) + "_" + format(month, '02') + ".txt"
    with open(raw_dir + file_name) as json_file:
        data = ujson.load(json_file)

    df = pd.DataFrame(data["response"]["docs"])

    return (df)

def separate_dict_col(df, col_name):
    new_df = pd.concat([df.drop([col_name], axis=1), df[col_name].apply(pd.Series)], axis=1)
    return new_df


def filter_articles(start_date, end_date, filters, cols, keywords=[]):
    dates = [dt for dt in rrule(MONTHLY, dtstart=start_date, until=end_date)]
    months = [(dt.year, dt.month) for dt in dates]

    output = pd.DataFrame(columns=cols)
    for m in tqdm(months):

        df = get_raw_json(m[1], m[0])
        df = separate_dict_col(df, 'headline')
        df = separate_dict_col(df, 'byline')
        try:
            for k, v in filters.items():
                df = df[df[k] == v]
            if keywords:
                for i in df.index:
                    if df['keywords'][i]:
                        kw_ls = [kw['value'] for kw in df['keywords'][i]]
                        if len(list(set(kw_ls) & set(keywords))) == 0:
                            df = df.drop(index=i)
                    else:
                        df = df.drop(index=i)
            output = pd.concat([output, df[cols]], axis=0, ignore_index=True)
        except KeyError:
            pass
    return output


def filter_file(month, year, filters, cols, keywords=[]):
    df = get_raw_json(month, year)
    df = separate_dict_col(df, 'headline')
    df = separate_dict_col(df, 'byline')
    try:
        for k, v in filters.items():
            df = df[df[k] == v]
        if keywords:
            for i in df.index:
                if df['keywords'][i]:
                    kw_ls = [kw['value'] for kw in df['keywords'][i]]
                    if len(list(set(kw_ls) & set(keywords))) == 0:
                        df = df.drop(index=i)
                else:
                    df = df.drop(index=i)
    except KeyError:
        pass
    return df


if __name__ == "__main__":
    cols = ['abstract', 'main', 'person', 'pub_date', 'keywords']
    filters = {
        #'section_name': 'World',
        #'subsection_name': "Europe"
    }
    keywords = ['Brazil']

    start_date = datetime(1998, 1, 1)
    end_date = datetime(2020, 6, 1)

    df = filter_articles(start_date, end_date,
                         filters, cols, keywords=keywords)

    print(df)
    # save to file
    file_name = 'data/brazil.csv'
    df.to_csv(file_name, index=False)
