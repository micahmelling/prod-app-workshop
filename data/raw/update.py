import warnings

import numpy as np
import pandas as pd


warnings.filterwarnings('ignore')
np.random.seed(42)


def recreate_target(df):
    df = df.drop('GradeClass', axis=1)
    df['gpa_b_and_up'] = np.where(
        df['GPA'] >= 3.0,
        1,
        0
    )
    return df


def add_useless_column(df):
    df['counselor'] = np.random.choice(['a', 'b', 'c', 'd'], size=len(df))
    return df


def add_outlier_rows(df):
    df_for_outliers = df.tail(3)
    df_for_outliers['Age'] = 0
    df = pd.concat([df, df_for_outliers], axis=0)
    df = df.reset_index(drop=True)
    return df


# TODO: fix music and volunteering - remove '.0' at the end
def convert_cols_to_str(df, cols):
    for col in cols:
        df[col] = np.where(
            df[col].isnull(),
            df[col],
            'cat_' + df[col].astype(str)
        )
        df[col] = df[col].str.replace('.0', '')
    return df


def add_missing_data_rows(df):
    df['Volunteering'][10] = np.nan
    df['Volunteering'][21] = np.nan
    df['Volunteering'][1_002] = np.nan
    df['Music'][198] = np.nan
    df['Music'][231] = np.nan
    df['Music'][2_402] = np.nan
    return df


if __name__ == "__main__":
    df = pd.read_csv('data/raw/Student_performance_data _.csv')
    df = recreate_target(df)
    df = add_useless_column(df)
    df = df.drop('Ethnicity', axis=1)
    df = add_outlier_rows(df)
    df = add_missing_data_rows(df)
    df = convert_cols_to_str(df, ['Gender', 'ParentalEducation', 'Tutoring', 'ParentalSupport',
                                  'Extracurricular', 'Sports', 'Music',	'Volunteering'])
    df.to_csv('data/raw/student_performance.csv', index=False)
