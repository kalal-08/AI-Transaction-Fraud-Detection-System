import zipfile, os
import pandas as pd

zip_path = 'creditcard.csv.zip'
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        print('Zip contains:', z.namelist())
        if 'creditcard.csv' in z.namelist():
            z.extract('creditcard.csv')
            print('Extracted creditcard.csv')
        else:
            z.extractall()
            print('Extracted all files')
else:
    print('No zip file found')

if os.path.exists('creditcard.csv'):
    df = pd.read_csv('creditcard.csv')
    print('CSV loaded. Columns:', list(df.columns))
    print('Shape:', df.shape)
    print('Dtypes sample:')
    print(df.dtypes.head(40).to_dict())
    print('First row:')
    print(df.head(1).to_dict())
else:
    print('creditcard.csv not present')
