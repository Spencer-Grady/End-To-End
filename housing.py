import os
import tarfile
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True) 
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
load_data = fetch_housing_data()


housing = load_housing_data()
print(housing.head())
print(housing.info())

#find out how many ocean proximity attributes there are
print(housing["ocean_proximity"].value_counts())
#look at other attributes as well
print(housing.describe())

housing.hist(bins=50, figsize=(10,7.5))
plt.show()