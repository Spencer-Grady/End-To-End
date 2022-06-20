import os
import tarfile
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32 #cyclic redundancy check
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
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

#housing.hist(bins=50, figsize=(10,7.5))
#plt.show()

# create test set to test model after it is made
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    
    return data.iloc[train_indices], data.iloc[test_indices] #returns the index of the sets, therefor is the length of the sets

housing_with_id = housing.reset_index() #adds an index column
housing_with_id["id"] = housing["longitude"] *1000 +housing["latitude"] #combines long and lat to one id since they wont ever change
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

print(len(train_set))
print(len(test_set))

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2* 32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing["income_cat"] = pd.cut(housing["median_income"],
                        bins = [0., 1.5, 3.0, 4.5, 6., np.inf],
                        labels = [1, 2, 3, 4, 5])
#housing["income_cat"].hist()
#plt.show()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True) #restore data to original without income cat

# DISCOVER AND VISULAIZE DATA TO GAIN INSIGHT #
housing = strat_train_set.copy()

# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"]/100, label="population", figsize=(10,7), c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,) #alpha shows just where high concentrations are for pattern
# plt.legend()
#plt.show()

# LOOKING FOR CORRELATIONS #
corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False)) #shows how each att correlates to house value

# attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
# scatter_matrix(housing[attributes], figsize=(12,8))
# plt.show()

# best indicator of house value, is income, lets focus on that
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
plt.show()

# EXPERIMENTING WITH ATTRIBUTE COMBINATIONS #
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))