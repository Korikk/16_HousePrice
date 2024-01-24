import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from helpers.data_prep import *
from helpers.eda import *

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

train = pd.read_csv("datasets/house_price/train.csv")
test = pd.read_csv("datasets/house_price/test.csv")
df = train.append(test).reset_index(drop=True)

# YOL HARITASI
# EDA
# check df
# Degisken analizleri(sayısal, categoric, num but cat,cat but car) (grab col names)
# target analizler,korelasyonlar

# drop - index icin
# birleştirme icin ikinci yöntem:
# df1 = pd.concat([train, test], ignore_index=True)
df.head()
train.shape
test.shape
df.shape


###############################
# EDA
###############################

def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)


# lowqualfinSF --> quantile 0.95--.99 -süpheli
# alley --> cakıl, asfalt ya da yok mu?
# EnclosedPorch --> quantile 0dan birden buyuk olma durumu
# MiscVal --> quantile 0dan birden buyuk olma durumu
# miscval aykırı olarak ayrı değerlnedirilebilir
# MasVnrType


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    grab_col_names for given dataframe

    :param dataframe:
    :param cat_th:
    :param car_th:
    :return:
    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    # cat_cols + num_cols + cat_but_car = değişken sayısı.
    # num_but_cat cat_cols'un içerisinde zaten.
    # dolayısıyla tüm şu 3 liste ile tüm değişkenler seçilmiş olacaktır: cat_cols + num_cols + cat_but_car
    # num_but_cat sadece raporlama için verilmiştir.

    return cat_cols, cat_but_car, num_cols, num_but_cat


cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)

for col in cat_but_car:
    cat_summary(df, col)

for col in num_but_cat:
    cat_summary(df, col)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()


for col in num_cols:
    num_summary(df, col)


# YORUMLAR
# LotShape farklı bölge aynı tarz ev farklı fiyat

def target_summary_with_cat(dataframe, target, categorical_cols):
    for categorical_col in categorical_cols:
        print(categorical_col)
        print("*************************")
        print(pd.DataFrame({"RATIO": 100 * dataframe[categorical_col].value_counts() / dataframe.shape[0],
                            "TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


target_summary_with_cat(df, "SalePrice", cat_cols)


def target_summary_with_num(dataframe, target, numerical_cols):
    for numerical_col in numerical_cols:
        print(
            dataframe.groupby(target).agg({"RATIO": 100 * dataframe[numerical_col].value_counts() / dataframe.shape[0],
                                           "TARGET_MEAN": dataframe.groupby(numerical_col)[target].mean()}),
            end="\n\n\n")


# def target_summary_with_num(dataframe, target, numerical_col):
#    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

target_summary_with_cat(df, "SalePrice", cat_cols)


def find_correlation(dataframe, numeric_cols, target, corr_limit=0.60):
    high_correlations = []
    low_correlations = []
    for col in numeric_cols:
        if col == target:
            pass
        else:
            correlation = dataframe[[col, target]].corr().loc[col, target]
            print(col, correlation)
            if abs(correlation) > corr_limit:
                high_correlations.append(col + ": " + str(correlation))
            else:
                low_correlations.append(col + ": " + str(correlation))
    return low_correlations, high_correlations


low_corrs, high_corrs = find_correlation(df, num_cols, "SalePrice")


# ['OverallQual: 0.7909816005838047',
# 'TotalBsmtSF: 0.6135805515591944',
# '1stFlrSF: 0.6058521846919166',
# 'GrLivArea: 0.7086244776126511',
# 'GarageArea: 0.6234314389183598']


# target_summary_with_cate göre:
# BIRLESTIRILEBILECEK SINIFLAR
# MSZoning --> RH - RM
# LotShape IR 123
# Utilities NoSeWa silinebilir

##################
# PREPROCESSİNG
###################


# RARE ANALYSIS


# def rare_analyser(dataframe, target, rare_perc):
#     rare_columns = [col for col in dataframe.columns if dataframe[col].dtypes == 'O'
#                     and (dataframe[col].value_counts() / len(dataframe) < rare_perc).any(axis=None)]
#
#     for col in rare_columns:
#         print(col, ":", len(dataframe[col].value_counts()))
#         print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
#                             "RATIO": dataframe[col].value_counts() / len(dataframe),
#                             "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")
#
#
# rare_analyser(df, "SalePrice", rare_perc=1)

for col in cat_cols:
    print(col, ":", len(df[col].value_counts()))
    print(pd.DataFrame({"COUNT": df[col].value_counts(),
                        "RATIO": df[col].value_counts() / len(df),
                        "TARGET_MEDIAN": df.groupby(col)["SalePrice"].median()}), end="\n\n\n")

# Sinif birlestirme
df["MSZoning"] = np.where(df.MSZoning.isin(["RH", "RM"]), "Rare", df["MSZoning"])
# Calistirilacak
df["LotShape"] = np.where(df.LotShape.isin(["IR1", "IR2", "IR3"]), "Rare", df["LotShape"])
df["LotConfig"] = np.where(df.LotConfig.isin(["FR2", "FR3"]), "FRRare", df["LotConfig"])

df["GarageQual"] = np.where(df.GarageQual.isin(["Fa", "Po"]), "Rare", df["GarageQual"])

df["BsmtFinType2"] = np.where(df.BsmtFinType2.isin(["GLQ", "ALQ"]), "RareExcellent", df["BsmtFinType2"])
df["BsmtFinType2"] = np.where(df.BsmtFinType2.isin(["BLQ", "LwQ", "Rec"]), "RareGood", df["BsmtFinType2"])

df["Heating"] = np.where(df.Heating.isin(["GasA", "GasW", "OthW"]), "RareGas", df["Heating"])
df["Heating"] = np.where(df.Heating.isin(["Floor", "Grav", "Wall"]), "Rare", df["Heating"])

df["HeatingQC"] = np.where(df.HeatingQC.isin(["TA", "Gd", "Fa"]), "RareGood", df["HeatingQC"])

df["GarageQual"] = np.where(df.GarageQual.isin(["TA", "Gd"]), "RareGood", df["GarageQual"])
df["GarageQual"] = np.where(df.GarageQual.isin(["Po", "Fa"]), "RarePoor", df["GarageQual"])

df["LandSlope"] = np.where(df.LandSlope.isin(["Mod", "Sev"]), "Rare", df["LandSlope"])

df["Condition1"] = np.where(df.Condition1.isin(["Artery", "Feedr"]), "Rare_art_feed", df["Condition1"])
df["Condition1"] = np.where(df.Condition1.isin(["PosA", "PosN"]), "RarePos", df["Condition1"])
df["Condition1"] = np.where(df.Condition1.isin(["RRNe", "RRNn"]), "RareRRN", df["Condition1"])

df["Fireplaces"] = np.where(df.Fireplaces.isin(["2", "3", "4"]), "Rare234", df["Fireplaces"])
df["GarageCars"] = np.where(df.GarageCars.isin(["4.000", "5.000"]), "Rare", df["GarageCars"])

df["BsmtCond"] = np.where(df.BsmtCond.isin(["Gd", "TA"]), "RareGdTA", df["BsmtCond"])
df["BsmtExposure"] = np.where(df.BsmtExposure.isin(["Av", "Mn"]), "RareAvMn", df["BsmtExposure"])
df["BsmtFinType1"] = np.where(df.BsmtFinType1.isin(["BLQ", "LwQ"]), "RareBLwQ", df["BsmtFinType1"])
df["BsmtFinType1"] = np.where(df.BsmtFinType1.isin(["ALQ", "Rec"]), "RareAlRec", df["BsmtFinType1"])

df["ExterCond"] = np.where(df.ExterCond.isin(["Ex", "Gd", "TA"]), "RareGood", df["ExterCond"])
df["ExterCond"] = np.where(df.ExterCond.isin(["Fa", "Po"]), "RarePoor", df["ExterCond"])
df["Foundation"] = np.where(df.Foundation.isin(["BrkTil", "Stone"]), "RareBrkSt", df["Foundation"])
df["Foundation"] = np.where(df.Foundation.isin(["CBlock", "Wood"]), "RareCBWood", df["Foundation"])

df["BldgType"] = np.where(df.BldgType.isin(["Duplex", "Twnhs"]), "RareDupTwnhs", df["BldgType"])

df["Exterior1st"] = np.where(df.Exterior1st.isin(["AsbShng", "AsphShn", "CBlock"]), "RareAsphShnCB", df["Exterior1st"])
df["Exterior1st"] = np.where(df.Exterior1st.isin(["HdBoard", "Stucco", "Wd Sdng", "WdShing"]), "RareHSwd",
                             df["Exterior1st"])
df["Exterior1st"] = np.where(df.Exterior1st.isin(["CemntBd", "Stone", "ImStucc"]), "RareHSwd", df["Exterior1st"])
df["Exterior1st"] = np.where(df.Exterior1st.isin(["Plywood", "BrkFace"]), "RarePlBrk", df["Exterior1st"])

df["Exterior2nd"] = np.where(df.Exterior2nd.isin(["AsbShng", "CBlock"]), "RarePlBrk", df["Exterior2nd"])
df["Exterior2nd"] = np.where(df.Exterior2nd.isin(["AsphShn", "Wd Sdng", "Wd Shng", "Stucco", "MetalSd", "Brk Cmn"]),
                             "Rareawwsmb", df["Exterior2nd"])
df["Exterior2nd"] = np.where(df.Exterior2nd.isin(["HdBoard", "BrkFace", "Plywood", "Stone"]), "RareBPS",
                             df["Exterior2nd"])
# vinlysd Imstucc bir arada düşünelebilir.

# fullbath 4 sınıf 3 sınıfa göre doldurulacak
df["FullBath"] = np.where(df.FullBath.isin(["3", "4"]), "Rare34", df["FullBath"])

df["GarageType"] = np.where(df.GarageType.isin(["2Types", "Basment"]), "Rare2TyBas", df["GarageType"])
df["GarageType"] = np.where(df.GarageType.isin(["CarPort", "Detchd"]), "Rare2CarDetch", df["GarageType"])

df["Fence"] = np.where(df.Fence.isin(["GdPrv", "MnPrv", "GdWo", "MnWw"]), "Rare",
                       df["Fence"])  # goodpriveciy ayrılabilir mi bakılsın # one hot

df["SaleType"] = np.where(df.SaleType.isin(["WD", "CWD"]), "RareWd", df["SaleType"])
df["SaleType"] = np.where(df.SaleType.isin(["ConLw", "ConLI", "ConLD"]), "RareConL", df["SaleType"])

df["SaleCondition"] = np.where(df.SaleCondition.isin(["Abnorml", "Family", "Alloca"]), "RareAbFaAll",
                               df["SaleCondition"])

df["OverallCond"] = np.where(df.OverallCond.isin(["6", "7", "8"]), "Rare678", df["OverallCond"])
df["OverallCond"] = np.where(df.OverallCond.isin(["3", "4"]), "Rare34", df["OverallCond"])

df["BsmtFullBath"] = np.where(df.BsmtFullBath.isin(["1.000 ", "2.000", "3.000"]), "Rare123", df["BsmtFullBath"])

"""
LandContour
MSZoning : 4
         COUNT  RATIO  TARGET_MEAN
C (all)     25  0.009    74528.000
FV         139  0.048   214014.062
RL        2265  0.776   191004.995
Rare       486  0.166   126675.226
"""
# Gozden Gecir: LandSlope, Condition1, BldgType, Exterior2nd, Fence, RoofStyle
drop_list = ["Street", "Alley", "LandContour", "Utilities", "LandSlope", "Condition2",
             "Heating", "PoolQC", "MiscFeature", "KitchenAbvGr", "BedroomAbvGr",
             "HalfBath", "RoofMatl", "FireplaceQu", "BsmtHalfBath",
             "RoofStyle", "ExterQual", "Electrical", "Functional", "FireplaceQu"]
# drop için gözden geçirilecek RoofStyle, ExterQual, Electrical, Functional, FireplaceQu

df.shape

# df["Alley"].value_counts()
# df.loc[df["Alley"].isnull(), "SalePrice"].mean()

[col for col in drop_list if col not in df.columns]


len([col for col in df.columns if col in drop_list])

df.drop(drop_list, axis=1, inplace=True)
df.shape

df.head()
len(drop_list)
## Gözlemledik
df.loc[df["GarageQual"] == "Ex", "SalePrice"]

## yapısal eksiklik için gözlemledik.
df.loc[df["BsmtFinType2"].isnull(), "BsmtFinType1"]

######################################
# LABEL ENCODING (2 sınıfı olan varsa) & ONE-HOT ENCODING (2 den fazla sınıf varsa)
######################################
df_ = df.copy()

df_.loc[df_["Fence"] == "Rare", "Fence"] = 1
df_.loc[df_["Fence"].isnull(), "Fence"] = 0

df["LotFrontage"]

oht = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
       "GarageType", "GarageFinish", "GarageQual", "GarageCond"]

na_cols = [col for col in df.columns if df[col].isnull().sum() > 0 and col not in oht]

cat_cols_new = [col for col in df_.columns if df_[col].dtypes == "O"]

df_.shape

df_ = one_hot_encoder(df_, cat_cols_new, drop_first=True)

check_df(df_)


# def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
#     original_columns = list(dataframe.columns)
#     dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
#     new_columns = [c for c in dataframe.columns if c not in original_columns]
#     return dataframe, new_columns


############################
# Eksik Gözlem
############################

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)

    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])

    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df_)

nan_cols = [col for col in df_.columns if df_[col].isnull().sum() > 0 and "SalePrice" not in col]

df_[nan_cols] = df_[nan_cols].apply(lambda x: x.fillna(x.median()), axis=0)


###################################
# Aykırı gözlem
###################################


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.05)
    quartile3 = dataframe[variable].quantile(0.95)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


# Verisetindeki ilgili değişkende herhangi bir eşik değerlere göre aykırı değer var mı? Bunu soran kod.
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)  # Thresholds hesaplama fonksiyonunu çağırıp.
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(
            axis=None):  # thresholds'un değerlerine göre up limitten büyük veya low limitten küçük herhangi bir değer varsa True yoksa False dönsün.
        return True
    else:
        return False


def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)  # eşik(threshold) değerleri hesapla
    if low_limit > 0:
        dataframe.loc[
            (dataframe[col_name] < low_limit), col_name] = low_limit  # alt sınırdan küçük olanları alt sınırla baskıla.
        dataframe.loc[
            (dataframe[col_name] > up_limit), col_name] = up_limit  # üst sınırdan büyük olanları üst sınırla baskıla.
    else:
        dataframe.loc[(dataframe[
                           col_name] > up_limit), col_name] = up_limit  # low limit 0 dan büyük değilse yani eksi ise sadece üst limite göre bir atama yap diyoruz.


out_col = [col for col in df_.columns if df_[col].nunique() > 2 and col not in "SalePrice"]

for i in out_col:
    print(i, check_outlier(df_, i))

for i in out_col:
    replace_with_thresholds(df_, i)

for i in out_col:
    print(i, check_outlier(df_, i))

############################################
# Robust
############################################

robust_col = [col for col in df_.columns if df_[col].nunique() > 2 and col not in ["Id", "SalePrice"]]

from sklearn.preprocessing import RobustScaler

for col in robust_col:
    transformer = RobustScaler().fit(df_[[col]])
    df_[col] = transformer.transform(df_[[col]])

check_df(df_)

#######################################
# MODEL: Random Forests
#######################################

train = df_[df_['SalePrice'].notnull()]
test = df_[df_['SalePrice'].isnull()].drop("SalePrice", axis=1)

X = train.drop(['SalePrice', "Id"], axis=1)
y = train["SalePrice"]
# y = np.log1p(train['SalePrice'])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=46)

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

lgb_model = LGBMRegressor().fit(X_train, y_train)
y_pred = lgb_model.predict(X_val)
np.sqrt(mean_squared_error(y_val, y_pred))

lgb_model = LGBMRegressor()

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1000],
               "max_depth": [3, 5, 8],
               "colsample_bytree": [1, 0.8, 0.5]}

lgbm_cv_model = GridSearchCV(lgb_model,
                             lgbm_params,
                             cv=10,
                             n_jobs=-1,
                             verbose=1).fit(X_train, y_train)

lgbm_cv_model.best_params_

#######################################
# Final Model
#######################################

lgbm_tuned = LGBMRegressor(**lgbm_cv_model.best_params_).fit(X_train, y_train)
y_pred = lgbm_tuned.predict(X_val)
np.sqrt(mean_squared_error(y_val, y_pred))

new_predict = pd.DataFrame()
new_predict["Id"] = test["Id"].astype(int)
y_pred_sub = lgbm_tuned.predict(test.drop("Id", axis=1))

new_predict['SalePrice'] = y_pred_sub


new_predict.to_csv('submission_lgthgbm1.csv', index=False)















def plot_importance(model, X, num=len(X)):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': X.columns})
    plt.figure(figsize=(10, 15))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('importances-01.png')
    plt.show()


plot_importance(lgb_model, X_train, num=50)
