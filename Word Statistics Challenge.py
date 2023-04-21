import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

table_data = pd.read_excel('WSC Input.xlsx',dtype={"Description":str, "TariffNumber":str})

list_of_tariff_nums = list(table_data["TariffNumber"].values)

for i in range(len(list_of_tariff_nums)):
    if len(list_of_tariff_nums[i]) == 9 :
        list_of_tariff_nums[i] = "0" + list_of_tariff_nums[i]           #adding a 0 if tarrif number has 9 chars

#-----Making table with words and their total count
list_of_descriptions = table_data["Description"].values

vectorizer = CountVectorizer()

cv_fit = vectorizer.fit_transform(list_of_descriptions)                             #vektorizacija svih reci

word_list = vectorizer.get_feature_names_out()                                # izdvajanje jedinstvenih reci
dict_of_words = dict(zip(word_list, np.asarray(cv_fit.sum(axis=0))[0]))       # pravljenje recnika sa zip funkcijom gde se prosledjuje rec i ukupan broj pojavljivanja te reci

dict_for_table = {"Word": list(dict_of_words.keys()),"Total Count": list(dict_of_words.values())}
ID_column = []
for index, word in enumerate(list(dict_of_words.keys()),1):
    ID_column.append(index)
table_of_words = pd.DataFrame.from_dict(data=dict_for_table)      #pravljenje tabele sa recima i njihovim pojavljivanjem
table_of_words.insert(0,"ID",ID_column)
print(table_of_words)
# table_of_words.to_csv("table_of_words.csv",index=False)


