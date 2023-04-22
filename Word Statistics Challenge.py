import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def calc_maxCount_num_of_words(list_of_tariff_nums :list,end:int,array_of_word_counts,insertmax:int,insertcount:int,type_of_text:str,table_of_words):
    list_of_tar_num_fields = []
    for tariff in list_of_tariff_nums:
        chapter = tariff[:end]
        list_of_tar_num_fields.append(chapter)

    temp_arr_of_word_counts = np.copy(array_of_word_counts)
    dict_of_indexes = {}
    indexes_to_delete = []
    for index, chapter in enumerate(list_of_tar_num_fields):
        if chapter not in dict_of_indexes:
            dict_of_indexes[chapter] = index
        else:
            temp_arr_of_word_counts[dict_of_indexes[chapter]] = np.add(
                temp_arr_of_word_counts[dict_of_indexes[chapter]], array_of_word_counts[index])
            indexes_to_delete.append(index)
    indexes_to_delete.sort(reverse=True)
    for index in indexes_to_delete:
        temp_arr_of_word_counts = np.delete(temp_arr_of_word_counts, index, axis=0)

    single_MaxCount = []
    single_MaxCount = np.amax(temp_arr_of_word_counts, axis=0)
    table_of_words.insert(insertmax, "Single " +type_of_text+ " Max-Count", single_MaxCount)

    count_of_words = []
    count_of_words = np.count_nonzero(temp_arr_of_word_counts, axis=0)
    table_of_words.insert(insertcount, "Unique " +type_of_text+ " Count", count_of_words)
    # print(table_of_words)



table_data = pd.read_excel('WSC Input.xlsx',dtype={"Description":str, "TariffNumber":str})      # ucitavanje excel dokumenta u memoriju

list_of_tariff_nums = list(table_data["TariffNumber"].values)

for i in range(len(list_of_tariff_nums)):
    if len(list_of_tariff_nums[i]) == 9 :
        list_of_tariff_nums[i] = "0" + list_of_tariff_nums[i]           # dodavanje 0 na pocetak tariff_number ako ima 9 karatketa


list_of_descriptions = table_data["Description"].values                 # uzimanje svih opisa i smestanje u promenljivu

vectorizer = CountVectorizer()

cv_fit = vectorizer.fit_transform(list_of_descriptions)                             #vektorizacija svih reci
arr_of_word_counts= cv_fit.toarray()
word_list = vectorizer.get_feature_names_out()                           # izdvajanje jedinstvenih reci

dict_of_words = dict(zip(word_list, np.asarray(cv_fit.sum(axis=0))[0]))       # pravljenje recnika sa zip funkcijom gde se prosledjuje rec i ukupan broj pojavljivanja te reci

dict_for_table = {"Word": list(dict_of_words.keys()),"Total Count": list(dict_of_words.values())}          #pravljenje dict za pravljenje tabele
ID_column = []
for index, word in enumerate(word_list,1):
    ID_column.append(index)
table_of_words = pd.DataFrame.from_dict(data=dict_for_table)            #pravljenje tabele sa recima i njihovim pojavljivanjem
table_of_words.insert(0,"ID",ID_column)                                 # dodavanje kolone ID sa vrednostima

#----Chapters-----
calc_maxCount_num_of_words(list_of_tariff_nums,2,arr_of_word_counts,3,4,"Chapter",table_of_words)

#----Headings-----
calc_maxCount_num_of_words(list_of_tariff_nums,4,arr_of_word_counts,5,6,"Heading",table_of_words)

#----Subheadings-------
calc_maxCount_num_of_words(list_of_tariff_nums,6,arr_of_word_counts,7,8,"Subheading",table_of_words)

#----Duty rates-------
calc_maxCount_num_of_words(list_of_tariff_nums,8,arr_of_word_counts,9,10,"Duty rate",table_of_words)

#----Tariffs-------
calc_maxCount_num_of_words(list_of_tariff_nums,10,arr_of_word_counts,11,12,"Tariff",table_of_words)

table_of_words.to_csv("result_table.csv",index=False)                   #saving table to result_table.csv