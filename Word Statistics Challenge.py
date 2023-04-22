import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def calc_maxcount_num_of_words(list_of_tariff_nums: list, cut_tarrif: int, array_of_word_counts, insertmax: int, insertcount: int, type_of_text: str, table_of_words):
    """
    Calculates highest number of word occurrences and number of unique text types where word occurs and puts them in a table as columns.
    :param list_of_tariff_nums: List of all tariff numbers
    :param cut_tarrif: At what position to cut the tariff number
    :param array_of_word_counts: Array that contains word counts
    :param insertmax: Index where to insert the highest number of word occurrences
    :param insertcount: Index where to insert number of unique text types where word occurs
    :param type_of_text:
    :param table_of_words: Table to insert the columns
    :return: None
    """
    list_of_tar_num_fields = []
    for tariff in list_of_tariff_nums:
        chapter = tariff[:cut_tarrif]                                           # izdvajanje dela tariff_number-a
        list_of_tar_num_fields.append(chapter)                                  # cuvanje tog dela u listi

    temp_arr_of_word_counts = np.copy(array_of_word_counts)                     # pravljenje niza koji ce se obradjivati
    dict_of_indexes = {}
    indexes_to_delete = []
    for index, chapter in enumerate(list_of_tar_num_fields):
        if chapter not in dict_of_indexes:
            dict_of_indexes[chapter] = index                                # ako ne postoji takav isecak tariff_number-a onda se dodaje u dict
        else:
            temp_arr_of_word_counts[dict_of_indexes[chapter]] = np.add(      # ako postoji onda se vrste sabiraju u jednu vrstu
                temp_arr_of_word_counts[dict_of_indexes[chapter]], array_of_word_counts[index])
            indexes_to_delete.append(index)                                 # cuvanje indeksa koji treba da se obrisu
    indexes_to_delete.sort(reverse=True)                                    # sortiranje indeksa od najveceg ka najmanjem zbog brisanja
    for index in indexes_to_delete:
        temp_arr_of_word_counts = np.delete(temp_arr_of_word_counts, index, axis=0)     #brisanje reda po indeksu

    single_MaxCount = np.amax(temp_arr_of_word_counts, axis=0)                 # dobijanje najvece vrednosti u koloni
    table_of_words.insert(insertmax, "Single " + type_of_text + " Max-Count", single_MaxCount)      # ubacivanje kolone sa najvecim brojem pojavljivanja

    count_of_words = np.count_nonzero(temp_arr_of_word_counts, axis=0)          # dobijanje broja pojavljivanja reci
    table_of_words.insert(insertcount, "Unique " + type_of_text + " Count", count_of_words)         # ubacivanje kolone sa brojem pojavljivanja reci
    # print(table_of_words)


table_data = pd.read_excel('WSC Input.xlsx', dtype={"Description": str, "TariffNumber": str})      # ucitavanje excel dokumenta u memoriju

list_of_tariff_nums = list(table_data["TariffNumber"].values)           # uzimanje svih vrednosti Tariff_number-a

for i in range(len(list_of_tariff_nums)):
    if len(list_of_tariff_nums[i]) == 9:
        list_of_tariff_nums[i] = "0" + list_of_tariff_nums[i]           # dodavanje "0" na pocetak tariff_number ako ima 9 karatketa


list_of_descriptions = table_data["Description"].values                 # uzimanje svih opisa i smestanje u promenljivu

vectorizer = CountVectorizer()

cv_fit = vectorizer.fit_transform(list_of_descriptions)                             # vektorizacija svih reci u tf-idf vektore
arr_of_word_counts = cv_fit.toarray()
word_list = vectorizer.get_feature_names_out()                           # izdvajanje jedinstvenih reci

dict_of_words = dict(zip(word_list, np.asarray(cv_fit.sum(axis=0))[0]))       # pravljenje recnika sa zip funkcijom gde se prosledjuje rec i ukupan broj pojavljivanja te reci

dict_for_table = {"Word": list(dict_of_words.keys()), "Total Count": list(dict_of_words.values())}          # pravljenje dict za pravljenje tabele
ID_column = []
for index, word in enumerate(word_list, 1):
    ID_column.append(index)                                             # pravljenje liste ID-ova
table_of_words = pd.DataFrame.from_dict(data=dict_for_table)            # pravljenje tabele sa recima i njihovim pojavljivanjem
table_of_words.insert(0, "ID", ID_column)                                 # dodavanje kolone ID sa vrednostima

#----Chapters-----
calc_maxcount_num_of_words(list_of_tariff_nums, 2, arr_of_word_counts, 3, 4, "Chapter", table_of_words)

#----Headings-----
calc_maxcount_num_of_words(list_of_tariff_nums, 4, arr_of_word_counts, 5, 6, "Heading", table_of_words)

#----Subheadings-------
calc_maxcount_num_of_words(list_of_tariff_nums, 6, arr_of_word_counts, 7, 8, "Subheading", table_of_words)

#----Duty rates-------
calc_maxcount_num_of_words(list_of_tariff_nums, 8, arr_of_word_counts, 9, 10, "Duty rate", table_of_words)

#----Tariffs-------
calc_maxcount_num_of_words(list_of_tariff_nums, 10, arr_of_word_counts, 11, 12, "Tariff", table_of_words)

table_of_words.to_csv("result_table.csv", index=False)                   # cuvanje tabele u "result_table.csv"
