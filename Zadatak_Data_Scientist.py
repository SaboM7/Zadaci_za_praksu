import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests

table1 = pd.read_csv('descriptive_attributes.csv')
table2 = pd.read_csv('numeric_atributes.csv', dtype={"startYear": int, "averageRating": float})

#-----1-----
merged_table = table1.merge(table2, on="movieID", how="inner")     # spajanje tabelo po koloni movieID koristeci inner join

# merged_table.to_csv("merged_table.csv",index= False)                      # cuvanje spojene tabele

#-----2-----
num_of_no_isGood = merged_table['isGood'].isnull().sum()                  # prebrojavanje nepopunjenih polja u isGood koloni
print(num_of_no_isGood)

#-----3-----
merged_table['averageRating'] = merged_table['averageRating'].fillna(0.0)      # popunjavanje nedostajucih vrednosti u averageRating koloni

#-----4-----
temp_table = merged_table[(merged_table['startYear'] > 2000) & (merged_table['averageRating'] > 3.0)]   # pravljenje pomocne tabele sa odgovarajucim vrednostima
print(temp_table)
num_of_temp_table_rows = len(temp_table.index)                                    # prebrojavanje redova u tabeli
print(num_of_temp_table_rows)

#-----5-----
merged_table.hist(column="averageRating", bins=10)                               # pravljenje histograma sa 10 polja
plt.xlabel('Prosecna ocena')
plt.ylabel('Broj Filmova')
plt.title('Ocene filmova')

plt.show()

# -----6-----
def get_avg_rating_videotype(table):
    """
    Calculates and prints average ratings for movie type
    :param table: Table to perform the calculation of average rating
    :return: None
    """
    unique_types = table['titleType'].unique()                              # nalazenje svih kategorija videa
    for unique_type in unique_types:
        table_for_avgRating = table[(table['titleType'] == unique_type)]    # izdvajanje redova odredjene kategorije
        avg_rating = table_for_avgRating.averageRating.mean()               # racunanje prosecne ocene kategorije
        avg_rating = round(avg_rating, 2)                                   # zaokruzivanje prosecne ocene kategorije
        print(f"Average rating for {unique_type} is : {avg_rating}")


get_avg_rating_videotype(merged_table)

# -----7-----
list_of_titles = list(merged_table["originalTitle"].values)               # pravljenje liste svih orginalnih naziva videa
list_of_titles.insert(0, "The French Connection")                          # ubacivanje naziva sa kojim poredimo na prvo mesto

tfidf_vectorizer = TfidfVectorizer()                                      # inicijalizacija Tfidf objekta za pravljenje vektora

tfidf_matrix = tfidf_vectorizer.fit_transform(list_of_titles)             # pravljenje matrice Tfidf vektora


def calculating_similarities(list_of_titles: list, tfidf_matrix):
    """
    Calculates similarity of the first title on the list with other titles on the list using tf-idf matrix
    :param list_of_titles: List of titles to calculate similarities
    :param tfidf_matrix: Matrix of tf-idf vectors of titles
    :return: A list of similarities with first title on the list
    """
    sim_column = []
    for i in range(1, len(list_of_titles)):                                   # petlja za kosinusno poredjenje zadatog naslova i svih ostalih naslova
        cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[i])
        cosine_sim = round(float(cosine_sim), 4)                              # zaokruzivanje vrednosti na 4 decimale
        sim_column.append(cosine_sim)
    return sim_column


sim_column = calculating_similarities(list_of_titles, tfidf_matrix)
merged_table["similarity"] = sim_column                                   # upisivanje vrednosti slicnosti u kolonu "similarity"
temp_table = merged_table.sort_values("similarity", ascending=False).head(10)
print(temp_table)                                                         # ispisivanje 10 redova sa najvecom slicnoscu


# -----8----
vectorizer = CountVectorizer()
list_of_titles = list(merged_table["originalTitle"].values)                  # izdvajanje svih orginalnih naziva videa

cv_fit = vectorizer.fit_transform(list_of_titles)                             # vektorizacija svih reci

word_list = vectorizer.get_feature_names_out()                                # izdvajanje jedinstvenih reci
dict_of_words = dict(zip(word_list, np.asarray(cv_fit.sum(axis=0))[0]))       # pravljenje recnika sa zip funkcijom gde se prosledjuje rec i ukupan broj pojavljivanja te reci

table_of_words = pd.DataFrame.from_dict(data=dict_of_words, orient='index')      # pravljenje tabele sa recima i njihovim pojavljivanjem
table_of_words.to_csv("table_of_words.csv")


# -----9----
results = requests.get('https://api.coindesk.com/v1/bpi/currentprice.json')       # slanje get http zahteva i smestanje povratnog rezultata u promenljivu
res_parsed = results.json()                                                       # izvlacenje vrednosti iz rezultata u json formatu
rate_float = res_parsed["bpi"]["USD"]["rate_float"]                               # uzimanje vrednosti rate_float iz USD iz bpi
print(rate_float)


#-----Dodatni zadatak-----

# Možemo uzeti vrednost te kategorije koje trazimo u promenljivu vrednost_kategorije.
# Zatim izvrsiti sledeci algoritam:
# for predmet in lista_predmeta:
#     if predmet.kategorija == vrednost_kategorije:
#         lista_predmeta.remove(predmet)
#         lista_predmeta.insert(0,predmet)
# Ovim postupkom bi svi predmeti iz zeljene kategorije bili prvi na listi
