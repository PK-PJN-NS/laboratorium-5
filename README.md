# Przetwarzanie języka naturalnego
# Laboratorium 5: rekomendacje oparte na treści

1. Skopiować zawartość niniejszego repozytorium
na dysk lokalny. Wybrać temat rekomendacji:
[dzieła literackie](https://drive.google.com/open?id=1gTd7BCxM_C9aPAmvfVO9F3uCi875fBEL)
albo [filmy](https://drive.google.com/open?id=18amHXSBYJupR6drnVhjS8qYzf3VHVVlS).
Pobrać i rozpakować odpowiedni plik, zawierający
bazę danych SQLite z treścią artykułów polskiej
Wikipedii na dany temat.

    ```
    git clone https://github.com/PK-PJN-NS/laboratorium-5.git
    cd laboratorium-5
    unzip literatura.zip <<<LUB>>> unzip filmy.zip
    ```

2. Zainstalować `scikit-learn` — bibliotekę
z narzędziami do uczenia maszynowego,
oraz `mwparserfromhell` — pakiet do obróbki
tekstów korzystających ze znaczników MediaWiki.

    ```
    pip install sklearn
    pip install mwparserfromhell
    ```

3. Wymyślić i zapisać tytuły pięciu dzieł,
odpowiednio literackich albo filmowych,
które wejdą w skład zbioru testowego
dla różnych wersji opracowywanego systemu.

4. Systemy rekomendacji oparte na treści
(*content-based recommendations*)
polecają obiekty na podstawie ich cech.
Tym się różnią od systemów rekomendacji
korzystających z zachowań użytkowników
(*collaborative filtering*)
i systemów hybrydowych,
którymi się nie zajmujemy,
bo mają niewiele wspólnego
z przetwarzaniem języka naturalnego.

5. Nasz system będzie rekomendować
tytuły dzieł literackich albo filmów
na podstawie ich opisów w polskiej Wikipedii.
Żeby do tego doszło, należy przekształcić
tekst tych opisów na liczby.
Zwykle każdemu obiektowi (dziełu)
przypisuje się wektor (listę/tablicę jednowymiarową)
liczb rzeczywistych.

6. Najprostszy sposób przekształcania
opisów słownych na liczby
korzysta z multizbiorów wyrazów
(angielska nazwa *bag of words* jest barwniejsza).
W bibliotece `scikit-learn`
służy do tego `CountVectorizer`,
tworzący na podstawie listy napisów rzadką macierz,
której wiersze zawierają liczebności
poszczególnych wyrazów
w odpowiednim napisie.
Najlepiej wyjaśni to przykład.

    ```python
    >>> # Używamy własnego token_pattern w celach dydaktycznych,
    ... # ponieważ domyślny token_pattern ignoruje jednoliterowe wyrazy.
    >>> v = CountVectorizer(token_pattern=r'\b\w+\b')
    >>> X = v.fit_transform([
    ...     'Ogniem i mieczem',
    ...     'W pustyni i w puszczy'])
    >>> # Wyrazy są zamieniane na małe litery.
    >>> v.get_feature_names()
    ['i', 'mieczem', 'ogniem', 'pustyni', 'puszczy', 'w']
    >>> X.todense()
    matrix([[1, 1, 1, 0, 0, 0],
            [1, 0, 0, 1, 1, 2]])
    >>> # ['i', 'mieczem', 'ogniem']
    ... # ['i', 'pustyni', 'puszczy', 'w', 'w']
    ```

7. Uruchomić program `zadanie.py`.
Do konstruktora klasy `CountVectorizer`
przekazujemy argumenty,
oznaczające tworzenie cech z wyrazów
(w odróżnieniu np. od *N*-gramów)
i odrzucanie wyrazów, które łącznie
występują w tekstach mniej niż 3 razy:

    ```python
    vectorizer = CountVectorizer(
            analyzer='word',
            min_df=3)
    ```

8. Program `zadanie.py` tworzy plik
o nazwie `model.pickle`.
Zmienić jego nazwę na `baseline.pickle`
i uruchomić program `rekomenduj.py`
z argumentem `baseline.pickle`.
W Unixie robi się to tak:

    ```
    ./rekomenduj.py baseline.pickle
    ```

9. Wpisujemy tytuły naszych pięciu dzieł
i kopiujemy do dokumentu tekstowego
podane przez program rekomendacje
(w tej wersji programu `zadanie.py` będą one nędzne;
wkrótce je poprawimy).
Jeśli któraś rekomendacja nic nam nie mówi,
zaglądamy do Wikipedii,
żeby sobie wyrobić o niej zdanie.
Przy wpisywaniu tytułów
nie boimy się naciskania klawisza tabulacji,
żeby je uzupełniać.
Powtarzam: nie boimy się.
Dla zaznajomienia się z działaniem uzupełniania
proszę wpisać dwa pierwsze znaki upatrzonego tytułu
i dwa razy nacisnąć klawisz tabulacji.
Uwaga: z powodu niedoskonałej hierarchii kategorii
w Wikipedii bazy danych zawierają śladowe ilości artykułów,
które nie dotyczą dzieł literackich lub filmowych.
Jeśli system rekomenduje tytuł takiego artykułu,
nie należy tego uważać za obciążającą go wadę,
tylko ignorować ten tytuł
przy porównywaniu jakości rekomendacji.
Program `rekomenduj.py` podaje takie tytuły dzieł,
dla których jak najmniejsza jest *odległość kosinusowa*
multizbiorów wyrazów w dotyczących ich artykułach
od multizbioru artykułu o danym dziele.
Odległość kosinusowa jest równa 1 − cos *θ*,
gdzie *θ* to kąt pomiędzy wektorami,
reprezentującymi odpowiednie multizbiory
(patrz:
[*cosine similarity*](https://en.wikipedia.org/wiki/Cosine_similarity)
w Wikipedii).

10. Główną wadą multizbiorów wyrazów
jest przypisywanie jednakowej wagi
liczebności wyrazów częstych (np. *się*, *rok*, *film*),
które mało wpływają na znaczenie tekstu,
i wyrazów rzadkich (np. *verne*, *żeglarz*),
których współwystępowanie
powinno być silniejszym sygnałem
podobieństwa tekstów.
Zaradzimy temu na dwa sposoby:

    a. Użyjemy listy *wyrazów nieinformatywnych* (*stop words*),
    których wystąpienia nie będą zliczane.
    Skopiować listę takich wyrazów
    ze strony pod adresem https://pl.wikipedia.org/wiki/Wikipedia:Stopwords
    do zmiennej `STOP_WORDS`,
    dopisać `stop_words=STOP_WORDS`
    jako argument `CountVectorizer`,
    stworzyć nowy model przez uruchomienie
    programu `zadanie.py`,
    skopiować testowe rekomendacje do dokumentu tekstowego
    i ocenić, czy są sensowniejsze.
    Opcjonalnie przenieść `model.pickle` pod inną nazwę.

    b. Zastąpić `CountVectorizer` przez `TfidfVectorizer`,
    stworzyć nowy model itd.
    `TfidfVectorizer` korzysta ze wzoru
    [TF-IDF](https://pl.wikipedia.org/wiki/TFIDF)
    (*term frequency-inverse document frequency*),
    który przypisuje niższe wagi częstym cechom,
    a wyższe — rzadkim. Oto przykład:

    ```python
    >>> v = TfidfVectorizer(token_pattern=r'\b\w+\b')
    >>> X = v.fit_transform([
    ...     'Ogniem i mieczem',
    ...     'W pustyni i w puszczy'])
    >>> v.get_feature_names()
    ['i', 'mieczem', 'ogniem', 'pustyni', 'puszczy', 'w']
    >>> X.todense()
    matrix([[0.449, 0.631, 0.631, 0.   , 0.   , 0.   ],
            [0.279, 0.   , 0.   , 0.392, 0.392, 0.784]])
    ```

11. Przeprowadzić poniższe eksperymenty.
Jeśli któryś eksperyment nie poprawia jakości modelu,
należy go wycofać przed przejściem do kolejnego podpunktu.

    a. Dodać do `STOP_WORDS` wybrane (z głową!)
    najczęstsze wyrazy, wypisane przez program
    `zadanie.py`, np. 'kategoria', 'linki', 'zewnętrzne' itp.
    Stworzyć nowy model itd.

    b. Dotychczas usuwaliśmy liczby z tekstów artykułów.
    Sprawdzić wpływ ich nieusuwania na jakość modelu:
    usunąć fragment `0-9` z napisu przy zmiennej `NONLETTERS_RE`.
    Stworzyć nowy model itd.

    c. Dotychczas nie pozostawialiśmy argumentów
    szablonów Wikipedii w tekscie artykułów.
    Wiele artykułów ma na końcu rozbudowane
    szablony nawigacyjne, np. do innych dzieł
    tego samego twórcy.
    Sprawdzić wpływ pozostawiania argumentów szablonów
    na jakość modelu:
    zamienić wartość `keep_template_params`
    w wywołaniu funkcji `mwp.parse()`
    z `False` na `True`.
    Stworzyć nowy model itd.

    d. Ze znanych powodów nie stosujemy
    stemmingu ani lematyzacji wyrazów.
    Sprawdzić wpływ na jakość modelu
    używania ich substytutu: *N*-gramów.
    Zmienić w wywołaniu `TfidfVectorizer`
    `analyzer` na `'char_wb'`,
    dopisać `ngram_range=(5, 5)`
    i `max_features=40_000` (na przykład).
    Parametr `max_features`, który określa,
    ile najczęstszych *N*-gramów należy
    pozostawić w modelu, jest niezbędny,
    żeby model miał rozsądne rozmiary,
    bo hasła zawierają znacznie więcej
    różnych *N*-gramów niż różnych wyrazów.
    Stworzyć nowy model itd.

    e. Zadanie nadobowiązkowe:
    próbować poprawić najlepszy
    do tej pory model.
    Skorzystać z
    [dokumentacji klasy `TfidfVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer).
    Można na przykład spróbować
    dodać argument `max_df=0.7`.
    Stworzyć nowy model itd.

12. W sprawozdaniu zamieścić
opisy sposobu tworzenia kolejnych modeli,
rekomendacje otrzymane przy ich użyciu
i słowną ocenę jakości tych rekomendacji.
Wytłuścić punkt sprawozdania,
odpowiadający najlepszemu modelowi.
