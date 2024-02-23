# Pobieranie danych: Funkcja fetch_spam_data pobiera zbiory danych zawierające wiadomości e-mail
# sklasyfikowane jako "ham" (prawdziwe wiadomości) i "spam" ze wskazanych URL-i, zapisuje je lokalnie i
# ekstrahuje zawartość archiwów. Następnie zwraca ścieżki do katalogów zawierających te wiadomości.
#
# Wczytywanie e-maili: Przy pomocy funkcji load_email, e-maile są wczytywane z plików i parsowane do
# obiektów Pythona, co umożliwia łatwy dostęp do różnych części wiadomości, takich jak nagłówki, treść i załączniki.
#
# Analiza struktury wiadomości: Za pomocą funkcji get_email_structure analizowana jest struktura każdej
# wiadomości e-mail, a funkcja structures_counter zlicza wystąpienia każdej struktury w zbiorze danych.
# Daje to wgląd w to, jak często różne formaty wiadomości (np. czysty tekst, HTML, wieloczęściowe) pojawiają się w danych.
#
# Podział danych na zbiory treningowe i testowe: Dane są dzielone na zbiory treningowe i testowe przy
# użyciu funkcji train_test_split. To przygotowuje dane do procesu trenowania i testowania modelu klasyfikującego
# wiadomości jako spam lub ham.
#
# Przetwarzanie HTML na zwykły tekst: Funkcja html_to_plain_text konwertuje treść wiadomości z formatu HTML
# na zwykły tekst. Jest to przydatne, ponieważ treść HTML może zawierać znaczniki i elementy, które nie niosą
# znaczącej informacji dla procesu klasyfikacji (np. znaczniki formatowania tekstu).
#
# Ekstrakcja treści wiadomości: Funkcja email_to_text przetwarza każdą wiadomość e-mail, wyodrębniając
# jej treść w formie zwykłego tekstu, niezależnie od formatu wiadomości (czy to czysty tekst, czy HTML).
# W przypadku wiadomości w formacie HTML, funkcja ta wykorzystuje html_to_plain_text do konwersji na zwykły tekst.

import tarfile
from pathlib import Path
import urllib.request
import email
import email.policy
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
import re
from html import unescape
import nltk
import urlextract
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score

# Wczyt danych
def fetch_spam_data():
    spam_root = "http://spamassassin.apache.org/old/publiccorpus/"
    ham_url = spam_root + "20030228_easy_ham.tar.bz2"
    spam_url = spam_root + "20030228_spam.tar.bz2"

    spam_path = Path() / "datasets" / "spam"
    spam_path.mkdir(parents=True, exist_ok=True)
    for dir_name, tar_name, url in (("easy_ham", "ham", ham_url),
                                    ("spam", "spam", spam_url)):
        if not (spam_path / dir_name).is_dir():
            path = (spam_path / tar_name).with_suffix(".tar.bz2")
            print("Pobieranie", path)
            urllib.request.urlretrieve(url, path)
            tar_bz2_file = tarfile.open(path)
            tar_bz2_file.extractall(path=spam_path)
            tar_bz2_file.close()
    return [spam_path / dir_name for dir_name in ("easy_ham", "spam")]

ham_dir, spam_dir = fetch_spam_data()

ham_filenames = [f for f in sorted(ham_dir.iterdir()) if len(f.name) > 20]
spam_filenames = [f for f in sorted(spam_dir.iterdir()) if len(f.name) > 20]

# print(len(ham_filenames))
# print(len(spam_filenames))

# Analiza składni tych wiadomości
def load_email(filepath):
    with open(filepath, "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)

ham_emails = [load_email(filepath) for filepath in ham_filenames]
spam_emails = [load_email(filepath) for filepath in spam_filenames]
# print(ham_emails[1].get_content().strip())
# print("==========================================================")
# print(spam_emails[6].get_content().strip())


# Przyjrzyjmy się różnym typom używanych struktur danych:
def get_email_structure(email):
    if isinstance(email, str):
        return email
    payload = email.get_payload()
    if isinstance(payload, list):
        multipart = ", ".join([get_email_structure(sub_email)
                               for sub_email in payload])
        return f"multipart({multipart})"
    else:
        return email.get_content_type()


def structures_counter(emails):
    structures = Counter()
    for email in emails:
        structure = get_email_structure(email)
        structures[structure] += 1
    return structures

# print(structures_counter(ham_emails).most_common())
# print(structures_counter(spam_emails).most_common())

# Przyjrzymy się nagłówkam:
# for header, value in spam_emails[0].items():
#     print(header, ":", value)
# print(spam_emails[0]["Subject"])


# Dzielimy dane na zbiory testowy i treningowy
X = np.array(ham_emails + spam_emails, dtype=object)
y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

# Przekształcenie składni HTML w zwykły tekst
def html_to_plain_text(html):
    text = re.sub('<head.*?>.*?</head>', '', html, flags=re.M | re.S | re.I)
    text = re.sub('<a\s.*?>', ' HYPERLINK ', text, flags=re.M | re.S | re.I)
    text = re.sub('<.*?>', '', text, flags=re.M | re.S)
    text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)
    return unescape(text)


html_spam_emails = [email for email in X_train[y_train==1]
                    if get_email_structure(email) == "text/html"]
sample_html_spam = html_spam_emails[7]
# print(sample_html_spam.get_content().strip()[:1000], "...")
# print(html_to_plain_text(sample_html_spam.get_content())[:1000], "...")


# Napiszmy teraz funkcję pobierającą daną wiadomość i zwracającą jej treść w postaci zwykłego tekstu, bez względu na jej format:
def email_to_text(email):
    html = None
    for part in email.walk():
        ctype = part.get_content_type()
        if not ctype in ("text/plain", "text/html"):
            continue
        try:
            content = part.get_content()
        except: # w przypadku problemów z kodowaniem
            content = str(part.get_payload())
        if ctype == "text/plain":
            return content
        else:
            html = content
    if html:
        return html_to_plain_text(html)

# print(email_to_text(sample_html_spam)[:100], "...")


# Wprowadźmy analizę słowotwóczą:
stemmer = nltk.PorterStemmer()
# for word in ("Computations", "Computation", "Computing", "Computed", "Compute",
#              "Compulsive"):
#     print(word, "=>", stemmer.stem(word))

# Potrzebny jest nam również mechanizm zastępowania adresów URL wyrazem "URL"
url_extractor = urlextract.URLExtract()
# some_text = "Czy wykryje github.com i https://youtu.be/7Pq-S557XQU?t=3m32s"
# print(url_extractor.find_urls(some_text))


# Składamy te elementy w jeden transformator
class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):
    '''
    EmailToWordCounterTransformer
    Ten transformator konwertuje e-maile na słownikowe zliczenia słów. Proces ten obejmuje kilka etapów przetwarzania tekstu:

    Konwersja tekstu e-maila na zwykły tekst: Używa wcześniej zdefiniowanej funkcji email_to_text do ekstrakcji treści e-maili.

    Czyszczenie tekstu: Obejmuje opcjonalne działania takie jak zamiana tekstu na małe litery, usuwanie interpunkcji,
    zamiana adresów URL na słowo "URL", zamiana liczb na słowo "NUMBER", oraz usuwanie nagłówków e-maila jeśli jest
    to zaznaczone w konstruktorze klasy.

    Stemming: Zmniejsza słowa do ich rdzeni (np., "fishing", "fished", "fisher" wszystkie stają się "fish").
    To ujednolica różne formy tego samego słowa, pozwalając modelowi na lepsze zrozumienie i klasyfikację tekstów.

    Zliczanie słów: Tworzy słownik zliczeń słów dla każdego przetworzonego e-maila.
    '''
    def __init__(self, strip_headers=True, lower_case=True,
                 remove_punctuation=True, replace_urls=True,
                 replace_numbers=True, stemming=True):
        self.strip_headers = strip_headers
        self.lower_case = lower_case
        self.remove_punctuation = remove_punctuation
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers
        self.stemming = stemming

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = []
        for email in X:
            text = email_to_text(email) or ""
            if self.lower_case:
                text = text.lower()
            if self.replace_urls and url_extractor is not None:
                urls = list(set(url_extractor.find_urls(text)))
                urls.sort(key=lambda url: len(url), reverse=True)
                for url in urls:
                    text = text.replace(url, " URL ")
            if self.replace_numbers:
                text = re.sub(r'\d+(?:\.\d*)?(?:[eE][+-]?\d+)?', 'NUMBER', text)
            if self.remove_punctuation:
                text = re.sub(r'\W+', ' ', text, flags=re.M)
            word_counts = Counter(text.split())
            if self.stemming and stemmer is not None:
                stemmed_word_counts = Counter()
                for word, count in word_counts.items():
                    stemmed_word = stemmer.stem(word)
                    stemmed_word_counts[stemmed_word] += count
                word_counts = stemmed_word_counts
            X_transformed.append(word_counts)
        return np.array(X_transformed)

X_few = X_train[:3]
X_few_wordcounts = EmailToWordCounterTransformer().fit_transform(X_few)
# print(X_few_wordcounts)


# Teraz musimy je przekształcić w postać wektorową
class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):
    '''
    2. WordCounterToVectorTransformer
    Następnie, transformator WordCounterToVectorTransformer konwertuje słownikowe zliczenia słów na wektory liczbowe, które mogą być bezpośrednio użyte przez algorytmy uczenia maszynowego. Realizowane kroki to:

    Budowa słownika (vocabulary): Wybiera najczęściej występujące słowa do utworzenia ograniczonego zestawu słów
    (określonego przez vocabulary_size). Każde słowo otrzymuje unikalny indeks.

    Transformacja zliczeń słów na wektory: Dla każdego e-maila tworzy wektor, gdzie każda pozycja odpowiada
    jednemu słowu ze słownika, a wartość w tej pozycji to liczba wystąpień danego słowa w e-mailu. Jeśli słowo
    nie występuje, jego wartość to 0. Jest to reprezentacja typu "bag of words" (BOW).

    Użycie macierzy rzadkiej: Z powodu dużej ilości zer (wiele słów ze słownika nie występuje w danym e-mailu)
    do przechowywania tych wektorów używana jest macierz rzadka (csr_matrix), co jest efektywne pod względem pamięciowym.
    '''
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size

    def fit(self, X, y=None):
        total_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_count[word] += min(count, 10)
        most_common = total_count.most_common()[:self.vocabulary_size]
        self.vocabulary_ = {word: index + 1
                            for index, (word, count) in enumerate(most_common)}
        return self

    def transform(self, X, y=None):
        rows = []
        cols = []
        data = []
        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)
        return csr_matrix((data, (rows, cols)),
                          shape=(len(X), self.vocabulary_size + 1))

vocab_transformer = WordCounterToVectorTransformer(vocabulary_size=10)
X_few_vectors = vocab_transformer.fit_transform(X_few_wordcounts)
# print(X_few_vectors)
# print(vocab_transformer.vocabulary_)

# Trenujemy model
preprocess_pipeline = Pipeline([
    ("email_to_wordcount", EmailToWordCounterTransformer()),
    ("wordcount_to_vector", WordCounterToVectorTransformer()),
])

X_train_transformed = preprocess_pipeline.fit_transform(X_train)

log_clf = LogisticRegression(max_iter=1000, random_state=42)
score = cross_val_score(log_clf, X_train_transformed, y_train, cv=3)
print(score.mean())


# Wynik precyzji/czułości na zbiorze testowym
X_test_transformed = preprocess_pipeline.transform(X_test)

log_clf = LogisticRegression(max_iter=1000, random_state=42)
log_clf.fit(X_train_transformed, y_train)

y_pred = log_clf.predict(X_test_transformed)

print(f"Precyzja: {precision_score(y_test, y_pred):.2%}")
print(f"Czułość: {recall_score(y_test, y_pred):.2%}")