Simple exercise from book Hands-On Machine Learning with Scikit-Learn, Keras, and Tensorflow - Aurélien Géron

Data Acquisition:
The fetch_spam_data function retrieves email datasets classified as "ham" (genuine emails) and "spam" from specified URLs, saves them locally, and extracts the contents of the archives. It returns paths to directories containing these emails.

Email Loading:
Using the load_email function, emails are loaded from files and parsed into Python objects, allowing easy access to different parts of the emails such as headers, content, and attachments.

Email Structure Analysis:
The get_email_structure function analyzes the structure of each email, while the structures_counter function counts occurrences of each structure in the dataset. This provides insights into how often different email formats (e.g., plain text, HTML, multipart) appear in the data.

Data Splitting into Train and Test Sets:
Data is split into training and test sets using the train_test_split function, preparing the data for training and testing a spam or ham classification model.

HTML to Plain Text Conversion:
The html_to_plain_text function converts email content from HTML format to plain text. This is useful as HTML content may contain formatting tags and elements that are not informative for the classification process.

Email Content Extraction:
The email_to_text function processes each email, extracting its content as plain text regardless of the message format (plain text or HTML).

Word Count Analysis:
A word count analysis is performed using NLTK's stemming to reduce words to their roots and URL extraction for replacing URLs with a standardized term ("URL").

Transformations into Vector Representation:
Two transformers (EmailToWordCounterTransformer and WordCounterToVectorTransformer) convert email word counts into numeric vectors suitable for machine learning algorithms, using a bag-of-words representation.

Model Training and Evaluation:
A logistic regression model is trained and evaluated using precision and recall scores on the test set, demonstrating the effectiveness of the text processing and classification pipeline.
