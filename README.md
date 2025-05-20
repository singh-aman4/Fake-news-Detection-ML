<h1>Fake News Detection Using Machine Learning</h1>

<h2>Introduction</h2>

This project focuses on the development of a Fake News Detection system using classical machine learning algorithms. The objective is to create a model capable of identifying whether a given news article is real or fake based on its textual content. The project leverages natural language processing techniques for text preprocessing and feature extraction, followed by training and evaluating multiple machine learning classification models.

Fake news has become a major issue in the digital information age, with misinformation spreading quickly across platforms. A system that can accurately classify the legitimacy of news content has practical applications in journalism, media, and information verification platforms.

<h2>Dataset Description</h2>

The dataset used for this project is publicly available and was sourced from Kaggle. It consists of two separate CSV files: one containing real news articles and the other containing fake news articles. Both datasets include columns such as the title of the article, the main text body, subject, and date of publication.

For this project, only the text content was considered relevant for classification. Labels were assigned to each record, with 1 representing real news and 0 representing fake news. The two datasets were merged and shuffled to create a balanced training environment.

<h2>Data Preprocessing</h2>

Before training the machine learning models, the raw text data was preprocessed using several steps to make it suitable for analysis and classification. Preprocessing included the following operations:
	•	Conversion of all text to lowercase for normalization.
	•	Removal of punctuation marks and special characters to reduce noise.
	•	Tokenization of text into individual words.
	•	Removal of stopwords using the NLTK library to eliminate common but irrelevant words.
	•	Stemming using the PorterStemmer algorithm to reduce words to their root form.

All preprocessing steps were implemented through a custom function called clean_text, which was applied to the entire dataset to ensure consistency between training and manual testing inputs.

<h2>Feature Extraction</h2>

To convert the cleaned textual data into a numerical format suitable for machine learning models, the TF-IDF (Term Frequency-Inverse Document Frequency) vectorization technique was used. This method emphasizes words that are more unique to a given document while down-weighting commonly used words. The TfidfVectorizer from the sklearn.feature_extraction.text module was used with a fixed vocabulary size to maintain computational efficiency and reduce overfitting.

<h2>Model Building and Training</h2>

The project involved training and comparing multiple machine learning classification algorithms. These models were trained using the TF-IDF-transformed data and evaluated on a held-out test set. The following models were implemented:
	•	Logistic Regression: A linear model used for binary classification. It performed reasonably well on this dataset and served as a baseline.
	•	Decision Tree Classifier: A non-linear model that creates a tree-like structure of decisions. It tends to overfit but is useful for understanding data structure.
	•	Random Forest Classifier: An ensemble model based on decision trees. It provides improved accuracy and robustness by averaging multiple decision trees.
	•	Support Vector Machine (SVM): A powerful model for high-dimensional spaces, especially effective with a linear kernel in this case. It was used for its ability to handle text classification tasks effectively.

Each model was trained on 80% of the dataset and tested on the remaining 20%. Accuracy scores were calculated for each model to compare performance.

<h2>Evaluation and Visualization</h2>

To better understand the performance of the classifiers, confusion matrices were plotted using the seaborn library. These matrices help visualize the number of true positives, false positives, true negatives, and false negatives.

The confusion matrices were generated using the confusion_matrix function from sklearn.metrics and plotted with sns.heatmap. This helped in diagnosing whether the model was biased toward one class and provided insights into specific types of errors made by the classifier.

Additionally, classification reports including precision, recall, and F1-score were generated to give a more detailed understanding of the model’s effectiveness beyond raw accuracy.

<h2>Manual Testing Function</h2>

A custom function named manual_testing was developed to allow real-time testing of new, unseen news articles. This function performs the following steps:
	•	Accepts a news text input from the user.
	•	Applies the same cleaning process used during training to ensure consistency.
	•	Transforms the cleaned input using the already fitted TF-IDF vectorizer.
	•	Uses a trained machine learning model to predict whether the news is fake or real.
	•	Returns a clear classification output.

This function allows for practical experimentation and demonstration of the model’s real-world application.

<h2>Limitations and Improvements</h2>

Although the models provided acceptable accuracy, they showed limitations when it came to generalizing on new, manually entered data. Some of the possible reasons include:
	•	The dataset might be slightly imbalanced despite shuffling.
	•	The manually entered news samples could vary significantly in tone or structure from the training data.
	•	More advanced models like deep learning or transformer-based language models (e.g., BERT) could be used for further improvement.

Additionally, more comprehensive text preprocessing such as lemmatization, n-gram feature extraction, and using domain-specific stopwords could improve results.

<h2>Future Enhancements</h2>

To improve and expand this project, the following enhancements are suggested:
	•	Integration with a frontend interface using Streamlit or Flask for interactive use.
	•	Use of larger and more diverse datasets.
	•	Implementation of deep learning models such as LSTM or fine-tuned BERT for more nuanced language understanding.
	•	Deployment of the model as a web application or API for public use.
