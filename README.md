# Sentiment Analysis Project README

## Project Title: Sentiment Analysis of Product Reviews

### Overview
This project performs sentiment analysis on product reviews using various machine learning models. The goal is to classify reviews as either **POSITIVE**, **NEGATIVE**, or **NEUTRAL**. The notebook demonstrates data preprocessing, feature engineering, and model evaluation using different techniques.

### Dataset
The project utilizes a CSV file named `Product_Reviews.csv`, which contains product reviews and their corresponding sentiment labels. The columns in the dataset are:
-   **Product ID**: A unique identifier for the product.
-   **Product Review**: The text content of the review.
-   **Sentiment**: The sentiment label, which can be POSITIVE, NEGATIVE, or NEUTRAL.

### Key Steps & Techniques
The Jupyter Notebook performs the following steps:

1.  **Data Loading**: The `pandas` library is used to read the `Product_Reviews.csv` file into a DataFrame.
2.  **Text Preprocessing**:
    * **Lowercase Conversion**: The product reviews are converted to lowercase to ensure consistency.
    * **Stopword Removal**: Common English stopwords are removed from the reviews to clean the text and prepare it for analysis.
3.  **Feature Engineering**:
    * **Bag of Words (BOW)**: This technique creates a feature vector for each review based on the frequency of words.
    * **TF-IDF (Term Frequency-Inverse Document Frequency)**: This method weighs word importance by a combination of how often a word appears in a review and how rare the word is across all reviews.
4.  **Model Training & Evaluation**:
    * The project uses a split-based evaluation function (`evaluate_accuracy`) to test the performance of the models.
    * **Logistic Regression**: The notebook trains and evaluates a Logistic Regression model on various feature combinations (Lemmatized + TF-IDF, Stemmed + TF-IDF, Lemmatized + BOW, and Stemmed + BOW).
    * **Naive Bayes**: A Multinomial Naive Bayes model is also trained and evaluated using the same feature combinations.

### How to Run the Notebook
1.  **Dependencies**: Ensure you have Python installed. The required libraries are `pandas`, `nltk`, and `scikit-learn`. You can install them using pip:
    ```bash
    pip install pandas nltk scikit-learn
    ```
2.  **Dataset**: Place the `Product_Reviews.csv` file in the same directory as the Jupyter Notebook.
3.  **Run**: Execute the cells in the `Sentiment_Analysis.ipynb` notebook in order. The notebook will handle all the data preprocessing, model training, and evaluation steps automatically.
