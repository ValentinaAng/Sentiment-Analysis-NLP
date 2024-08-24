# Title: Twitter US Airlines Sentiment Analysis

## Project Overview:

This project aims to analyze the sentiment of tweets directed at various U.S. airlines. The dataset used in this analysis is the "Twitter US Airline Sentiment" dataset, which contains tweets classified as positive, neutral, or negative. The goal of this project is to build a machine learning model that can accurately predict the sentiment of tweets based on their text content.

Initially, I will develop a model without using pre-trained models, employing `GloVe (Global Vectors for Word Representation) embeddings in combination with logistic regression`. This approach will provide a baseline for understanding how well traditional text vectorization techniques and a simple classifier perform on this task.

In the second part of the project, I will leverage a pre-trained model, specifically `BERT (Bidirectional Encoder Representations from Transformers)`, to perform sentiment analysis. Using BERT, I aim to improve the model's performance by taking advantage of its deep contextual understanding of language, which is expected to capture nuanced sentiment information more effectively than traditional methods.

## About Dataset:

The "Twitter US Airline Sentiment" dataset is publicly available on Kaggle and contains over 14,000 tweets about major U.S. airlines, labeled with sentiment categories. The dataset provides valuable insights into customer satisfaction and sentiment toward different airlines.

[Dataset URL](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)

**Dataset Features**

| Feature                       | Description                                                                           |
|-------------------------------|---------------------------------------------------------------------------------------|
| `tweet_id`                    | Unique identifier for each tweet.                                                     |
| `airline_sentiment`           | The sentiment label for the tweet, which can be one of the following categories:      |
|                               | - positive                                                                           |
|                               | - neutral                                                                            |
|                               | - negative                                                                           |
| `airline_sentiment_confidence`| Confidence level in the sentiment classification.                                     |
| `negativereason`              | The reason for negative sentiment, if applicable (e.g., "Late Flight," "Cancelled Flight"). |
| `negativereason_confidence`   | Confidence level in the negative reason classification.                               |
| `airline`                     | The airline to which the tweet is directed (e.g., "United," "American").              |
| `airline_sentiment_gold`      | A gold standard label for the tweet's sentiment, if available.                        |
| `name`                        | The name of the user who posted the tweet.                                            |
| `negativereason_gold`         | A gold standard label for the negative reason, if available.                          |
| `retweet_count`               | The number of retweets for the tweet.                                                 |
| `text`                        | The actual tweet content.                                                             |
| `tweet_coord`                 | Coordinates of the tweet, if available.                                               |
| `tweet_created`               | The timestamp when the tweet was created.                                             |
| `tweet_location`              | The location of the tweet, if available.                                              |
| `user_timezone`               | The timezone of the user who posted the tweet.                                        |


## Table of Content:
1. Data Analysis
2. Data Preprocessing
3. Pretrained-Glove Embeddings
4. Model Training and Evaluating
5. Prediction
6. Conclusion


## Results:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.89      | 0.70   | 0.78     | 1818    |
| 1     | 0.57      | 0.71   | 0.63     | 460     |
| 2     | 0.47      | 0.68   | 0.55     | 613     |
| **Accuracy** |  |  | **0.70** | **2891** |
| **Macro Avg** | 0.64 | 0.70   | 0.66     | 2891    |
| **Weighted Avg** | 0.75 | 0.70   | 0.71     | 2891    |

## Conclusion:

#### Predictions:

1. **For the input `['i hate you']`**:
   - **Predicted sentiment**: Negative
   - **Probability distribution**:
     - **Class 0 (negative)**: 55.9%
     - **Class 1 (positive)**: 29.8%
     - **Class 2 (neutral)**: 14.3%

2. **For the input `['i love you, you are beautiful']`**:
   - **Predicted sentiment**: Positive
   - **Probability distribution**:
     - **Class 0 (negative)**: 1.6%
     - **Class 1 (positive)**: 97.1%
     - **Class 2 (neutral)**: 1.2%

These predictions align with the expected sentiments of the inputs, demonstrating that the model can provide meaningful predictions for new data.

**Recommendations for Improvement:**
- **Address Class Imbalance**: Balancing the target classes could enhance the model's ability to correctly classify minority classes.
- **Explore Other Models**: Investigate alternative algorithms or ensemble methods that may perform better with imbalanced datasets.
- **Enhance Features**: Incorporate additional features or advanced preprocessing techniques to improve model performance.

Overall, while the current model provides valuable insights, there is potential for improvement through addressing class imbalance and exploring other modeling approaches.
