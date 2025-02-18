# Sentiment Analysis with Logistic Regression

## **Project Overview**
This project involves building a **logistic regression model from scratch** to classify **movie reviews** as either **positive (1) or negative (0)** using word-level sentiment scores from the **VADER Sentiment Analysis** tool.

## **Goal**
The objective is to implement **logistic regression with gradient descent** to accurately predict sentiment based on the dataset provided.

## **Dataset & Preprocessing**
- **Dataset**: A CSV file (`train.csv`) containing movie reviews and their corresponding sentiment labels.
- **Data Cleaning**:
  - Drop rows with **Neutral Sentiment**.
  - Convert text into numerical features using **VADER's word-level sentiment scores**.
  - Replace each word with its corresponding sentiment score from the **VADER lexicon**.
  - If a word is **not found** in the VADER lexicon, assign a **score of 0**.
  - Ensure all embeddings have the **same length** by **padding shorter sequences with zeros**.

### **Example Transformation**
#### **Input Sentence:**
```plaintext
"I really love this amazing product!"
```
#### **Numerical Representation:**
```plaintext
[0, 0.1779, 3.2, 0, 2.9, 0]
```
#### **After Padding:**
```plaintext
[0, 0.1779, 3.2, 0, 2.9, 0, 0, 0]
```

## **Model Implementation**
- Implement **logistic regression** from scratch.
- Use **gradient descent** for optimization.
- **Do not use pre-built deep learning models**.
- Utilize libraries like `pandas`, `numpy`, and `matplotlib` for **data processing and visualization**.

## **Training Strategy**
1. **Split `train.csv`** into **training (80%)** and **validation (20%)** sets.
2. Use a **classification threshold of 0.5**:
   - **Predicted Probability ≥ 0.5** → **Classify as 1 (Positive)**.
   - **Predicted Probability < 0.5** → **Classify as 0 (Negative)**.
3. Use **gradient descent** to optimize model weights.
4. Evaluate model performance using **accuracy score**.

## **Model Evaluation**
- **Metric Used**: Accuracy Score
- **Formula**:
  ```
  Accuracy = (Number of Correct Predictions) / (Total Predictions)
  ```

### **Example Code for VADER Sentiment Analysis**
```python
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER lexicon
nltk.download('vader_lexicon')

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# Access the VADER lexicon
vaderLex = analyzer.lexicon

# Example
print(vaderLex.get("happy"))  # Output: 2.7
```

## **Submission Format**
The final output must be saved as a **CSV file** with two columns:

| id  | Vader_Binary_Sentiment |
|-----|------------------------|
| 0   | 0                      |
| 1   | 1                      |
| 2   | 1                      |
| 3   | 0                      |

- Ensure your submission file **matches this format** to be correctly evaluated.

## **How to Run the Project**
1. Install required dependencies:
   ```bash
   pip install numpy pandas nltk matplotlib scikit-learn
   ```
2. Run the preprocessing script to clean the dataset.
3. Train the logistic regression model using gradient descent.
4. Evaluate the model using accuracy score.
5. Generate predictions on the test dataset.
6. Save predictions as `submission.csv`.



## **Tags**
- `Accuracy Score`
- `Sentiment Analysis`
- `Logistic Regression`
- `Gradient Descent`

---

**Authors:** Moughel Mohamed Souhail,Haytham Raiss,Mouad Wadi

For questions or feedback, feel free to reach out!