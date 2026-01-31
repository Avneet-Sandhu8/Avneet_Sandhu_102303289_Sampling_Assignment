# Avneet_Sandhu_102303289_Sampling_Assignment
#  Sampling Assignment – Credit Card Fraud Detection

 **Name :** Avneet Sandhu 
 **Roll No :**  102303289 

---

## Objective

The primary objective of this assignment is to:

- Understand the **critical importance of sampling techniques** in handling imbalanced datasets
- Analyze how **different sampling strategies** affect the performance of various machine learning models
- Apply practical sampling methods to real-world credit card fraud detection data

###  Dataset Context

The dataset used is a **credit card fraud detection dataset**, where:
-  Fraudulent transactions are **extremely rare** compared to legitimate transactions
-  This severe class imbalance can **mislead model accuracy metrics**
-  Proper sampling techniques are **essential** for fair model evaluation

---

##  Repository Structure

```
Sampling-Assignment/
│
├──  data/
│   └── Creditcard_data.csv         
│
├──  colab_notebooks/
│   └── Sampling_Assignment.ipynb    
│
├──  results/
│   └── accuracy_results.csv         
│
├──  screenshots/
│   └── (plots and outputs)         
│
└──  README.md                      
```

---

##  Dataset Description

| Attribute | Description |
|-----------|-------------|
| **Dataset Name** | Credit Card Transactions |
| **Target Column** | `Class` |
| **Class Labels** | `0` → Non-fraud (Legitimate) <br> `1` → Fraud (Fraudulent) |
| **Data Characteristic** | Highly Imbalanced |

---

##  Problem of Imbalanced Data

### Class Distribution Issue

In the original dataset:
-  **Non-fraud transactions**: Overwhelmingly high proportion
-  **Fraud transactions**: Extremely low proportion

### Consequences of Imbalance

This severe imbalance causes:
-  **Misleadingly high accuracy** but poor fraud detection capability
-  **Model bias** toward the majority class (non-fraud)
-  **Poor generalization** on minority class (fraud)

### Solution

**Balance the dataset before training models** to ensure fair learning across both classes.

---

##  Methodology

### Step 1: Analyze Class Imbalance
- Investigate the original dataset distribution
- Quantify the imbalance ratio between fraud and non-fraud classes

### Step 2: Balance the Dataset

Two complementary techniques were applied:

| Technique | Description |
|-----------|-------------|
| **Undersampling** | Reducing the majority class samples |
| **Oversampling** | Duplicating the minority class samples |

The final **balanced dataset** was used as the foundation for all sampling techniques.

---

##  Sampling Techniques Implementation

Five distinct samples were created using different statistical sampling methods:

| Sample ID | Sampling Technique | Description |
|-----------|-------------------|-------------|
| **Sampling 1** | Simple Random Sampling | Each sample has equal probability of selection |
| **Sampling 2** | Systematic Sampling | Samples selected at fixed intervals |
| **Sampling 3** | Stratified Sampling | Population divided into strata, samples from each |
| **Sampling 4** | Cluster Sampling | Population divided into clusters, random clusters selected |
| **Sampling 5** | Bootstrap Sampling | Sampling with replacement |

---

##  Sample Quality Validation

For each sample, the following **statistical measures** were calculated to ensure representativeness:

-  **Mean** - Central tendency measure
-  **Median** - Middle value in ordered data
-  **Variance** - Spread of data points
-  **Standard Deviation** - Average deviation from mean
-  **Skewness** - Asymmetry of distribution
-  **Kurtosis** - Tailedness of distribution

This rigorous validation ensured that all samples are **statistically representative** of the population.

---

##  Machine Learning Models

Five different classification models were implemented and evaluated:

| Model Code | Model Name | Type |
|------------|------------|------|
| **M1** | Logistic Regression | Linear Classifier |
| **M2** | Decision Tree | Tree-based Classifier |
| **M3** | Random Forest | Ensemble Method |
| **M4** | K-Nearest Neighbors (KNN) | Instance-based Learning |
| **M5** | Support Vector Machine (SVM) | Kernel-based Classifier |

Each model was trained and tested on **all five samples** to evaluate sampling impact.

---

##  Performance Results

### Accuracy Comparison Table (%)

| Model | Sampling 1 | Sampling 2 | Sampling 3 | Sampling 4 | Sampling 5 |
|-------|-----------|-----------|-----------|-----------|-----------|
| **M1_Logistic** | 88.04 | 94.57 | 90.22 | 91.3 | 84.78 |
| **M2_DecisionTree** | 100.0 | 97.83 | 95.65 | 95.65 | 98.91 |
| **M3_RandomForest** | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| **M4_KNN** | 90.22 | 90.22 | 90.22 | 89.13 | 91.3 |
| **M5_SVM** | 80.43 | 66.3 | 55.43 | 59.78 | 72.83 |

###  Key Performance Insights

-  **Random Forest (M3)** achieved perfect **100% accuracy** across ALL sampling techniques
-  **Decision Tree (M2)** showed excellent performance with accuracy ranging from **95.65% to 100%**
-  **Logistic Regression (M1)** demonstrated consistent performance between **84.78% and 94.57%**
-  **KNN (M4)** maintained stable accuracy around **89-91%** across different samples
-  **SVM (M5)** showed high variability, ranging from **55.43% to 80.43%**

---

##  Detailed Analysis

### Model Performance Analysis

####  Best Performers
- **Random Forest** consistently achieved perfect classification
- **Decision Tree** showed robust performance with minimal variation

####  Sampling Impact
- All sampling techniques worked well with Random Forest
- SVM showed significant sensitivity to sampling method
- **Sampling 2 (Systematic Sampling)** generally produced strong results

####  Consistency
- **Random Forest** and **Decision Tree** showed excellent stability
- **Logistic Regression** and **KNN** maintained moderate consistency
- **SVM** demonstrated high sensitivity to sampling technique choice

### Key Observations

1. **Ensemble methods** (Random Forest) outperformed individual classifiers
2. **Tree-based models** (Decision Tree, Random Forest) handled balanced data exceptionally well
3. **SVM** performance varied significantly, suggesting sensitivity to data distribution
4. **Sampling technique choice** has substantial impact on certain models but minimal effect on others

---

##  Conclusions

### Critical Findings

1.  **Sampling is absolutely essential** for handling imbalanced datasets effectively
2.  **Different models respond differently** to various sampling techniques
3.  **Tree-based and ensemble methods** demonstrated superior and consistent performance
4.  **Balanced data leads to more reliable** model training and evaluation
5.  **Model selection matters** - choose algorithms that handle your sampling strategy well

### Best Practices Learned

- **Always balance your dataset** before training on imbalanced problems
- **Test multiple sampling techniques** to find the optimal approach
- **Accuracy alone is misleading** - consider precision, recall, and F1-score
- **Ensemble methods provide robustness** against sampling variations
- **Validate sample quality** using statistical measures

### Recommendations

- Use **Random Forest or Decision Tree** for credit card fraud detection
- Apply **systematic or bootstrap sampling** for consistent results
- **Validate on multiple samples** to ensure model generalization
- Consider **additional metrics** beyond accuracy for imbalanced problems

---

##  Technologies & Libraries

### Development Environment
- **Platform**: Google Colab
- **Language**: Python 3.8+

### Core Libraries

| Library | Purpose |
|---------|---------|
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computing |
| **Matplotlib** | Data visualization |
| **Scikit-learn** | Machine learning algorithms |
| **SciPy** | Statistical computations |

---

##  Access the Implementation

The complete implementation with detailed code, visualizations, and explanations is available in:

```
 colab_notebooks/Sampling_Assignment.ipynb
```

---

##  Learning Outcomes

Through this assignment, the following concepts were mastered:

-  Understanding and implementing various **sampling techniques**
-  Handling **class imbalance** in real-world datasets
-  Evaluating **multiple machine learning models** systematically
-  Analyzing the **impact of data preprocessing** on model performance
-  Making **data-driven decisions** for model selection

---

##  Contact

For questions or discussions about this project:

- **Name**: Avneet Sandhu
- **Roll No**: 102303289

---
