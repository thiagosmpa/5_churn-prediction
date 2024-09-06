# Churn Prediction using Autogluon
## Project 5


This project utilizes Autogluon to predict customer churn, representing the potential of a customer to leave a subscription service. The dataset is sourced from Kaggle: [https://www.kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Requirements

* Python 3.10
* Libraries listed in `requirements.txt`

## Data Analysis

Initial analysis reveals a churn rate of 26%, which is concerning. [Figure 1] illustrates this distribution. This necessitates further analysis to identify service improvement areas and reduce churn.

![Figure 1]('images/churn.png')

Data inspection identified the `TotalCharges` column as numeric, but imported as an object. This was rectified using the following code:

```python
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
```

This code converts non-numeric values to NaN using `coerce` and fills them with the median.

## Numerical Variables

Focusing on `tenure` and `MonthlyCharges`, we observe the following:

![Figure 2]('images/num.png')

* `MonthlyCharges` (in dollars) and `tenure` (in months) show an inverse relationship with churn. Longer tenure and lower monthly charges correlate with lower churn.
* The median `MonthlyCharges` for churned customers is around $80, compared to $65 for retained customers. This suggests a pricing strategy opportunity.

## Categorical Variables

Analysis shows that longer contract durations and payment via "Electronic check" are associated with higher churn, despite these categories having a large customer base. [Figure 3] visualizes this.

![Figure 3]('images/cat.png')

While these categories can't be eliminated, the "Electronic check" payment method warrants attention. Strategies for easier payment could improve customer retention.

## Preprocessing and Modeling

Various preprocessing approaches were tested. Label encoding for the `Contract` column proved effective, as its categories ("Monthly", "One year", "Two year") have a progressive nature that aligns with the observed churn pattern. Other categorical columns were one-hot encoded using pandas' `get_dummies`.

Despite achieving acceptable accuracy, recall was low, likely due to the imbalanced dataset. SMOTE was tested to address this, but the best configuration utilized Autogluon's "medium quality" preset, label encoding for `Contract`, and no SMOTE.

## Results

The best model performance is shown below.

![Figure 4]('images/acc.png')

This project highlights the importance of careful data analysis and feature engineering in achieving optimal churn prediction results. 
