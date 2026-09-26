# 🚢 Titanic: Machine Learning from Disaster

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)](https://www.kaggle.com/competitions/titanic)

> **Predicting survival on the Titanic with advanced Feature Engineering and Ensemble Learning (Random Forest Classifier).**  
> Achieved **81.56% Validation Accuracy** through domain-informed feature transformations and tree regularization.

---

## 📌 Project Overview

The sinking of the RMS Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, resulting in the deaths of 1,502 out of 2,224 passengers and crew.

While there was an element of chance in surviving, clear patterns emerge indicating that certain demographics—such as women, children, and upper-class passengers—were much more likely to survive than others.

The goal of this project is to build an end-to-end Machine Learning pipeline that predicts passenger survival based on passenger metadata (demographics, passenger class, family relationships, and ticket details).

---

## 🚀 Key Highlights & Results

- **Validation Accuracy**: **`81.56%`** on a held-out validation set (80/20 train-test split, `random_state=42`).
- **Engineered Features**: Extracted social titles, created family size metrics, flag for solo travelers, and imputed missing attributes cleanly.
- **Controlled Complexity**: Regulated Random Forest depth (`max_depth=5`, `min_samples_split=5`) to prevent overfitting on small tabular datasets.
- **Production Pipeline**: Automated alignment of training and testing categorical encodings using `reindex`.

---

## 🛠️ Feature Engineering Pipeline

Feature engineering played the most critical role in elevating model accuracy. Raw features were processed and transformed as follows:

| Feature Name | Source | Transformation Description | Rationale |
| :--- | :--- | :--- | :--- |
| **`Title`** | `Name` | Extracted titles using regex (`Mr`, `Mrs`, `Miss`, `Rare`). | Social status and marital status strongly dictate rescue priority. |
| **`FamilySize`** | `SibSp` + `Parch` + 1 | Total number of family members traveling together. | Small-to-medium families had higher survival rates than individuals or very large families. |
| **`IsAlone`** | `FamilySize` | Binary flag (`1` if traveling solo, `0` otherwise). | Solo travelers had significantly lower survival odds compared to escorted passengers. |
| **`Sex`** | `Sex` | Binary mapping (`male: 0`, `female: 1`). | Fundamental protocol: "Women and children first". |
| **`Age` Imputation** | `Age` | Replaced missing values with training median (`median()`). | Preserves distribution without leaking test data into training statistics. |
| **`Fare` Imputation** | `Fare` | Replaced missing values with training median. | Accommodates missing fare values in test sets safely. |
| **`Embarked` Imputation** | `Embarked` | Replaced missing values with training mode (`mode()[0]`). | Categorical baseline imputation for boarding ports. |
| **One-Hot Encoding** | `Embarked`, `Title` | `pd.get_dummies(..., drop_first=True)` with column alignment. | Converts categorical variables into numerical dummy indicators without dummy variable trap. |

---

## 📊 Feature Importance Breakdown

Feature importances derived from the trained `RandomForestClassifier`:

```text
Sex           ████████████████████████████  26.52%
Title_Mr      █████████████████████        20.92%
Pclass        ████████████                  11.68%
Fare          ███████████                   10.73%
Age           ████████                       7.70%
FamilySize    ███████                        6.51%
Title_Mrs     ███████                        6.37%
Title_Miss    █████                          4.70%
Embarked_S    ██                             1.79%
IsAlone       █                              1.31%
Embarked_Q    █                              0.97%
Title_Rare    █                              0.82%
```

### 💡 Insights from the Model:
1. **Gender & Titles dominate**: `Sex` and `Title_Mr` account for over **47%** of the model's decision weight, demonstrating that gender and social title were the primary drivers of evacuation survival.
2. **Socioeconomic Status**: `Pclass` (Class 1st, 2nd, 3rd) and `Fare` together represent ~**22.4%** importance, confirming that upper-class passengers had vastly better lifeboat access.
3. **Family Dynamics**: Combining `FamilySize` and `IsAlone` helped the tree distinguish between isolated individuals and families navigating the evacuation together.

---

## 🤖 Model Configuration & Training

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Hyperparameters tuned to balance bias & variance
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=5,
    random_state=42
)

# Training & Validation
model.fit(X_train, y_train)
val_predictions = model.predict(X_val)
acc = accuracy_score(y_val, val_predictions)
print(f"Validation Accuracy: {acc * 100:.2f}%")  # 81.56%
```

---

## 📁 Repository Structure

```text
├── feature_engginering.ipynb   # Main notebook containing full feature engineering & modeling pipeline
├── Main.ipynb                  # Exploratory Data Analysis (EDA) & baseline prototyping
├── submission.csv              # Official predictions formatted for Kaggle competition (418 rows)
├── .gitignore                  # Git ignore rules for checkpoints and virtual environments
└── README.md                   # Comprehensive project documentation
```

---

## 💻 Getting Started

### 1. Prerequisites
Make sure you have Python 3.8+ installed along with Jupyter environment:

```bash
pip install pandas numpy scikit-learn jupyter
```

### 2. Clone the Repository
```bash
git clone git@github.com:FuncSmile/Titanic---Machine-Learning-from-Disaster.git
cd Titanic---Machine-Learning-from-Disaster
```

### 3. Run the Notebook
Launch Jupyter Notebook or Jupyter Lab:
```bash
jupyter notebook feature_engginering.ipynb
```

---

## 📤 Kaggle Submission

To submit predictions directly using Kaggle CLI:

```bash
kaggle competitions submit -c titanic -f submission.csv -m "Random Forest with Feature Engineering (81.56% Val Acc)"
```

---

## 👤 Author

- **GitHub**: [@FuncSmile](https://github.com/FuncSmile)
- **Kaggle**: [@favaaa](https://www.kaggle.com/favaaa)

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
