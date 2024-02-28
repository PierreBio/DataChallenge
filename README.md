# DataChallenge

![logo_data_challenge](https://github.com/PierreBio/DataChallenge/assets/45881846/8e962398-670c-40b8-ae51-4fecbc9fe7f6)

This project is carried out in the context of the Artificial Intelligence Masters of **TelecomParis**.

## Project
This year's challenge is about text classification. For privacy reasons, you are only provided with the embedding learned on the original documents.

### Fair document classification

The task is straightforward: assign the correct category to a text. This is thus a multi-class classification task with 28 classes to choose from.

The most adopted paradigm consists in training a network f:X→Rd
which, from a given document x∈X, extracts a feature vector z∈Rd which synthetizes the relevant caracteristics of doc. The diagnostic phase then consists, from an document x, to predict the label of the document based on the extracted features z. In this data challenge d=768.

We directly provide you the embedding of each text (learned with BERT).

The goal of this competition is to design a solution that is both accurate for predicting the label as well as fair with respect to some sensitive attribute (e.g. gender). Fairness in this context means that the model should not be biased toward a certain minority group present in the data. We explain this paradigm further in the evaluation part.

### Downloading the Data

```
import pickle
import pandas as pd
with open('data-challenge-student.pickle', 'rb') as handle:
    # dat = pickle.load(handle)
    dat = pd.read_pickle(handle)

X = dat['X_train']
Y = dat['Y']
S = dat['S_train']
```

### Evaluation

First of all, the accuracy of the solutions are evaluated according to the Macro F1 metric, The Macro F1 score is simply the arithmetic average of the F1 score for each class.

We will also analyse proposed solutions according to their fairness with respect to the provided sensitive attribute (S). In other words, we want you to design a solution that is not biased towards one group in particular. To be specific, we will use (1-equal opportunity gap) between protected groups. A fair model is a model where this criteria is close to 1.

Overall, your model should satisfy both criteria so the evaluation metric is the average between the macro F1 and the fairness criteria.

The file evaluator.py contains the required functions to compute the final score on which you will be ranked.

### Baseline
Let us use a logistic regression as our naive baseline model. Note that this model does not take into accout the sensitive attribute S

```
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from evaluator import *

# Train the logistic regression
X_train, X_test, Y_train, Y_test, S_train, S_test = train_test_split(X, Y, S, test_size=0.3, random_state=42)
clf = LogisticRegression(random_state=0, max_iter=5000).fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
eval_scores, confusion_matrices_eval = gap_eval_scores(Y_pred, Y_test, S_test, metrics=['TPR'])

final_score = (eval_scores['macro_fscore']+ (1-eval_scores['TPR_GAP']))/2
print(final_score)
```

### Preparing the submission file
Now we are ready to prepare a submission file. In the pickle you have access to some additional test data (X_test, S_test) and you should submit your prediction for Y. Note that with the current model, you do not need S_test but we provide it to you in case you want to use it in a debiasing strategy.

```
# Load the "true" test data
X_test = dat['X_test']
S_test = dat['S_test']
# Classify the provided test data with you classifier
y_test = clf.predict(X_test)
results=pd.DataFrame(y_test, columns= ['score'])

results.to_csv("Data_Challenge_MDI_341.csv", header = None, index = None)
# np.savetxt('y_test_challenge_student.txt', y_test, delimiter=',')
```

## How to setup?

- First, clone the repository:

```
git clone https://github.com/PierreBio/DataChallenge.git
```

- Then go to the root of the project:

```
cd DataChallenge
```

- Create a virtual environment:

```
py -m venv venv
```

- Activate your environment:

```
.\venv\Scripts\activate
```

- Install requirements:

```
pip install -r requirements.txt
```

## How to launch?

- Once the project is setup, you can launch it:

```
py -m src.main
```

## Results

See [our results](docs/RESULTS.md).

## Ressources
