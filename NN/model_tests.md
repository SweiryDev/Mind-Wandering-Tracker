Tests:

---
#### T1

sequence_length = 30
max_layer_neurons = 256
epochs=2, batch_size=64


```
Classification Report:
               precision    recall  f1-score   support

         0.0       0.71      0.62      0.67     73697
         1.0       0.13      0.19      0.16     22715

    accuracy                           0.52     96412
   macro avg       0.42      0.41      0.41     96412
weighted avg       0.58      0.52      0.55     96412
```

---
#### T2

sequence_length = 30
max_layer_neurons = 512
epochs=2, batch_size=64

```
Classification Report:
               precision    recall  f1-score   support

         0.0       0.76      0.55      0.64     73697
         1.0       0.23      0.44      0.30     22715

    accuracy                           0.52     96412
   macro avg       0.50      0.49      0.47     96412
weighted avg       0.64      0.52      0.56     96412
```
---
#### T3 - good one

sequence_length = 30
max_layer_neurons = 1024
epochs=2, batch_size=64

```
Classification Report:
               precision    recall  f1-score   support

         0.0       0.85      0.59      0.70     73697
         1.0       0.34      0.67      0.45     22715

    accuracy                           0.61     96412
   macro avg       0.60      0.63      0.58     96412
weighted avg       0.73      0.61      0.64     96412

```
---
#### T4

sequence_length = 30
max_layer_neurons = 2048
epochs=2, batch_size=64

```
Classification Report:
               precision    recall  f1-score   support

         0.0       0.75      0.78      0.77     73697
         1.0       0.17      0.14      0.15     22715

    accuracy                           0.63     96412
   macro avg       0.46      0.46      0.46     96412
weighted avg       0.61      0.63      0.62     96412

```
---
#### T5 Overfit ?

sequence_length = 30
max_layer_neurons = 1024
epochs=10, batch_size=64

```
Classification Report:
               precision    recall  f1-score   support

         0.0       0.71      0.64      0.67     73697
         1.0       0.13      0.17      0.14     22715

    accuracy                           0.53     96412
   macro avg       0.42      0.40      0.41     96412
weighted avg       0.57      0.53      0.55     96412

```
---
#### T6 - better (with class weights)

sequence_length = 30
max_layer_neurons = 1024
epochs=2, batch_size=64

```
Classification Report:
               precision    recall  f1-score   support

         0.0       0.77      0.89      0.83     73697
         1.0       0.27      0.13      0.18     22715

    accuracy                           0.71     96412
   macro avg       0.52      0.51      0.50     96412
weighted avg       0.65      0.71      0.67     96412
```
---
#### T7 - More accurate 

sequence_length = 90
max_layer_neurons = 1024
epochs=2, batch_size=64

```
Best F1-Score for Class 1: 0.5988
Found at threshold: 0.35

\n==== Classification Report with Optimal Threshold ====\n
              precision    recall  f1-score   support

         0.0       0.93      0.71      0.80     73637
         1.0       0.47      0.84      0.60     22715

    accuracy                           0.74     96352
   macro avg       0.70      0.77      0.70     96352
weighted avg       0.82      0.74      0.76     96352
```
---