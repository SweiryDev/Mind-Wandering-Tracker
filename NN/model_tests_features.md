Parameters:

sequence_length = 30 * 3  # fps * seconds
max_layer_neurons = 256
epochs=20
batch_size=400

#### Eyes [LEAR, REAR]

Classification Report:
               precision    recall  f1-score   support

         0.0       0.90      0.54      0.67     65629
         1.0       0.17      0.61      0.27     10435

    accuracy                           0.55     76064
   macro avg       0.54      0.58      0.47     76064
weighted avg       0.80      0.55      0.62     76064

#### Eyebrow [LEEAR, REEAR]
To much false positive!

Classification Report:
               precision    recall  f1-score   support

         0.0       0.88      0.52      0.66     65629
         1.0       0.16      0.56      0.25     10435

    accuracy                           0.53     76064
   macro avg       0.52      0.54      0.45     76064
weighted avg       0.78      0.53      0.60     76064

#### Nose Position [NXP, NYP] - DISCARDED
Not confident but has some good true positives

Classification Report:
               precision    recall  f1-score   support

         0.0       0.85      0.63      0.72     65629
         1.0       0.12      0.31      0.17     10435

    accuracy                           0.58     76064
   macro avg       0.48      0.47      0.45     76064
weighted avg       0.75      0.58      0.65     76064

#### Mouth [IMAR, OMAR]
Too much False positive but great true positive catch

Classification Report:
               precision    recall  f1-score   support

         0.0       0.87      0.85      0.86     65629
         1.0       0.17      0.19      0.18     10435

    accuracy                           0.76     76064
   macro avg       0.52      0.52      0.52     76064
weighted avg       0.77      0.76      0.76     76064

#### Face Position [FXP, FYP] - DISCARDED
Not accurate

Classification Report:
               precision    recall  f1-score   support

         0.0       0.86      0.72      0.78     65629
         1.0       0.12      0.25      0.17     10435

    accuracy                           0.65     76064
   macro avg       0.49      0.48      0.47     76064
weighted avg       0.76      0.65      0.70     76064

#### Face Rotation [LFRR, RFRR]
Too much false positive.

Classification Report:
               precision    recall  f1-score   support

         0.0       0.88      0.73      0.79     65629
         1.0       0.17      0.36      0.24     10435

    accuracy                           0.68     76064
   macro avg       0.53      0.54      0.52     76064
weighted avg       0.78      0.68      0.72     76064

#### Mouth Nose [MNAR]
Too much false positive.

Classification Report:
               precision    recall  f1-score   support

         0.0       0.87      0.81      0.84     65629
         1.0       0.17      0.25      0.20     10435

    accuracy                           0.73     76064
   macro avg       0.52      0.53      0.52     76064
weighted avg       0.78      0.73      0.75     76064

#### Mouth Chin [MCAR] - DISCARDED
Not accurate

Classification Report:
               precision    recall  f1-score   support

         0.0       0.85      0.86      0.85     65629
         1.0       0.04      0.03      0.03     10435

    accuracy                           0.75     76064
   macro avg       0.44      0.45      0.44     76064
weighted avg       0.74      0.75      0.74     76064

#### Mouth combination [IMAR, OMAR, MNAR]
Too much FP

Classification Report:
               precision    recall  f1-score   support

         0.0       0.87      0.81      0.84     65629
         1.0       0.18      0.26      0.22     10435

    accuracy                           0.74     76064
   macro avg       0.53      0.54      0.53     76064
weighted avg       0.78      0.74      0.76     76064

#### > 0.2 1.0 f1 score [LEAR, REAR, LEEAR, REEAR, LFRR, RFRR, MNAR]
Too much FP

Classification Report:
               precision    recall  f1-score   support

         0.0       0.90      0.65      0.75     65629
         1.0       0.19      0.54      0.29     10435

    accuracy                           0.63     76064
   macro avg       0.55      0.59      0.52     76064
weighted avg       0.80      0.63      0.69     76064