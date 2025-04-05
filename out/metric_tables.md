# Model Performance Tables

## F1 Scores
| Model                  | Identity Declaration |  Accusation  | Interrogation | Call for Action |   Defense    |   Evidence   | Average |
|------------------------|----------------------|--------------|---------------|-----------------|--------------|--------------|---------|
| bert                   |     81.66(1.87)      | 66.23(1.23)* |  90.23(0.57)* |   78.32(0.73)*  | 43.11(0.74)  | 56.79(1.27)  |  69.39  |
| whisper-audio          |     50.75(2.57)      | 45.39(1.46)  |  59.34(0.64)  |   52.26(1.60)   | 36.17(2.04)  | 39.10(0.70)  |  47.17  |
| whisper-audio-and-text |     84.29(2.38)*     | 65.69(0.99)  |  89.62(0.50)  |   77.26(1.30)   | 46.67(2.22)* | 57.70(0.25)* |  70.20* |

Best score in each column is marked with an asterisk (\*)

## Accuracy Scores
| Model                  | Identity Declaration |  Accusation  | Interrogation | Call for Action |   Defense    |   Evidence   | Average |
|------------------------|----------------------|--------------|---------------|-----------------|--------------|--------------|---------|
| bert                   |     97.78(0.26)      | 89.52(0.86)* |  96.21(0.20)* |   96.65(0.17)*  | 82.01(0.07)  | 92.19(0.34)  |  92.39* |
| whisper-audio          |     94.97(0.07)      | 83.70(1.06)  |  85.54(0.54)  |   93.28(0.36)   | 80.85(2.78)  | 89.67(0.45)  |  88.00  |
| whisper-audio-and-text |     98.19(0.21)*     | 89.08(0.78)  |  95.96(0.20)  |   96.37(0.08)   | 82.15(5.16)* | 92.57(0.29)* |  92.39* |

Best score in each column is marked with an asterisk (\*)