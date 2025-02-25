## Saída do terminal :
Scores de validação cruzada: [0.95081967 0.96721311 0.95081967 0.98360656 0.98360656 0.95081967 0.95081967 0.96721311 0.96721311 0.91666667]

Acurácia média: 0.958879781420765

### Caso 1: Considerando todas as perguntas do questionário

Acurácia no conjunto de teste: 0.9672

Matriz de Confusão:

[84  2]

[ 2 34]

Relatório de Classificação:

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| **False**    | 0.90      | 0.98   | 0.94     | 86      |
| **True**     | 0.93      | 0.75   | 0.83     | 36      |
| **accuracy** |           |        | 0.91     | 122     |
| **macro avg**| 0.92      | 0.86   | 0.88     | 122     |
| **weighted avg** | 0.91   | 0.91   | 0.91     | 122     |

### Caso 2: Considerando apenas de AQ-1 a A-10

Acurácia no conjunto de teste: 0.6639

Matriz de Confusão:

[65 21]

[20 16]


Relatório de Classificação:


|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| **False**    | 0.76      | 0.76   | 0.76     | 86      |
| **True**     | 0.43      | 0.44   | 0.44     | 36      |
| **accuracy** |           |        | 0.66     | 122     |
| **macro avg**| 0.60      | 0.60   | 0.60     | 122     |
| **weighted avg** | 0.67   | 0.66   | 0.67     | 122     |

### Caso 3: Considerando apenas AQ9, AQ6 e AQ5
Acurácia no conjunto de teste: 0.8689

Matriz de Confusão:

[81  5]

[11 25]

Relatório de Classificação:


|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| **False**    | 0.88      | 0.94   | 0.91     | 86      |
| **True**     | 0.83      | 0.69   | 0.76     | 36      |
| **accuracy** |           |        | 0.87     | 122     |
| **macro avg**| 0.86      | 0.82   | 0.83     | 122     |
| **weighted avg** | 0.87   | 0.87   | 0.87     | 122     |


### Caso 4: Considerando apenas de AQ9, AQ6, AQ5, AQ4, AQ3 e AQ2
Acurácia no conjunto de teste: 0.9098

Matriz de Confusão:

[84  2]

[ 9 27]

Relatório de Classificação:


|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| **False**    | 0.90      | 0.98   | 0.94     | 86      |
| **True**     | 0.93      | 0.75   | 0.83     | 36      |
| **accuracy** |           |        | 0.91     | 122     |
| **macro avg**| 0.92      | 0.86   | 0.88     | 122     |
| **weighted avg** | 0.91   | 0.91   | 0.91     | 122     |


