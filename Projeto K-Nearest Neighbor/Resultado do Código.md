## Saída do terminal :
Scores de validação cruzada: [0.95081967 0.96721311 0.95081967 0.98360656 0.98360656 0.95081967 0.95081967 0.96721311 0.96721311 0.91666667]

Acurácia média: 0.958879781420765

### Caso 1: Considerando todas as perguntas do questionário

Acurácia no conjunto de teste: 0.9672

Matriz de Confusão:
$$
\begin{bmatrix}
84 & 2 \\
2  & 34
\end{bmatrix}
$$

Relatório de Classificação:
$$
\begin{array}{c|c|c|c|c}
 & \text{precision} & \text{recall} & \text{f1-score} & \text{support} \\
\hline
\text{False} & 0.98 & 0.98 & 0.98 & 86 \\
\text{True}  & 0.94 & 0.94 & 0.94 & 36 \\
\hline
\text{accuracy} &  &  & 0.97 & 122 \\
\text{macro avg} & 0.96 & 0.96 & 0.96 & 122 \\
\text{weighted avg} & 0.97 & 0.97 & 0.97 & 122 \\
\end{array}
$$

### Caso 2: Considerando apenas de AQ-1 a A-10

Acurácia no conjunto de teste: 0.6639

Matriz de Confusão:
$$
\begin{bmatrix}
65 & 21 \\
20 & 16
\end{bmatrix}
$$

Relatório de Classificação:
$$
\begin{array}{c|c|c|c|c}
 & \text{precision} & \text{recall} & \text{f1-score} & \text{support} \\
\hline
\text{False} & 0.76 & 0.76 & 0.76 & 86 \\
\text{True}  & 0.43 & 0.44 & 0.44 & 36 \\
\hline
\text{accuracy} &  &  & 0.66 & 122 \\
\text{macro avg} & 0.60 & 0.60 & 0.60 & 122 \\
\text{weighted avg} & 0.67 & 0.66 & 0.67 & 122 \\
\end{array}
$$

### Caso 3: Considerando apenas AQ9, AQ6 e AQ5
Acurácia no conjunto de teste: 0.8689

Matriz de Confusão:
$$
\begin{bmatrix}
81 & 5 \\
11 & 25
\end{bmatrix}
$$

Relatório de Classificação:
$$
\begin{array}{c|c|c|c|c}
 & \text{precision} & \text{recall} & \text{f1-score} & \text{support} \\
\hline
\text{False} & 0.88 & 0.94 & 0.91 & 86 \\
\text{True}  & 0.83 & 0.69 & 0.76 & 36 \\
\hline
\text{accuracy} &  &  & 0.87 & 122 \\
\text{macro avg} & 0.86 & 0.82 & 0.83 & 122 \\
\text{weighted avg} & 0.87 & 0.87 & 0.87 & 122 \\
\end{array}
$$

### Caso 4: Considerando apenas de AQ9, AQ6, AQ5, AQ4, AQ3 e AQ1
Acurácia no conjunto de teste: 0.9098

Matriz de Confusão:
$$
\begin{bmatrix}
84 & 2 \\
9 & 27
\end{bmatrix}
$$

Relatório de Classificação:
$$
\begin{array}{c|c|c|c|c}
 & \text{precision} & \text{recall} & \text{f1-score} & \text{support} \\
\hline
\text{False} & 0.90 & 0.98 & 0.94 & 86 \\
\text{True}  & 0.93 & 0.75 & 0.83 & 36 \\
\hline
\text{accuracy} &  &  & 0.91 & 122 \\
\text{macro avg} & 0.92 & 0.86 & 0.88 & 122 \\
\text{weighted avg} & 0.91 & 0.91 & 0.91 & 122 \\
\end{array}
$$



