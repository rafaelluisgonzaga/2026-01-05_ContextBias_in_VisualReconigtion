
![Context Bias in Visual Recognition](cover/IMAGE.png)
# Context Bias in Visual Recognition
## Human-in-the-Loop Learning with Controlled Semantics

Este projeto investiga **como modelos de Machine Learning aprendem (ou falham em aprender) conceitos visuais**, demonstrando de forma prática o impacto de:
- ausência de semântica,
- semântica contextual leve,
- vazamento de informação (label leakage),
- e feedback humano no loop de aprendizado.

O foco **não é maximizar acurácia**, mas **compreender o comportamento do modelo**, seus erros e atalhos.

---

## 🎯 Objetivo

Demonstrar que:

1. Features visuais simples (ex: brilho) são insuficientes para discriminar cidades.
2. Semântica externa **não-identificadora** melhora o desempenho e a estrutura do erro.
3. Variáveis altamente identificadoras produzem resultados ilusórios (leakage).
4. Feedback humano pode ser usado para **ajuste qualitativo**, não apenas métrico.

---

## 📂 Estrutura do Projeto

```
.
├── data
│   ├── raw/                 # imagens organizadas por cidade
│   ├── processed/
│   │   ├── dataset.csv
│   │   └── feedback.csv
├── src
│   ├── preprocess.py        # extração de features
│   ├── train.py             # treino do modelo
│   ├── evaluate.py          # avaliação Top-K
│   └── app.py               # interface human-in-the-loop
├── requirements.txt
└── README.md
```

---

## 🧠 Dados

- 8 cidades europeias
- 20 imagens por cidade
- Total: **160 imagens**
- Imagens reais, sem data augmentation

---

## 🔧 Features Utilizadas

### Baseline
- `brightness`: brilho médio da imagem

Resultado:
- Acurácia ≈ **0.06**
- Próximo ao acaso (1/8 ≈ 12.5%)

---

### Modelo com Semântica Leve (sem vazamento)
- `brightness`
- `region` (macro-região europeia)

Resultado:
- Acurácia Top-1 ≈ **0.27**
- Top-3 Accuracy = **1.00**

> Embora o modelo erre a cidade exata, a cidade correta aparece consistentemente entre as três mais prováveis.

---

## 📊 Avaliação Correta: Top-K Accuracy

Em problemas com múltiplas classes, Top-1 é excessivamente rígido.

Este projeto utiliza:
- **Top-1** → acerto exato
- **Top-3** → aproximação semântica
- **Top-5** → contexto correto

Resultado observado:
```
Top-1 Accuracy: 0.27
Top-3 Accuracy: 1.00
Top-5 Accuracy: 1.00
```

---

## 👤 Human-in-the-Loop

O projeto inclui uma interface interativa onde:

1. O modelo apresenta Top-3 hipóteses
2. O humano escolhe a cidade correta
3. O feedback é salvo explicitamente

Arquivo gerado:
```
data/processed/feedback.csv
```

Cada linha contém:
```
image_path, chosen_city, true_city, correct
```

---

## 🔁 Uso do Feedback Humano

O feedback humano **não sobrescreve rótulos** nem treina o modelo online.

Ele é utilizado para:
- análise qualitativa dos erros,
- identificação de ambiguidades recorrentes,
- e possível incorporação futura via reponderação.

> **“O feedback humano é utilizado para análise qualitativa e pode ser incorporado ao treinamento via reponderação de amostras, reforçando padrões onde o modelo apresenta maior incerteza.”**

Resultados observados:
- Acurácia global permanece estável ou pode cair levemente
- O modelo se torna mais calibrado e menos confiante em erros

---

## 🧠 Conclusões

- Acurácia isolada não reflete aprendizado real.
- Semântica externa ajuda apenas quando controlada.
- Vazamento de identidade gera métricas enganosas.
- Top-K revela estrutura semântica aprendida.
- Feedback humano melhora qualidade interpretativa.

---

## ✅ Estado Final

O projeto é **intencionalmente encerrado neste ponto**.

Avançar para CNNs ou embeddings pré-treinados alteraria a proposta central:
> compreender comportamento, não maximizar performance.

---

## 📌 Observação Final

Este projeto demonstra maturidade em Machine Learning ao:
- evitar overfitting artificial,
- priorizar interpretabilidade,
- fechar o ciclo completo: dados → modelo → erro → humano → ajuste.
"""