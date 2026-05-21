# Anteproyecto — Redes Neuronales y Aprendizaje Profundo

**Curso:** SI3014 — Redes Neuronales y Aprendizaje Profundo  
**Autoras:** Mariana Valderrama Castañeda · Sara López Marín · Alexandra Hurtado David

---

## 1. Definición del Problema

La moniliasis (_Monilia roreri_) es la enfermedad fúngica más devastadora del cacao (_Theobroma cacao_) en América Latina. En Colombia —donde el cacao es un cultivo de alto valor económico y social, especialmente en regiones como Antioquia, Huila y Arauca— esta enfermedad puede ocasionar pérdidas de hasta el 80% de la cosecha si no se detecta a tiempo.

El diagnóstico tradicional depende de la inspección visual manual por parte de expertos, lo cual es costoso en mano de obra, lento para cubrir grandes extensiones de cultivo, y subjetivo en etapas tempranas de infección.

**Tipo de tarea:** Clasificación binaria de imágenes (sana vs. infectada).

**Objetivo:** Desarrollar un modelo de clasificación de imágenes basado en redes neuronales convolucionales (CNN) capaz de distinguir automáticamente entre mazorcas de cacao **sanas** y mazorcas **infectadas con Monilia roreri**, a partir de fotografías tomadas en campo bajo condiciones reales de iluminación y oclusión.

Una herramienta de diagnóstico visual automatizada permitiría a agricultores y técnicos agrícolas detectar la enfermedad en etapas tempranas, reducir el uso de fungicidas y optimizar las decisiones de cosecha, contribuyendo a la seguridad alimentaria y la sostenibilidad de los cultivos de cacao en Colombia.

---

## 2. Dataset

**Dataset principal:** CocoaMoniliaDataSet (Alvarado et al., 2026)  
**Dataset de soporte:** RipSetCocoaCNCH12 (2023) — usado como referencia para entender la variabilidad visual del fruto.

### Descripción del CocoaMoniliaDataSet

- **Total de imágenes:** 1,953
- **Clases originales (4):** h0 (sana), m1 (protuberancias), m2 (mancha marrón/oleosa + esporulación), m3 (esporulación avanzada)
- **Clases utilizadas (2):** Las tres clases de Monilia (m1, m2, m3) se agrupan en una sola etiqueta **"infectada"**, resultando en clasificación binaria:
  - `0 — sana (h0)`
  - `1 — infectada (m1 + m2 + m3)`

**Justificación del enfoque binario:** El objetivo es la _detección temprana_ de la enfermedad, no la clasificación de su etapa. Para el agricultor, la decisión relevante es saber si una mazorca está sana o enferma.

| Entrada (X)                                 | Etiqueta (y)           | Modalidad |
| ------------------------------------------- | ---------------------- | --------- |
| Imagen RGB de mazorca de cacao (campo real) | 0: sana / 1: infectada | Imágenes  |

---

## 3. Estrategia de División del Dataset

| Conjunto      | Proporción | # Imágenes aprox. |
| ------------- | ---------- | ----------------- |
| Entrenamiento | 60%        | ~1,171            |
| Validación    | 20%        | ~390              |
| Prueba        | 20%        | ~390              |

- **Estratificación por clase:** sí, para mantener la proporción de sanas vs. infectadas en cada split.
- **Justificación:** Con 1,953 imágenes, un split 60/20/20 garantiza suficientes muestras para entrenamiento sin sacrificar la capacidad de evaluación. La estratificación es necesaria dado que las clases colapsadas podrían presentar leve desbalance.

---

## 4. Preprocesamiento de Datos

**Operaciones comunes (entrenamiento, validación y prueba):**

- Redimensionamiento a 224×224 píxeles (RGB)
- Normalización con media y desviación estándar de ImageNet: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`

**Augmentación (solo entrenamiento):**

- Rotaciones aleatorias (±30°)
- Flip horizontal y vertical aleatorio
- Variación de brillo y contraste (`ColorJitter`)
- `RandomResizedCrop` con escala [0.8, 1.0]

**Justificación:** El dataset es relativamente pequeño (~1,953 imágenes). La augmentación es clave para mejorar la generalización y simular las variaciones naturales de iluminación y posición en campo.

---

## 5. Arquitecturas Propuestas

Se comparan tres modelos con complejidad creciente, siguiendo las temáticas del curso.

### Modelo 1 — CNN Baseline (from scratch)

Sirve como línea de referencia para cuantificar el beneficio de arquitecturas más complejas.

```
Input: 224×224×3
│
├── Bloque Conv 1: Conv2D(32, 3×3) → BN → ReLU → MaxPool(2×2)
├── Bloque Conv 2: Conv2D(64, 3×3) → BN → ReLU → MaxPool(2×2)
├── Bloque Conv 3: Conv2D(128, 3×3) → BN → ReLU → MaxPool(2×2)
├── Bloque Conv 4: Conv2D(256, 3×3) → BN → ReLU → MaxPool(2×2)
│
├── GlobalAveragePooling2D
├── Dense(256) → ReLU → Dropout(0.5)
└── Dense(2) → Softmax    ← {sana, infectada}
```

### Modelo 2 — CNN con Data Augmentation y Regularización

Mismo backbone que el Modelo 1, con augmentación agresiva y mayor regularización. Permite estudiar el efecto de Dropout, Weight Decay y Batch Normalization de forma aislada.

```
Input: 224×224×3  (con augmentación en entrenamiento)
│
├── Mismos 4 bloques convolucionales
├── GlobalAveragePooling2D
├── Dense(256) → ReLU → Dropout(0.5)
├── Dense(128) → ReLU → Dropout(0.3)
└── Dense(2) → Softmax
```

### Modelo 3 — Transfer Learning con ResNet18

Aprovecha representaciones preentrenadas en ImageNet adaptadas al dominio de mazorcas de cacao. Fase 1: backbone congelado, se entrena solo el clasificador. Fase 2: fine-tuning de las últimas capas.

```
Pesos preentrenados (ImageNet) — ResNet18
        ↓
  Backbone ResNet18 (congelado en fase 1)
        ↓
  GlobalAveragePooling
        ↓
  Dense(256) → ReLU → Dropout(0.4)
        ↓
  Dense(2) → Softmax    ← {sana, infectada}

Fine-tuning fase 2: se descongela layer4 de ResNet18
```

**Justificación:** Con ~1,953 imágenes, entrenar una red profunda desde cero es propenso a overfitting. ResNet18 ilustra el concepto de skip connections del curso y es la arquitectura de transfer learning más accesible pedagógicamente.

---

## 6. Estrategia de Entrenamiento

| Hiperparámetro | Modelo 1 (Baseline) | Modelo 2 (Aug + Reg) | Modelo 3 (ResNet18 TL)        |
| -------------- | ------------------- | -------------------- | ----------------------------- |
| Loss           | CrossEntropyLoss    | CrossEntropyLoss     | CrossEntropyLoss              |
| Optimizer      | Adam                | Adam                 | Adam                          |
| Learning rate  | 1e-3                | 1e-3                 | 1e-4 (fase 1) / 1e-5 (fase 2) |
| Batch size     | 32                  | 32                   | 32                            |
| Épocas         | 50                  | 50                   | 20 (fase 1) + 20 (fase 2)     |
| LR Scheduler   | ReduceLROnPlateau   | ReduceLROnPlateau    | ReduceLROnPlateau             |
| Dropout        | 0.5                 | 0.5 / 0.3            | 0.4                           |
| Weight decay   | —                   | 1e-4                 | 1e-4                          |

**Hardware previsto:** GPU (Google Colab o similar).

---

## 7. Estrategia de Validación

| Métrica                  | Justificación                                         |
| ------------------------ | ----------------------------------------------------- |
| F1-score (macro)         | Métrica principal — robusta ante desbalance de clases |
| Accuracy                 | Referencia general                                    |
| Recall (clase infectada) | Crítico: minimizar falsos negativos                   |
| AUC-ROC                  | Evalúa la capacidad discriminativa del modelo         |

- **Early stopping:** paciencia de 10 épocas sobre `val_loss`
- **Criterio de selección:** mejor F1-score en validación
- Se guardan checkpoints del mejor modelo en cada época

---

## 8. EDA Inicial

- Distribución de clases (sana vs. infectada) antes y después del colapso de etiquetas
- Visualización de ejemplos por clase (al menos 2 imágenes por categoría)
- Histogramas de distribución de píxeles por canal (R, G, B)
- Análisis de tamaños originales de imágenes antes del redimensionamiento

> Se incluirán al menos 2 visualizaciones: distribución de clases y ejemplos representativos de imágenes sanas e infectadas.

---

## 9. Pregunta de Investigación y Objetivo

**Pregunta de investigación:**

> ¿Puede una red neuronal convolucional entrenada con datos de campo distinguir mazorcas de cacao sanas de infectadas con _Monilia roreri_, y en qué medida mejora el desempeño al incorporar transfer learning frente a una CNN entrenada desde cero?

**Objetivo:** Comparar el desempeño de tres arquitecturas CNN de complejidad creciente (baseline, regularización aumentada, transfer learning con ResNet18) para la detección binaria de moniliasis en mazorcas de cacao, evaluando la mejora en F1-score y Recall sobre el conjunto de prueba.

---

## 10. Referencias

1. Alvarado, J., Restrepo-Arias, J.F., Velásquez, D., Branch-Bedoya, J.W., Maiza, M. (2026). CocoaMoniliaDataSet: A cocoa pod dataset to detect and classify Monilia roreri in real conditions. _Data in Brief, 64_, 112447. https://doi.org/10.1016/j.dib.2025.112447
2. RipSetCocoaCNCH12 (2023). Labeled Dataset for Ripeness Stage Detection, Semantic and Instance Segmentation of Cocoa Pods. _MDPI Data, 8_(6), 112. https://doi.org/10.3390/data8060112
3. He, K., Zhang, X., Ren, S., Sun, J. (2016). Deep Residual Learning for Image Recognition. _CVPR 2016_. https://arxiv.org/abs/1512.03385
4. Saleem, M.H., Potgieter, J., Arif, K.M. (2019). Plant Disease Detection and Classification by Deep Learning. _Plants, 8_(11), 468. https://doi.org/10.3390/plants8110468
5. Goodfellow, I., Bengio, Y., Courville, A. (2016). _Deep Learning_. MIT Press. https://www.deeplearningbook.org
