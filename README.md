# Detección de Enfermedades en Plantas con CNNs

**Curso:** SI3014 — Redes Neuronales y Aprendizaje Profundo  
**Autoras:** Mariana Valderrama Castañeda · Sara López Marín · Alexandra Hurtado David

---
## Pregunta de Investigación y Objetivo

**Pregunta de investigación:**
> ¿Puede una red neuronal convolucional entrenada con imágenes de hojas de plantas distinguir
> entre plantas sanas y enfermas, y en qué medida mejora el desempeño al incorporar
> transfer learning frente a una CNN entrenada desde cero?

**Objetivo:**
Comparar el desempeño de tres arquitecturas CNN de distintas complejidades(baseline, 
regularización aumentada, transfer learning con ResNet18) para la clasificación binaria 
de enfermedades en plantas, evaluando la mejora en F1-score y Recall sobre el conjunto 
de prueba usando el dataset PlantVillage integrando todos los temas centrales desarollados a lo largo del curso. 

## Descripción del Proyecto

Este proyecto entrena y compara tres modelos de clasificación de imágenes basados en CNNs para distinguir automáticamente entre hojas de plantas **sanas** y **enfermas**, usando el dataset **PlantVillage** disponible en Kaggle.

PlantVillage contiene ~20,638 imágenes de hojas de plantas con 15 clases originales. En este proyecto colapsamos todas las clases en clasificación **binaria**: `sana` vs `enferma`.

| Modelo | Descripción |
|--------|-------------|
| **Modelo 1** | CNN construida desde cero (*baseline*) |
| **Modelo 2** | Misma CNN + más augmentación y regularización |
| **Modelo 3** | ResNet18 con Transfer Learning |

---

## Cómo ejecutar el código

### Opción A — Kaggle (recomendado)

El notebook está optimizado para ejecutarse directamente en Kaggle, donde el dataset PlantVillage ya se encuentra disponible sin necesidad de descarga manual.

1. Ve a [kaggle.com](https://www.kaggle.com) e inicia sesión
2. Crea un nuevo notebook o sube `proyectofinalredes.ipynb` desde *File → Import Notebook*
3. En la sección *Add Data*, busca y agrega el dataset `emmarex/plantdisease`
4. Activa la GPU en *Session options → Accelerator → GPU*
5. Ejecuta todas las celdas

### Opción B — Local / Google Colab

#### 1. Clonar el repositorio

```bash
git clone https://github.com/alexahurtado08/ProyectoFinalRedesNeuronales.git
cd ProyectoFinalRedesNeuronales
```

#### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

O con conda:

```bash
conda env create -f environment.yml
conda activate plant-disease
```

#### 3. Configurar el dataset

1. Ve a [kaggle.com](https://www.kaggle.com) → Account → API → *Create New Token*
2. Descarga el archivo `kaggle.json`
3. Súbelo cuando la celda del notebook te lo pida (o colócalo en `~/.kaggle/kaggle.json`)

#### 4. Ejecutar el notebook

```bash
jupyter notebook proyectofinalredes.ipynb
```

O súbelo directamente a **Google Colab** y activa la GPU en *Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU*.

---

## Estructura del Repositorio
ProyectoFinalRedesNeuronales/

│

├── proyectofinalredes.ipynb -----------------------------------             # Notebook principal (EDA + entrenamiento + evaluación)

├── requirements.txt  -----------------------------------                   # Dependencias del proyecto

├── environment.yml ----------------------------------                      # Entorno conda (alternativa)

├── README.md  ----------------------------------                           # Este archivo

│
├── checkpoints/  ---------------------------------                        # Pesos del mejor modelo por cada arquitectura

│   ├── modelo1_baseline_best.pt

│   ├── modelo2_regularizado_best.pt

│   └── modelo3_fase2_best.pt

│
└── figures/   ---------------------------------                          # Visualizaciones generadas

├── eda_distribucion.png

├── eda_ejemplos.png

├── eda_pixeles.png

├── Modelo_1_-CNN_Baseline_curvas.png

├── Modelo_2-CNN_Regularizada_curvas.png

├── Modelo_3-_ResNet18_Transfer_Learning_curvas.png

├── comparacion_modelos.png

└── matrices_confusion.png


---

## Dataset

**Fuente:** [PlantVillage — Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)

- **Total de imágenes:** ~20,638
- **Clases originales:** 15 (combinaciones de especie de planta y enfermedad)
- **Clases utilizadas (colapso binario):**
  - `sana` ← carpetas que contienen `healthy` en el nombre
  - `enferma` ← todas las demás clases
- **Modalidad:** Imágenes RGB de hojas de plantas

| Conjunto      | Proporción | # Imágenes aprox. |
|---------------|------------|-------------------|
| Entrenamiento | 60%        | ~12,383           |
| Validación    | 20%        | ~4,128            |
| Prueba        | 20%        | ~4,128            |

Split con estratificación por clase (`stratify=labels`, `random_state=42`).

---

## Preprocesamiento

**Todos los conjuntos:**
- Redimensionamiento a 224×224 píxeles (RGB)
- Normalización ImageNet: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`

**Solo entrenamiento (augmentación):**
- Flip horizontal y vertical aleatorio
- Rotaciones aleatorias (±30°)
- `ColorJitter` (brillo 0.3, contraste 0.3, saturación 0.2)
- `RandomResizedCrop` con escala [0.8, 1.0]

---

## Arquitecturas

### Modelo 1 — CNN Baseline (from scratch)

Input: 224×224×3


│

├── Bloque Conv 1: Conv2D(32, 3×3) → BN → ReLU → MaxPool(2×2)

├── Bloque Conv 2: Conv2D(64, 3×3) → BN → ReLU → MaxPool(2×2)

├── Bloque Conv 3: Conv2D(128, 3×3) → BN → ReLU → MaxPool(2×2)

├── Bloque Conv 4: Conv2D(256, 3×3) → BN → ReLU → MaxPool(2×2)

│

├── GlobalAveragePooling2D

├── Dense(256) → ReLU → Dropout(0.5)

└── Dense(2) → Softmax    ← {sana, enferma}


### Modelo 2 — CNN con Regularización Aumentada

Input: 224×224×3  (con augmentación en entrenamiento)

│

├── Mismos 4 bloques convolucionales

├── GlobalAveragePooling2D

├── Dense(256) → ReLU → Dropout(0.5)

├── Dense(128) → ReLU → Dropout(0.3)

└── Dense(2) → Softmax


### Modelo 3 — Transfer Learning con ResNet18

Pesos preentrenados (ImageNet) — ResNet18

↓

Backbone ResNet18 (congelado en fase 1)

↓

GlobalAveragePooling

↓

Dense(256) → ReLU → Dropout(0.4)

↓

Dense(2) → Softmax    ← {sana, enferma}

Fine-tuning fase 2: se descongela layer4 de ResNet18


---

## Estrategia de Entrenamiento

| Hiperparámetro | Modelo 1 (Baseline) | Modelo 2 (Aug + Reg) | Modelo 3 (ResNet18 TL) |
|----------------|---------------------|----------------------|------------------------|
| Loss           | CrossEntropyLoss    | CrossEntropyLoss     | CrossEntropyLoss       |
| Optimizer      | Adam                | Adam                 | Adam                   |
| Learning rate  | 1e-3                | 1e-3                 | 1e-4 (fase 1) / 1e-5 (fase 2) |
| Batch size     | 32                  | 32                   | 32                     |
| Épocas máx.    | 30 (early stopping) | 30 (early stopping)  | 10 + 10 (early stopping) |
| LR Scheduler   | ReduceLROnPlateau   | ReduceLROnPlateau    | ReduceLROnPlateau      |
| Dropout        | 0.5                 | 0.5 / 0.3            | 0.4                    |
| Weight decay   | —                   | 1e-4                 | 1e-4                   |

**Hardware:** GPU (Kaggle / Google Colab).  
**Early stopping:** paciencia de 10 épocas sobre `val_loss`.  
**Criterio de selección:** mejor F1-score en validación.

---

## Métricas de Evaluación

| Métrica                  | Justificación |
|--------------------------|---------------|
| F1-score (macro)         | Métrica principal — robusta ante desbalance de clases |
| Accuracy                 | Referencia general |
| Recall (clase enferma)   | Crítico: minimizar falsos negativos (planta enferma clasificada como sana) |
| AUC-ROC                  | Evalúa la capacidad discriminativa del modelo |

---

## Referencias

1. Hughes, D., Salathé, M. (2015). An open access repository of images on plant health. *arXiv:1511.08060*. https://arxiv.org/abs/1511.08060
2. He, K., Zhang, X., Ren, S., Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*. https://arxiv.org/abs/1512.03385
3. Saleem, M.H., Potgieter, J., Arif, K.M. (2019). Plant Disease Detection and Classification by Deep Learning. *Plants, 8*(11), 468. https://doi.org/10.3390/plants8110468
4. Mohanty, S.P., Hughes, D., Salathé, M. (2016). Using Deep Learning for Image-Based Plant Disease Detection. *Frontiers in Plant Science, 7*, 1419.

---

## Declaración de Uso de IA

Durante el desarrollo de este proyecto se utilizaron herramientas de inteligencia artificial generativa, específicamente modelos de lenguaje como **Claude (Anthropic)**, como apoyo en las siguientes actividades:

- Estructuración y redacción de documentación (README, anteproyecto)
- Revisión y depuración de código
- Consultas sobre buenas prácticas en preprocesamiento y arquitecturas CNN
- Apoyo en la redacción de análisis e interpretación de resultados

Todo el contenido generado con asistencia de IA fue revisado, validado y ajustado por el equipo de trabajo.
