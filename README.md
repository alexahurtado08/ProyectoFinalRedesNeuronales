# Anteproyecto Redes Neuronales 

## Problema a resolver

La moniliasis (Monilia roreri) es la enfermedad fúngica más devastadora del cacao (Theobroma cacao) en América Latina. En Colombia, donde el cacao es un cultivo de alto valor económico y social —especialmente en regiones como Antioquia, Huila y Arauca— esta enfermedad puede ocasionar pérdidas de hasta el 80% de la cosecha si no se detecta a tiempo.
El diagnóstico tradicional depende de inspección visual manual por parte de expertos, lo cual es:

Costoso en mano de obra.
Lento para cubrir grandes extensiones de cultivo.
Subjetivo y propenso a errores en etapas tempranas de infección.

## Datos

Desarrollar un modelo de clasificación de imágenes basado en redes neuronales convolucionales (CNN) capaz de distinguir automáticamente entre mazorcas de cacao sanas y mazorcas infectadas con Monilia roreri a partir de fotografías tomadas en campo, bajo condiciones reales de iluminación y oclusión.

Relevancia
Una herramienta de diagnóstico visual automatizada permitiría a agricultores y técnicos agrícolas detectar la enfermedad en etapas tempranas, reducir el uso de fungicidas y optimizar las decisiones de cosecha, contribuyendo a la seguridad alimentaria y a la sostenibilidad de los cultivos de cacao en Colombia. 

## Arquitectura base

Datos
Dataset Principal: CocoaMoniliaDataSet
Dataset de Soporte: RipSetCocoaCNCH12 

Nota: El CocoaMoniliaDataSet es el dataset principal de entrenamiento y evaluación. El RipSetCocoaCNCH12 se usa como referencia para entender la variabilidad visual del fruto y puede integrarse en estrategias de preentrenamiento en dominio.
División: 70% entrenamiento / 15% validación / 15% prueba
   (con estratificación por clase para preservar balance)

Base

La arquitectura base es una CNN construida desde cero (from scratch), sin usar pesos preentrenados. Sirve como línea de referencia (baseline) para comparar el beneficio de arquitecturas más complejas y del transfer learning.

Descripción
Input: imagen RGB 224×224×3
│
├── Bloque Conv 1: Conv2D(32, 3×3) → BN → ReLU → MaxPool(2×2)
├── Bloque Conv 2: Conv2D(64, 3×3) → BN → ReLU → MaxPool(2×2)
├── Bloque Conv 3: Conv2D(128, 3×3) → BN → ReLU → MaxPool(2×2)
├── Bloque Conv 4: Conv2D(256, 3×3) → BN → ReLU → MaxPool(2×2)
│
├── GlobalAveragePooling2D
├── Dense(256) → ReLU → Dropout(0.5)
└── Dense(2) → Softmax   ← {sana, infectada}

Este modelo simple permite establecer un rendimiento de referencia con mínimas suposiciones sobre los datos. Al entrenarlo desde cero con el dataset disponible, es posible cuantificar cuánto mejoran las arquitecturas más avanzadas.


## Arquitectura propuesta según la naturaleza de los datos
Análisis de la Naturaleza de los Datos
Las imágenes del CocoaMoniliaDataSet presentan características que determinan las decisiones de diseño arquitectónico:
<img width="547" height="374" alt="imagen" src="https://github.com/user-attachments/assets/1e9c3838-8312-438a-92ca-ad3640cac7b0" />

Arquitectura Propuesta: 
Dado que la enfermedad de Monilia se manifiesta en regiones específicas de la superficie de la mazorca (manchas, decoloración localizada), se propone incorporar un mecanismo de atención espacial (CBAM — Convolutional Block Attention Module) que permita al modelo enfocarse automáticamente en las zonas patológicas relevantes.

Input: imagen RGB 224×224×3
│
├── Stem: Conv2D(64, 7×7, stride=2) → BN → ReLU → MaxPool
│
├── Bloque Residual 1 + CBAM → 64 filtros
├── Bloque Residual 2 + CBAM → 128 filtros
├── Bloque Residual 3 + CBAM → 256 filtros
├── Bloque Residual 4 + CBAM → 512 filtros
│       ↑
│   Conexiones residuales (skip connections)
│   para preservar gradientes en entrenamiento
│
├── GlobalAveragePooling2D
├── Dense(512) → ReLU → Dropout(0.4)
├── Dense(128) → ReLU → Dropout(0.3)
└── Dense(2) → Softmax

Justificación Técnica

Las conexiones residuales resuelven el problema del desvanecimiento del gradiente y permiten entrenar redes más profundas con datasets pequeños.
El módulo de atención de canal pondera la importancia relativa de cada mapa de características.
El módulo de atención espacial genera un mapa de calor que resalta las zonas con síntomas de Monilia, lo que también mejora la interpretabilidad del modelo.
Esta arquitectura es adecuada cuando las señales diagnósticas son localizadas y texturales, como en este caso.

## Arquitectura con transfer learning

Motivación
El transfer learning permite aprovechar representaciones visuales aprendidas en millones de imágenes (ImageNet) y adaptarlas al dominio específico de mazorcas de cacao, compensando el tamaño limitado del dataset.
Modelo Propuesto: EfficientNetV2-S con Fine-Tuning Progresivo
Se selecciona EfficientNetV2-S por su excelente balance entre precisión y eficiencia computacional, y por su demostrado desempeño en tareas de clasificación de enfermedades agrícolas.

Pesos preentrenados (ImageNet)
        ↓
   Fine-Tuning progresivo
        ↓
  Clasificador adaptado
  {sana / infectada con Monilia}

Métricas
Dado que el contexto agrícola requiere minimizar los falsos negativos (mazorca infectada clasificada como sana), se priorizan:

<img width="483" height="223" alt="imagen" src="https://github.com/user-attachments/assets/2da4a3bf-6895-4702-a974-aeac51fac546" />

## Referencias

CocoaMoniliaDataSet: Alvarado J., Restrepo-Arias J.F., Velásquez D., Branch-Bedoya J.W., Maiza M. (2026). CocoaMoniliaDataSet: A cocoa pod dataset to detect and classify Monilia roreri in real conditions. Data in Brief, 64, 112447. https://doi.org/10.1016/j.dib.2025.112447
RipSetCocoaCNCH12: (2023). RipSetCocoaCNCH12: Labeled Dataset for Ripeness Stage Detection, Semantic and Instance Segmentation of Cocoa Pods. MDPI Data, 8(6), 112. https://doi.org/10.3390/data8060112
CBAM: Woo S. et al. (2018). CBAM: Convolutional Block Attention Module. ECCV 2018.
EfficientNetV2: Tan M., Le Q.V. (2021). EfficientNetV2: Smaller Models and Faster Training. ICML 2021.
Transfer Learning en agricultura: Saleem M.H., Potgieter J., Arif K.M. (2019). Plant Disease Detection and Classification by Deep Learning. Plants, 8(11), 468.

### Autoras:
* Mariana Valderrama Castañeda 
* Sara López Marín
* Alexandra Hurtado David 
