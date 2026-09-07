# Inteligencia Regenerativa

Trabajos del curso de Inteligencia Regenerativa (modelos generativos profundos) — UAO.

## Semana 5 — Arquitectura y dimensionamiento de un CNN-VAE

Punto de partida: [`Semana 5/cnn_vae.ipynb`](Semana%205/cnn_vae.ipynb), un autoencoder variacional convolucional (CNN-VAE) entrenado sobre MNIST con espacio latente de dimensión 2.

Entregable: [`Semana 5/cnn_vae_dibujo.png`](Semana%205/cnn_vae_dibujo.png), el dibujo de la arquitectura (encoder, muestreo/reparametrización, decoder) con el dimensionamiento de cada capa — forma de salida y cantidad de parámetros, verificados contra el `model.summary()` real del notebook.

**Resumen de parámetros**

| Bloque | Capas | Parámetros |
|---|---|---|
| Encoder | Conv2D(32) → Conv2D(64) → Flatten → Dense(16) → z_mean(2) / z_log_var(2) | 69,076 |
| Decoder | Dense(3136) → Reshape(7×7×64) → Conv2DTranspose(64) → Conv2DTranspose(32) → Conv2DTranspose(1) | 65,089 |
| **Total** | | **134,165** |

Dimensión del espacio latente: **2**.

## Examen Práctico 1 — Parte 1 (Autoencoders, Denoising y Stacked Autoencoders)

**Grupo 1:** Sergio Gómez — Valeria Franco

Entregable único: [`Examen Practico 1/Examen_Practico_1_Completo.ipynb`](Examen%20Practico%201/Examen_Practico_1_Completo.ipynb),
un solo notebook ejecutado de principio a fin (62 celdas, sin errores) que cubre los cuatro
criterios de la rúbrica de la Parte 1:

### 1–2. Autoencoder + Reducción de dimensionalidad (CIFAR-10)

1. Carga y preprocesamiento de CIFAR-10 (normalización a [0, 1]).
2. Autoencoder convolucional (encoder Conv2D 32→64→128 con `strides=2`, cuello de botella
   `Dense` a 128 dimensiones; decoder simétrico con `Conv2DTranspose`).
3. Entrenamiento con **MSE** (12 épocas, `batch_size=128`), con curva de pérdida train/val.
4. Visualización de imágenes originales vs. reconstruidas.
5. Extracción del espacio latente (128D) y reducción a 2D con **PCA** y **t-SNE**, coloreado
   por clase (10 clases de CIFAR-10).

| Métrica | Valor |
|---|---|
| MSE final (entrenamiento / validación) | ≈ 0.0068 / ≈ 0.0070 |
| Dimensión del espacio latente | 128 |

Las 10 clases aparecen mezcladas en PCA/t-SNE — esperado, ya que el autoencoder solo optimiza
reconstrucción, no separación por clase (discusión completa en el notebook).

### 3. Denoising de Imágenes (Intel Image Classification)

Dataset **Intel Image Classification** (`puneet6060/intel-image-classification`, vía
`kagglehub`), 6 clases de paisajes (`buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`).

1. Carga de 6,000 imágenes redimensionadas a 64×64 y normalizadas.
2. Ruido gaussiano artificial añadido a las imágenes (`NOISE_FACTOR=0.3`).
3. Autoencoder convolucional (Conv2D + MaxPooling2D en el encoder, Conv2D + UpSampling2D en el
   decoder) entrenado para mapear imagen ruidosa → imagen limpia, con MSE (15 épocas).
4. Evaluación: MSE en test ≈ **0.0166**.
5. Comparación visual: original / con ruido / reconstruida.

### 4. Stacked Autoencoders (extracción de características + clasificación)

Mismo dataset Intel Image Classification, esta vez usando las etiquetas de clase.

1. Carga balanceada (500 imágenes/clase entrenamiento, 150/clase prueba) con etiquetas.
2. **Stacked Autoencoder**: encoder con 3 bloques `Conv2D + MaxPooling2D` apilados → vector
   latente de 256D; decoder simétrico con `Conv2DTranspose`. Entrenado sin supervisión (MSE).
3. Encoder congelado (`trainable = False`) usado como extractor de características fijo.
4. Clasificador (`Dense(128) → Dropout → Dense(6, softmax)`) entrenado sobre esas características
   con las etiquetas de clase.
5. Evaluación: accuracy, matriz de confusión, y visualización del espacio latente por clase
   (PCA y t-SNE).

| Métrica | Valor |
|---|---|
| MSE final Stacked AE (train / val) | ≈ 0.023 / ≈ 0.024 |
| Accuracy del clasificador (test) | ≈ 61% (línea base aleatoria: 16.7%) |

Mayor confusión entre `glacier`, `mountain` y `sea` (paisajes naturales visualmente similares) y
mejor desempeño en `forest` y `street` (matriz de confusión completa en el notebook).

## Guion de exposición

[`Examen Practico 1/GUION_EXPOSICION.md`](Examen%20Practico%201/GUION_EXPOSICION.md) — guion
completo para la presentación (máx. 20 min), con reparto por integrante, explicación celda por
celda de las cuatro secciones del notebook, y respuestas preparadas a preguntas frecuentes del
profesor.
