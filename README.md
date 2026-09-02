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
