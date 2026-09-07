# Guion de exposición — Examen Práctico 1 (Parte 1)

**Grupo 1: Sergio Gómez — Valeria Franco**
**Tiempo máximo: 20 minutos** (penalización de 0.1 por cada minuto adicional)

Reparto sugerido (ajusten según cuánto domine cada quien su parte, lo importante es que **ambos
puedan responder preguntas de cualquier sección**, no solo de la que expusieron):

| Bloque | Tiempo aprox. | Expone |
|---|---|---|
| Introducción | 1 min | Sergio |
| Parte 1 — Autoencoder + reconstrucción (CIFAR-10) | 4 min | Sergio |
| Parte 1 — PCA / t-SNE del espacio latente | 3 min | Sergio |
| Denoising de imágenes (Intel dataset) | 5 min | Valeria |
| Stacked Autoencoders + clasificación | 5 min | Valeria |
| Conclusiones y cierre | 2 min | Ambos |

Total: 20 min. Practiquen con cronómetro — la rúbrica descuenta 0.1 por cada minuto extra.

---

## 0. Introducción (Sergio) — 1 min

> "Buenas [tardes/días], somos Sergio Gómez y Valeria Franco, Grupo 1. Vamos a presentar la Parte
> 1 del examen práctico, que cubre cuatro cosas: un autoencoder convolucional para reconstrucción
> de imágenes, reducción de dimensionalidad de su espacio latente con PCA y t-SNE, un autoencoder
> para eliminar ruido de imágenes (denoising), y un stacked autoencoder que usamos para extraer
> características y clasificar imágenes. Todo está en un solo notebook,
> `Examen_Practico_1_Completo.ipynb`, corrido de principio a fin en Google Colab."

---

## 1. Autoencoder convolucional — Reconstrucción de imágenes (Sergio) — 4 min

**Notebook:** `Examen_Practico_1_Completo.ipynb`, secciones 1 a 5 (bloque CIFAR-10).

### Qué decir, celda por celda:

**Celda de imports:**
> "Usamos TensorFlow/Keras para construir y entrenar la red, y scikit-learn más adelante para
> PCA y t-SNE. Fijamos semillas aleatorias (`np.random.seed(42)`) para que los resultados sean
> reproducibles."

**Carga de CIFAR-10:**
> "Trabajamos con CIFAR-10, un dataset estándar de 60,000 imágenes a color de 32x32 píxeles, en
> 10 clases: avión, automóvil, pájaro, gato, ciervo, perro, rana, caballo, barco y camión. Se
> divide en 50,000 imágenes de entrenamiento y 10,000 de prueba. Normalizamos los píxeles al
> rango [0,1] dividiendo entre 255, porque la última capa del decoder usa activación sigmoide,
> que también produce valores en ese rango — así la comparación entre imagen original y
> reconstruida es directa."

**Arquitectura — Encoder (explica esto mostrando el `encoder.summary()`):**
> "El encoder tiene tres capas convolucionales con `strides=2`, que van reduciendo el tamaño
> espacial de la imagen: de 32x32 baja a 16x16, luego a 8x8, y finalmente a 4x4, mientras la
> profundidad (número de canales/filtros) aumenta: 32, 64 y 128. Esto es la idea central de un
> encoder convolucional: en lugar de usar pooling, usamos convoluciones con stride para reducir
> la resolución y que la red aprenda cómo comprimir la información. Al final aplanamos ese
> volumen 4x4x128 y lo proyectamos con una capa `Dense` a un vector de 128 números — ese es
> nuestro **espacio latente**."

**Arquitectura — Decoder:**
> "El decoder hace exactamente lo inverso: parte del vector de 128 dimensiones, lo expande con
> una capa `Dense` al tamaño 4x4x128, y usa `Conv2DTranspose` (convolución transpuesta) para ir
> aumentando la resolución hasta volver a 32x32x3. La última capa usa activación sigmoide para
> que la salida esté en [0,1], igual que la entrada normalizada."

**Por qué encoder y decoder son modelos separados (posible pregunta del profesor):**
> "Los construimos como dos modelos de Keras independientes (`encoder` y `decoder`) y luego los
> encadenamos en un tercer modelo, `autoencoder`. La razón es que más adelante, en la sección 6,
> necesitamos usar *solo* el encoder para extraer las representaciones latentes de las imágenes,
> sin pasar por el decoder."

**Entrenamiento:**
> "Entrenamos con **MSE (error cuadrático medio)** entre la imagen original y la reconstruida —
> tal como pide el enunciado — usando el optimizador Adam, 12 épocas y batch size de 128. Usamos
> el conjunto de prueba como validación para vigilar que no haya sobreajuste."

**Curva de pérdida (mostrar la gráfica):**
> "La pérdida de entrenamiento baja de 0.024 a 0.007, y la de validación se mantiene muy cerca de
> la de entrenamiento durante todo el proceso — eso nos dice que el modelo generaliza bien, no
> está memorizando el set de entrenamiento."

**Reconstrucciones (mostrar la comparación original vs. reconstruida):**
> "Aquí comparamos 8 imágenes del conjunto de prueba —que el modelo nunca vio durante el
> entrenamiento— contra su reconstrucción. Se reconoce claramente la forma, el color dominante y
> la composición general de cada imagen, pero se pierden detalles finos como texturas o bordes
> nítidos. Esto es esperado: estamos comprimiendo 3,072 valores de entrada (32×32×3) a solo 128
> en el cuello de botella, es una compresión de más de 24 veces."

---

## 2. PCA y t-SNE del espacio latente (Sergio) — 3 min

**Extracción del espacio latente:**
> "Una vez entrenado el autoencoder completo, usamos únicamente el `encoder` para obtener el
> vector latente de 128 dimensiones de cada imagen. Trabajamos con una muestra de 3,000 imágenes
> del conjunto de prueba por costo computacional, tomadas de forma aleatoria uniforme para no
> sesgar hacia ninguna clase."

**PCA:**
> "Aplicamos PCA (Análisis de Componentes Principales) para reducir esas 128 dimensiones a solo
> 2, y así poder graficarlas. PCA es una proyección **lineal** que busca las direcciones de
> mayor varianza en los datos. En nuestro caso, las dos primeras componentes principales explican
> en conjunto solo cerca del 40% de la varianza total — es decir, se pierde información
> considerable al bajar de 128 a 2 dimensiones, lo cual es normal."

**Gráfico de dispersión PCA (mostrar):**
> "En este scatter plot, cada punto es una imagen, coloreada según su clase real. Se ve que las
> 10 clases están bastante mezcladas — no hay fronteras limpias entre ellas."

**t-SNE:**
> "También aplicamos t-SNE, que a diferencia de PCA es una técnica **no lineal**, y en vez de
> preservar varianza global, busca preservar las relaciones de vecindad locales: puntos que están
> cerca en las 128 dimensiones deben quedar cerca también en 2D. Es más costosa computacionalmente
> pero suele mostrar agrupamientos más definidos."

**Por qué las clases se ven mezcladas (pregunta muy probable del profesor):**
> "Esto es un resultado *esperado y coherente*, no un error. El autoencoder se entrenó
> únicamente para minimizar el error de reconstrucción, es decir, para que la imagen de salida se
> parezca a la de entrada — **nunca vio las etiquetas de clase durante el entrenamiento**. No hay
> ninguna razón matemática para que su espacio latente organice las imágenes por categoría
> semántica. Para lograr una separación clara por clase se necesitaría señal de supervisión
> adicional — que es justo lo que hacemos más adelante en la sección de Stacked Autoencoders,
> donde sí usamos las etiquetas para entrenar un clasificador."

---

## 3. Denoising de Imágenes (Valeria) — 5 min

**Notebook:** `Examen_Practico_1_Completo.ipynb`, sección "Punto 2 — Denoising".

**Dataset (kagglehub):**
> "Para esta parte y la siguiente cambiamos de dataset: usamos **Intel Image Classification**,
> que tiene fotos de paisajes en 6 categorías: montaña, calle, edificios, mar, bosque y glaciar.
> Lo descargamos con la librería `kagglehub`, que en Colab usa una caché para que sea rápido.
> Buscamos automáticamente la carpeta `seg_train/seg_train` dentro del dataset descargado."

**Carga y preprocesamiento:**
> "Cargamos 6,000 imágenes, las redimensionamos a 64x64 píxeles y las normalizamos a [0,1], igual
> que en la Parte 1. Aquí, a diferencia de la Parte 1, **no necesitamos las etiquetas de clase**,
> porque el objetivo de un autoencoder de denoising es limpiar ruido, no clasificar."

**Ruido artificial (explica la función `add_noise`):**
> "Como no tenemos pares reales de imagen-ruidosa/imagen-limpia, generamos el ruido nosotros
> mismos: le sumamos a cada imagen ruido gaussiano aleatorio (`NOISE_FACTOR=0.3`, es decir,
> ruido con desviación estándar 0.3) y recortamos los valores para que sigan en el rango [0,1]
> con `np.clip`. Así construimos el par de entrenamiento: entrada = imagen ruidosa, salida
> objetivo = imagen original limpia."

**Arquitectura del autoencoder de denoising:**
> "Es una arquitectura parecida a la de la Parte 1, pero aquí usamos `MaxPooling2D` en vez de
> `strides=2` para reducir resolución en el encoder, y `UpSampling2D` en vez de
> `Conv2DTranspose` en el decoder — son dos formas equivalentes de hacer lo mismo, downsampling
> y upsampling. El encoder tiene tres bloques Conv2D+MaxPooling (32, 64, 128 filtros) y el decoder
> los invierte simétricamente hasta volver a 64x64x3 con activación sigmoide."

**Entrenamiento:**
> "La clave del denoising está en el `fit`: `autoencoder.fit(x_train_noisy, x_train, ...)` — la
> entrada es la imagen **con ruido** pero la salida objetivo (`y`) es la imagen **limpia**. Así el
> modelo aprende a mapear de 'ruidosa' a 'limpia', no a reconstruirse a sí misma. Entrenamos 15
> épocas con MSE."

**Resultado (mostrar curva de pérdida y comparación visual):**
> "El MSE final en el conjunto de prueba fue de aproximadamente 0.016. En la comparación visual de
> tres filas —original, con ruido, reconstruida— se ve que el modelo logra eliminar buena parte
> del ruido gaussiano y recuperar una imagen mucho más cercana a la original, aunque con algo de
> suavizado (pérdida de detalle fino), que es un efecto típico de los autoencoders de denoising
> entrenados con MSE."

**Por qué MSE y no otra pérdida (posible pregunta):**
> "MSE penaliza directamente la diferencia píxel a píxel entre la reconstrucción y el objetivo
> limpio, lo cual es exactamente lo que buscamos: minimizar cuánto se parece la imagen reconstruida
> a la versión sin ruido."

---

## 4. Stacked Autoencoders + Clasificación (Valeria) — 5 min

**Notebook:** `Examen_Practico_1_Completo.ipynb`, sección "Punto 3 — Stacked Autoencoders".

**Qué es un Stacked Autoencoder (explica el concepto antes del código):**
> "Un stacked autoencoder es, en esencia, un autoencoder con varios bloques de codificación
> 'apilados' uno tras otro, cada uno aprendiendo una representación cada vez más abstracta y
> compacta antes de llegar al vector latente final. La idea es que, al apilar más capas, el
> modelo puede aprender características más ricas y jerárquicas que un autoencoder de una sola
> capa."

**Reutilización del dataset con etiquetas:**
> "Reutilizamos el mismo dataset Intel Image Classification del denoising, pero esta vez **sí
> cargamos las etiquetas de clase**, porque el objetivo final es clasificar. Cargamos un
> conjunto balanceado: 500 imágenes por clase para entrenar y 150 por clase para probar (usando
> la partición oficial `seg_test` del dataset, imágenes que el modelo nunca vio)."

**Arquitectura — Encoder apilado:**
> "El encoder apila tres bloques `Conv2D + MaxPooling2D` (64x64 → 32x32 → 16x16 → 8x8, con 32,
> 64 y 128 filtros respectivamente), y termina en una capa `Dense` que produce un vector latente
> de 256 dimensiones. El decoder es el espejo, usando `Conv2DTranspose` para reconstruir hasta
> 64x64x3."

**Entrenamiento no supervisado (fase 1):**
> "Primero entrenamos el stacked autoencoder completo (encoder + decoder) para reconstrucción,
> igual que en la Parte 1, solo con MSE y **sin usar las etiquetas**. Esto le enseña al encoder a
> comprimir bien la información visual de las imágenes, de forma completamente no supervisada."

**Congelar el encoder y agregar clasificador (fase 2, el paso clave):**
> "Una vez entrenado, hacemos `encoder.trainable = False` — esto 'congela' los pesos del encoder,
> es decir, ya no se van a actualizar más. Encima de ese encoder congelado, agregamos un
> clasificador pequeño: una capa `Dense(128)` con `Dropout` para evitar sobreajuste, y una capa
> final `Dense(6, softmax)` que da la probabilidad de cada una de las 6 clases. Solo entrenamos
> este clasificador pequeño — con las etiquetas —, usando como entrada las características que ya
> extrae el encoder congelado."

**Por qué congelar el encoder (pregunta muy probable):**
> "Congelar el encoder nos permite evaluar honestamente si las características aprendidas de
> forma no supervisada —sin haber visto nunca una etiqueta— ya son útiles para distinguir clases.
> Si solo entrenáramos todo junto desde cero con las etiquetas, no estaríamos demostrando el valor
> del stacked autoencoder como extractor de características, sino simplemente entrenando una red
> convolucional supervisada normal."

**Resultados — Accuracy (mostrar la gráfica y el número):**
> "El clasificador alcanza una accuracy de prueba de aproximadamente 61 a 63%. Para ponerlo en
> contexto: con 6 clases, adivinar al azar da en promedio 16.7% de acierto. O sea que las
> características del stacked autoencoder, aprendidas sin ninguna supervisión, permiten clasificar
> muy por encima del azar."

**Matriz de confusión (mostrarla):**
> "La matriz de confusión muestra que las clases con apariencia muy distintiva, como 'forest' y
> 'street', se clasifican mejor —por ejemplo, forest acierta en 130 de 150 casos—, mientras que
> 'glacier', 'mountain' y 'sea' se confunden más entre sí. Tiene sentido: son paisajes naturales
> que comparten colores y texturas parecidas —cielo, agua, roca, nieve—, mientras que una calle
> o un bosque tienen una apariencia visual mucho más particular."

**PCA / t-SNE del espacio latente del stacked autoencoder:**
> "Repetimos la misma visualización de la Parte 1: proyectamos el espacio latente de 256
> dimensiones a 2D. Aquí sí se nota más agrupamiento por clase que en la Parte 1 con CIFAR-10 —
> por ejemplo, 'forest' y 'mountain' tienden a ocupar zonas distintas del plano—, aunque sigue sin
> ser una separación perfecta, coherente con la accuracy de ~62% y la matriz de confusión."

---

## 5. Conclusiones (Ambos) — 2 min

> "En resumen, mostramos cuatro aplicaciones de autoencoders: reconstrucción básica, reducción de
> dimensionalidad de su espacio latente, eliminación de ruido, y extracción de características no
> supervisadas para clasificación con un stacked autoencoder. Un patrón que se repite en los
> cuatro es que **la arquitectura encoder-decoder es la misma idea de fondo** —comprimir y luego
> reconstruir—, y lo que cambia es el objetivo de entrenamiento: reconstruir la misma imagen,
> limpiar ruido, o servir de base para un clasificador. Quedamos atentos a sus preguntas."

---

## Preguntas frecuentes que podría hacer el profesor (prepárense ambos para responderlas)

1. **¿Por qué usaron 128 (o 256) dimensiones en el espacio latente y no otro número?**
   Es un hiperparámetro que balancea compresión vs. calidad de reconstrucción; lo elegimos como
   un punto intermedio razonable dado el tamaño de imagen (32x32 y 64x64) — se puede ajustar y
   afecta directamente el nivel de detalle que el modelo puede conservar.

2. **¿Qué diferencia hay entre el autoencoder de la Parte 1 y el Stacked Autoencoder?**
   Son la misma idea (encoder-decoder entrenado con MSE), pero el Stacked Autoencoder tiene el
   objetivo adicional de servir como extractor de características para un clasificador
   posterior, y aquí lo llamamos "stacked" por tener varios bloques convolucionales apilados en
   el encoder.

3. **¿Por qué la separación de clases en PCA/t-SNE no es perfecta?**
   Porque el autoencoder (en ambos casos) se entrena para minimizar error de reconstrucción, no
   para maximizar separación entre clases — cualquier separación que aparece es un efecto
   secundario, no el objetivo directo de entrenamiento.

4. **¿Cómo generaron el ruido en el denoising?**
   Ruido gaussiano aditivo (`NOISE_FACTOR=0.3`) sobre la imagen normalizada, recortado a [0,1]
   con `np.clip` para mantener valores válidos de píxel.

5. **¿Por qué congelaron el encoder antes de clasificar?**
   Para medir de forma aislada la calidad de las características aprendidas sin supervisión,
   sin que el gradiente del clasificador las modifique — es la forma estándar de evaluar
   representaciones no supervisadas ("linear/shallow probing").

6. **¿Qué pasaría si entrenaran más épocas?**
   La pérdida de reconstrucción probablemente seguiría bajando un poco más, y la accuracy del
   clasificador también podría subir algo, pero ya se observa una curva que se está aplanando
   (rendimientos decrecientes), especialmente en el autoencoder de CIFAR-10 y en el de denoising.

7. **¿Usaron GPU?**
   El notebook corre en Google Colab, aprovechando GPU cuando está disponible, aunque el
   código funciona igual en CPU (solo más lento).
