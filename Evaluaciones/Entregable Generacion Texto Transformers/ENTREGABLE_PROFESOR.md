# Entregable — Generación de Texto con Transformers

**UNIVERSIDAD AUTÓNOMA DE OCCIDENTE — Inteligencia Artificial Generativa 2026-2S**
**Integrantes:** Sergio Gómez — Valeria Franco

Los tres notebooks están ejecutados de principio a fin en Google Colab con GPU T4 y guardados con
todas sus salidas (tablas, gráficas y texto generado).

## Enlaces

**Notebook completo (Puntos 1, 2 y 3):** [Abrir en Colab](https://colab.research.google.com/github/sergiogomezqui-bit/inteligencia-regenerativa/blob/master/Evaluaciones/Entregable%20Generacion%20Texto%20Transformers/Entregable_Completo_Puntos_1_2_3.ipynb) · [`Entregable_Completo_Puntos_1_2_3.ipynb`](Entregable_Completo_Puntos_1_2_3.ipynb)

Cada punto por separado:

| Punto | Abrir en Colab (desde GitHub) | Notebook en el repositorio |
|---|---|---|
| 1. Char-Llama | [Abrir en Colab](https://colab.research.google.com/github/sergiogomezqui-bit/inteligencia-regenerativa/blob/master/Evaluaciones/Entregable%20Generacion%20Texto%20Transformers/Punto1_Char_Llama_Quijote.ipynb) | [`Punto1_Char_Llama_Quijote.ipynb`](Punto1_Char_Llama_Quijote.ipynb) |
| 2. Word-Llama | [Abrir en Colab](https://colab.research.google.com/github/sergiogomezqui-bit/inteligencia-regenerativa/blob/master/Evaluaciones/Entregable%20Generacion%20Texto%20Transformers/Punto2_Word_Llama_Quijote.ipynb) | [`Punto2_Word_Llama_Quijote.ipynb`](Punto2_Word_Llama_Quijote.ipynb) |
| 3. RAG con Ollama | [Abrir en Colab](https://colab.research.google.com/github/sergiogomezqui-bit/inteligencia-regenerativa/blob/master/Evaluaciones/Entregable%20Generacion%20Texto%20Transformers/Punto3_RAG_Ollama.ipynb) | [`Punto3_RAG_Ollama.ipynb`](Punto3_RAG_Ollama.ipynb) |

Para volver a ejecutarlos: *Entorno de ejecución → Cambiar tipo de entorno → GPU T4* y luego
*Ejecutar todas*. Tiempos aproximados en T4: Punto 1 ≈ 35 min, Punto 2 ≈ 8 min, Punto 3 ≈ 8 min
(incluye instalar Ollama y descargar los modelos).

## Qué se hizo

1. **Char-Llama (Punto 1).** Se reemplazó Shakespeare por el texto completo de *Don Quijote de la
   Mancha* (Proyecto Gutenberg #2000, ~2 M de caracteres) y el decodificador clásico por una
   arquitectura tipo **Llama** programada desde cero en Keras: RMSNorm pre-norm, RoPE, SwiGLU,
   capas sin bias y atención causal. 6 bloques, `d_model = 256`, 8 cabezas, contexto de 128
   caracteres.
2. **Word-Llama (Punto 2).** Misma arquitectura y corpus, pero con tokenización por **palabra**:
   vocabulario de 12 000 palabras + `<unk>`, contexto de 64 tokens y muestreo top-k.
3. **RAG con Ollama (Punto 3).** Asistente de consulta sobre 3 artículos de arXiv
   (*Attention Is All You Need*, *LLaMA* y *Retrieval-Augmented Generation*). Embeddings con
   `nomic-embed-text`, generación con `llama3.2:3b`, fragmentos de ~800 caracteres con solapamiento
   de 150 y recuperación top-4 por similitud coseno.

## Resultados

### Modelos de lenguaje (Puntos 1 y 2)

| Métrica (conjunto de prueba) | Char-Llama | Word-Llama | Línea base |
|---|---|---|---|
| Perplejidad | **3.09** por carácter | **96.7** por palabra | 19.16 (unigramas de caracteres) · 441.8 (unigramas de palabras) |
| Bits por carácter (BPC) | 1.63 | **1.46** (equivalente) | 4.26 |
| Exactitud top-1 / top-5 | 0.642 / 0.905 | 0.239 / 0.458 | 0.181 / 0.514 (caracteres) |
| Parámetros | ~4.75 M | 9.29 M (6.1 M en embedding + salida) | — |
| Sobreajuste | No (`val_loss` baja las 8 épocas) | Sí (`val_loss` sube desde la época 3) | — |

Las perplejidades de carácter y de palabra no se comparan directamente; el BPC equivalente sí
permite compararlas. El de palabras es aproximado porque el 2.56 % de `<unk>` le ahorra predecir
las palabras raras.

### RAG (Punto 3)

| Métrica | Con RAG | Sin RAG (mismo `llama3.2:3b`) |
|---|---|---|
| Exactitud por palabras clave | **0.50** | 0.08 |
| Corrección según LLM-juez (1–5) | **3.75** | 2.75 |
| Fidelidad al contexto (1–5) | 3.58 | — |
| Similitud semántica | 0.672 | 0.676 |
| ROUGE-L F1 | 0.212 | 0.210 |

| Recuperación (k = 4) | Valor |
|---|---|
| Doc-Hit@1 (documento correcto primero) | 0.92 |
| Hit@4 (fragmento con la respuesta entre los 4) | 0.58 |
| MRR | 0.41 |
| Precisión de contexto@4 | 0.25 |
| Abstención ante preguntas fuera de los documentos | 2 de 2 |

**Lectura de los resultados:**

* El Char-Llama aprende ortografía, puntuación, diálogos y nombres propios solo a partir de
  caracteres, y reduce la perplejidad de 19.2 a 3.09 frente a la línea base.
* El Word-Llama produce texto sin errores ortográficos y con un contexto más largo, pero sobreajusta:
  ~450 k tokens son pocos para 12 000 palabras. Tiende a recombinar frases vistas (98 % de sus
  bigramas existen en el entrenamiento, frente a 76 % en texto real).
* Con RAG el modelo deja de inventar datos: sin contexto dice 768 en lugar de 512 para `d_model`
  y confunde RAG con otra cosa. La recuperación es el cuello de botella: las 4 preguntas sobre
  LLaMA no recuperaron el fragmento con la respuesta y el sistema respondió "No lo sé" (correcto,
  no alucina, pero son respuestas perdidas). Una búsqueda híbrida (BM25 + densa) o un *re-ranker*
  lo mejorarían.
