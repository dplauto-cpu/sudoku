# 📁 Estructura del proyecto

```
sudoku/
│
├── 📁 images/                       # Dataset de imágenes
│   ├── train/                       # 170 imágenes de entrenamiento
│   ├── val/
│   └── test/
│
├── 📁 labels/                       # Labels YOLO (bounding boxes)
│   ├── train/
│   └── val/
│
├── 📁 celdas_reales/                # Dataset etiquetado manualmente
│   ├── x_real.npy                   # Celdas reales extraídas con warp
│   └── y_real.npy                   # Etiquetas (1-9)
│
├── 📁 app/                          # ★ PASO 5-7 ★
│   ├── app.py                       # Aplicación Streamlit
│   ├── pipeline.py                  # Funciones del pipeline (Paso 4 + arreglos Paso 7)
│   ├── requirements.txt             # Dependencias
│   ├── modelo_yolo.pt               # Copia del modelo YOLO
│   ├── modelo_digitos_v3.keras      # CNN final (reentrenado, 464 celdas reales)
│   └── modelo_solver_1m.keras       # Solver neuronal (fallback, 1M partidas) — Paso 7
│
├── modelo_yolo.pt                   # ★ PASO 1 ★ Detector de cuadrícula
├── modelo_digitos_v2.keras          # CNN con augmentation (intermedio)
├── modelo_digitos_v3.keras          # ★ PASO 5-etiquetado ★ CNN final
│
├── sudoku_Paso1_actualizado.ipynb   # Entrenar YOLO
├── sudoku_Paso2b_warp.ipynb         # Segmentación con perspective warp
├── sudoku_Paso3b_CNN_mejorada.ipynb  # Reentrenar CNN con augmentation
├── sudoku_Paso4_etiquetado.ipynb    # Etiquetado manual + CNN final
├── sudoku_Paso5_Pipelinecompleto.ipynb  # Pipeline integrado
└── sudoku_Paso6_Streamlit.ipynb     # App Streamlit
```

---

## 📋 RESUMEN DEL PROYECTO

### ✅ PASO 1 — Detección de cuadrícula con YOLO (COMPLETADO)
- Modelo YOLO11 nano entrenado con dataset propio de sudokus
- 30 épocas de entrenamiento con GPU en Colab
- Precisión de detección: ~98-99% en imágenes de test
- Modelo guardado: `modelo_yolo.pt`
- Notebook reescrito en estilo consistente con el resto

### ✅ PASO 2b — Segmentación con Perspective Warp (COMPLETADO)
**Problema resuelto:** el recorte rectangular de YOLO producía celdas
torcidas con líneas de cuadrícula incluidas. La CNN recibía imágenes
corruptas y fallaba independientemente de su calidad.

**Solución implementada:**
- `encontrar_esquinas()`: detecta las 4 esquinas reales de la cuadrícula
  usando `adaptiveThreshold` + `findContours` + `approxPolyDP`
- `ordenar_esquinas()`: ordena los 4 puntos consistentemente
- `warp_perspectiva()`: transforma el trapezoide en cuadrado 450×450 px
- `segmentar_celdas()`: 81 celdas de 50×50 px con margen limpio
- `hay_digito()`: filtro simplificado por área bruta de píxeles (umbral 6%)

**Bugs corregidos durante la implementación:**
- `return` dentro del bucle → solo procesaba 1 celda (luego 9)
- Binarización Otsu en celdas vacías → píxeles aleatorios
  → solución: `adaptiveThreshold` con `THRESH_BINARY_INV`
- Falsos positivos en columna 0 → margen extra de 3px en borde izquierdo

**Resultado:** 81 celdas limpias, dígitos centrados, fondo negro

### ✅ PASO 3b (CNN mejorada) — Reentrenamiento con Augmentation (COMPLETADO)
**Problema:** CNN entrenada con MNIST puro no reconocía dígitos impresos.
Solo reconocía 6, 8 y 9 (los más "gordos"). El 1, 2, 3, 5, 7 fallaban.

**Solución:**
- Augmentation realista: erosión/dilatación, rotación ±5°,
  desplazamiento ±3px, ruido gaussiano, variación de brillo
- `N_AUG = 8`: 8 versiones por imagen → dataset ×9
- `BatchNormalization` en cada bloque convolucional
- Guardado como `modelo_digitos_v2.keras`

**Resultado:** mejora parcial, seguía sin reconocer dígitos finos

### ✅ PASO 4-etiquetado — CNN con Datos Reales (COMPLETADO)
**Problema:** el augmentation no fue suficiente para cerrar el domain gap
MNIST manuscrito → tipografía impresa de sudoku.

**Solución:**
- Extracción automática de celdas de 170 imágenes de train con warp
- Etiquetado manual interactivo (~200 celdas, bloques de 20)
- Oversampling ×50 de celdas reales para que compitan con MNIST
- Reentrenamiento mezclando MNIST aumentado + celdas reales
- `EarlyStopping` con `restore_best_weights`
- Guardado como `modelo_digitos_v3.keras`

**Resultado:** 100% de precisión en todos los dígitos 1-9

### ✅ PIPELINE COMPLETO — Integración y ajuste fino (COMPLETADO)
**Ajuste final:** `hay_digito()` simplificado a área bruta (sin filtro
de contornos que descartaba dígitos finos con poca "masa")

**Resultado final:**
- 22/23 dígitos correctamente detectados
- Falsos 1s en columna 0 → resuelto con margen_izq extra
- 23/23 dígitos correctos → backtracking resuelve el sudoku

### ✅ PASO 5 (Pipeline completo) — Funciones integradas (COMPLETADO)
- `imagen_a_matriz()`: YOLO + warp + CNN → matriz 9×9
- `resolver_sudoku()`: backtracking clásico
- `mostrar_resultado()`: pipeline completo con overlay
  (pistas en blanco, solución en verde)
- `pipeline.py` exportado para Streamlit

### ⚠️ PASO 6 (Streamlit) — Demo web (FUNCIONANDO, con limitaciones conocidas pero no resueltas)
- `app.py` generado y ejecutado en local
- Interfaz: subir foto → mostrar solución lado a lado
- Matrices detectada y resuelta en formato tabular
- Ejecutar en local: `streamlit run app.py`

### ⚠️ PASO 7 — Debugging en producción con fotos reales (tampoco mejoró sensiblemente la incertidumbre)

**Problema de partida:** la app funcionaba sin errores de código, pero en
la mayoría de fotos reales devolvía "Sin solución. Algún dígito mal
leído." o "Solo N pistas. Mínimo 17." El 100% de precisión del Paso 4 era
sobre el dataset de celdas ya recortadas — no sobre fotos nuevas
completas, y el backtracking exige que **todas** las pistas leídas sean
correctas (un solo dígito mal leído invalida el tablero entero).

**Bugs y mejoras, en el orden en que se encontraron:**

1. **Mismatch de preprocesado train/inferencia en `segmentar_celdas()`**
   (Paso4-etiquetado usaba `margen` constante; `Paso2b`/`pipeline.py` usaban
   `margen+3` en fila/columna 0) → corregido para que coincidan exactamente.
   Se reetiquetaron las celdas reales (206 → 464) con el recorte corregido
   y se reentrenó `modelo_digitos_v3.keras`. Mejora marginal: el cuello de
   botella real seguía siendo otro.

2. **Bug de carga de modelos `.keras` entrenados en Colab**: Keras local
   (3.12.2) no reconocía la clave `quantization_config` que añade una
   versión más nueva de Keras al guardar capas `Dense`/`Embedding`.
   Solución: el `.keras` es un zip (`config.json` + `model.weights.h5`);
   se repara quitando esa clave del `config.json` sin tocar los pesos.
   Mismo bug y mismo arreglo aplicado a `cnn_1m.keras` (modelo "solver"
   de ejemplo del profesor, ver `6-DL_Project/`).

3. **El backtracking nunca tolera errores de OCR**: un solo dígito mal
   leído que choque con otro en su fila/columna/caja hace el tablero
   irresoluble de raíz — ningún solver (clásico ni neuronal) puede
   completarlo sin tocar las pistas. `resolver_sudoku()` además podía
   tardar minutos en confirmar "sin solución" en tableros poco
   restringidos (sin heurística de poda). Se añadió:
   - `limpiar_conflictos()`: detecta pistas duplicadas en la misma
     fila/columna/caja y borra las repetidas antes de intentar resolver.
     **Este fue el cambio que más mejoró la tasa de éxito** (1/8 → 5/8
     en el set de prueba).
   - Límite de iteraciones en `resolver_sudoku()` para no colgar la app.
   - `resolver_sudoku_neural()`: fallback opcional usando un CNN-solver
     (`modelo_solver_1m.keras`, entrenado con 1M de partidas, recibe el
     tablero 81-celdas y predice dígito por celda) con enmascarado de
     candidatos válidos. Solo rellena celdas vacías, no corrige pistas
     dadas — por eso no ayuda en el caso más común (conflicto directo
     entre pistas), pero queda como red de seguridad.

4. **Procesos huérfanos de Streamlit**: tras varios `Ctrl+C` + relanzar,
   quedaron procesos viejos compitiendo por el puerto 8501 sin que el
   reinicio los matara realmente, causando que persistiera un error ya
   corregido en el código. Solución: identificar y matar explícitamente
   los PID viejos (`taskkill`) antes de relanzar.

5. **Restos de líneas de cuadrícula colándose dentro de las celdas**:
   visualizando directamente el mosaico de las 81 celdas binarizadas (no
   solo las confianzas numéricas) se vieron rayas de la cuadrícula
   contaminando el contenido que recibe la CNN. Se añadió un filtro
   morfológico (`largo_linea` en `segmentar_celdas()`) que detecta y
   borra rachas largas y rectas de píxeles. Mejora adicional: 5/8 → 6/8.

6. **Regresión del filtro anterior: borraba los dígitos "1"** (una línea
   vertical centrada es indistinguible de un resto de cuadrícula para un
   filtro que no mira la posición). Confirmado con fotos reales: el área
   de un "1" pasaba de ~0.13-0.15 a ~0.04-0.06 (por debajo del umbral de
   `hay_digito`, 0.06) y la celda se trataba como vacía. Arreglado
   restringiendo el borrado a una banda pegada al borde de la celda
   (`banda_borde=7` sobre 28px) en vez de a toda la celda — mantiene el
   "1" intacto y conserva la mejora del punto 5.

7. **Bug de renderizado (cosmético, no afectaba la solución)**: el
   overlay dibujaba el dígito en blanco encima de **todas** las celdas
   con pista, incluso las que ya mostraban ese número en la foto
   original, creando un efecto de "doble dígito" fantasma. Arreglado:
   solo se pinta encima de las celdas que el solver rellenó (verde); las
   pistas originales se dejan tal cual están en la foto.

**Pendiente / sin resolver en el plazo del ejercicio (4 días laborables):**
- Sangrado de tinta entre celdas adyacentes en fotos con cierta
  rotación/inclinación (un dígito puede generar una lectura fantasma en
  la celda vecina); `limpiar_conflictos()` detecta el conflicto pero no
  siempre se queda con la copia correcta.
- Umbral de confianza de la CNN (0.70) descarta algunas lecturas
  correctas por poco (caso real visto: "1" leído bien con confianza
  0.648). Bajarlo un poco (≈0.65) podría rascar algunas pistas más sin
  asumir mucho riesgo, pero no se ha probado de forma sistemática.
- Una docena de celdas etiquetadas manualmente en el Paso 4 mostraban
  dígitos rotados 90°; no se ha podido reproducir en las fotos de test
  actuales, así que no está claro si es un problema generalizado de
  `encontrar_esquinas()`/`ordenar_esquinas()` o puntual de algunas fotos.

**Resultado al cierre de la sesión:** 6/8 en el set de prueba fijo
(antes de hoy: 1/8), más las fotos reales nuevas probadas durante la
sesión. Suficiente para la demo de entrega; las limitaciones de arriba
quedan documentadas para una posible iteración futura.

---

## 🔄 Pipeline final

```
FOTO DEL SUDOKU
      ↓
YOLO (modelo_yolo.pt)
  → Detecta cuadrícula con confianza >50%
      ↓
Perspective Warp (OpenCV)
  → encontrar_esquinas() → ordenar_esquinas() → warp_perspectiva()
  → Grid 450×450 px perfectamente alineado
      ↓
segmentar_celdas()
  → 81 celdas 50×50 px → margen 4px (+3px en col/fila 0)
  → adaptiveThreshold → MORPH_OPEN → 28×28 float32
      ↓
hay_digito() — umbral área bruta 6%
  → descarta celdas vacías
      ↓
CNN (modelo_digitos_v3.keras)
  → pred[0]=0 (no existe el 0 en sudoku)
  → confianza >= 70%
      ↓
resolver_sudoku() — Backtracking
      ↓
Overlay: pistas (blanco) + solución (verde)
      ↓
STREAMLIT muestra resultado
```

---

## 📊 Estado actual

| Componente | Estado | Modelo/Archivo |
|---|---|---|
| YOLO — detección de cuadrícula | ✅ Funcionando | `modelo_yolo.pt` |
| Perspective warp — segmentación | ✅ Sin restos de líneas, "1" protegido | Paso 2b + Paso 7 |
| CNN — reconocimiento de dígitos | ⚠️ Buena en celdas aisladas, falla en alguna foto completa | `modelo_digitos_v3.keras` (464 celdas reales) |
| Limpieza de conflictos directos | ✅ Antes de resolver | `limpiar_conflictos()` |
| Backtracking — resolución | ✅ Con límite de iteraciones | `resolver_sudoku()` |
| Fallback neuronal (solver 1M partidas) | ⚠️ Integrado, solo rellena vacíos | `modelo_solver_1m.keras` |
| Streamlit — demo | ⚠️ Funciona ocasionalmente, 6/8 en set de prueba | `app.py` |

---

## 💡 Lecciones aprendidas según iban llegando los problemas

**1. El problema raíz no siempre está donde parece.**
Se invirtió tiempo mejorando la CNN cuando el problema real era la
segmentación. Una celda mal extraída no puede ser reconocida por
ninguna CNN, por muy buena que sea.

**2. Visualizar antes de optimizar.**
El diagnóstico visual (las 81 celdas preprocesadas) reveló inmediatamente
que las celdas llegaban torcidas. Sin esa visualización se habrían
seguido ajustando umbrales sin llegar a ningún lado.

**3. Cuidado con Domain gap.**
MNIST tiene 99%+ de accuracy pero es inútil para dígitos impresos de
sudoku. La precisión en el dataset de entrenamiento no dice nada sobre
el rendimiento en producción si los dominios son distintos.

**4. Pocos datos reales valen más que mucho augmentation.**
206 celdas etiquetadas manualmente + oversampling ×50 superaron a
decenas de miles de imágenes MNIST con augmentation agresivo.

**5. Los bugs de indentación en Python son especialmente irritantes.**
Un `return` mal indentado hizo que la función procesara 1 celda en
lugar de 81, y el error no era obvio hasta ver `✅ 1 celda extraída`.

**6. Simplificar cuando las condiciones mejoran.**
`hay_digito()` empezó con filtros complejos de contorno y forma.
Con el warp las celdas llegaban limpias y bastó con medir el área
bruta de píxeles. La complejidad extra solo causaba falsos negativos.

**7. El backtracking exige perfección; un solver "tolerante" la mayoría
de las veces no existe gratis.** Un solo dígito mal leído que choque con
otro invalida el tablero entero. Limpiar los conflictos obvios *antes*
de resolver dio más rendimiento que cualquier ajuste de la CNN o un
solver neuronal de respaldo — y eso último solo sirve si puede tocar las
pistas, no solo las celdas vacías.

**8. Un filtro de "limpieza" puede destruir justo la señal que buscas si
comparte forma con el ruido.** Un dígito "1" es geométricamente casi
idéntico a un resto de línea de cuadrícula (ambos son rachas verticales
largas y rectas). La posición dentro de la celda (centro vs borde) es la
diferencia que sí importa — filtrar por longitud sin mirar posición
genera una regresión silenciosa que solo se ve probando con dígitos "1"
reales, no con métricas agregadas.

**9. Verificar versiones de Keras entre el entorno de entrenamiento
(Colab) y el de inferencia (local) antes de asumir que un modelo está
roto.** Un `.keras` guardado con una versión más nueva puede fallar al
cargar localmente por una clave de config no reconocida
(`quantization_config`), sin que el modelo en sí tenga ningún problema.
Es un zip editable: se puede reparar sin reentrenar.

**10. Reiniciar un proceso no es lo mismo que matarlo.** `Ctrl+C` en una
terminal no siempre mata al proceso hijo real cuando se lanza a través
de un wrapper (`streamlit.exe` → `python.exe`). Procesos huérfanos
compitiendo por el mismo puerto producen errores que parecen "el código
no se ha actualizado" cuando en realidad es "hay dos servidores a la
vez".

**11. Mirar literalmente lo que ve el modelo gana a cualquier métrica.**
Las confianzas numéricas (0.70, 1.00...) no revelaron el problema de las
líneas de cuadrícula ni el de los "1" borrados. Verlo en un mosaico de
celdas sí, en segundos.
