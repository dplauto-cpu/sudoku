# Sudoku Solver — Foto → Solución

Proyecto de Deep Learning: dada una fotografía de un sudoku, la app
detecta la cuadrícula, lee los dígitos, resuelve el tablero y devuelve
la imagen con la solución superpuesta.

## Demo

```bash
cd CODE
pip install -r requirements.txt
streamlit run app.py
```

Sube una foto del sudoku (cuadrícula completa, bien iluminada) y la app
muestra la solución al lado, con las pistas originales tal cual estaban
en la foto y los dígitos resueltos en verde.

## Pipeline (6 fases — ver detalle en `esquema_sudoku_texto.txt`)

```
FOTO
  → YOLO localiza la cuadrícula (MODELO_YOLO)
  → Corrección de perspectiva + segmentación en 81 celdas (OpenCV)
  → CNN lee el dígito de cada celda, o vacío (MODELO_CNN)
  → Se limpian conflictos directos (misma pista repetida en fila/col/caja)
  → Backtracking resuelve el tablero (fallback: solver neuronal, MODELO_SOLVER)
  → Overlay con la solución sobre la foto original
```

## Estructura del repositorio

| Carpeta | Contenido |
|---|---|
| `MODELO_YOLO/` | Dataset (imágenes + labels) y notebook de entrenamiento del detector de cuadrícula |
| `MODELO_CNN/` | Notebooks del desarrollo del lector de dígitos (warp/segmentación, entrenamiento con MNIST+augmentation, reentrenamiento con celdas reales etiquetadas a mano) |
| `MODELO_SOLVER/` | Dataset de partidas y nota sobre el solver neuronal de respaldo (ver su README) |
| `NOTEBOOKS_INTEGRACION/` | Notebooks que integran todo en `pipeline.py` y generan `app.py` |
| `CODE/` | Código final: `app.py` (demo Streamlit), `pipeline.py` (funciones del pipeline) y `CODE/MODELOS/` con los 3 modelos entrenados/usados |
| `BACKUPS_NO_ENTREGAR/` | Copias de seguridad de versiones anteriores del modelo CNN, fuera del flujo de entrega |
| `schedule_actualizado.md` | Bitácora de desarrollo: qué se hizo, qué bugs se encontraron y cómo se arreglaron |
| `esquema_sudoku_texto.txt` | Documento de diseño original del pipeline por fases |

## Modelos usados

1. **YOLO11n** (`CODE/MODELOS/modelo_yolo.pt`) — detecta la cuadrícula del sudoku en la foto.
2. **CNN de dígitos** (`CODE/MODELOS/modelo_digitos_v3.keras`) — clasifica cada una de las 81 celdas (vacía o dígito 1-9). Entrenada con MNIST aumentado + celdas reales etiquetadas a mano.
3. **Solver neuronal** (`CODE/MODELOS/modelo_solver_1m.keras`) — red entrenada con 1M de partidas que rellena celdas vacías como red de seguridad cuando el backtracking clásico falla. Solo complementa, no sustituye, al backtracking (no puede corregir pistas ya leídas, solo completar huecos).

## Limitaciones conocidas

El punto más frágil del pipeline es la lectura de dígitos: un solo
dígito mal leído invalida el tablero completo para el backtracking. Tras
una sesión de depuración (ver `schedule_actualizado.md`, sección "PASO
7") la tasa de resolución en el set de prueba pasó de 1/8 a 6/8.
Quedan casos sin resolver del todo: sangrado de tinta entre celdas
vecinas en fotos con cierta inclinación, y algunas lecturas correctas
descartadas por estar justo por debajo del umbral de confianza de la
CNN. Detalle completo de los 7 problemas encontrados y sus arreglos en
`schedule_actualizado.md`.
