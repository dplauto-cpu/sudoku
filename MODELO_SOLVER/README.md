# MODELO_SOLVER

Red neuronal que recibe el tablero de sudoku como un array plano de 81
valores (0 = celda vacía, 1-9 = dígito) y devuelve, para cada celda, una
distribución de probabilidad sobre los dígitos 0-9.

## Origen

Este modelo (`cnn_1m.keras`, copiado a `CODE/MODELOS/modelo_solver_1m.keras`
en este repo) **no se entrenó en este proyecto**: es el ejemplo/pista
proporcionado por el profesor en `6-DL_Project/`, entrenado con 1 millón
de partidas. `sudoku.csv` (en `DATA/`) es el dataset de partidas de
origen — no hay notebook de entrenamiento propio para este modelo aquí.

## Uso en el pipeline

Se usa como **fallback opcional** en `resolver_sudoku_neural()`
(`CODE/pipeline.py`): si el backtracking clásico no encuentra solución,
se intenta rellenar las celdas vacías con este modelo, enmascarando en
cada paso los dígitos que violarían las reglas de fila/columna/caja.

**Limitación importante**: solo puede rellenar celdas que el OCR detectó
como vacías. No puede corregir una pista ya leída (aunque esté mal), así
que no ayuda en el caso más común de fallo: dos pistas leídas que
chocan entre sí en la misma fila/columna/caja. Para esos casos,
`limpiar_conflictos()` (que sí actúa sobre las pistas) es la pieza que
realmente importa — ver `schedule_actualizado.md`.
