"""
pipeline.py — Pipeline completo para resolver sudokus desde fotos.
Importado por app.py (Streamlit).
"""

import cv2
import numpy as np
from ultralytics import YOLO
import keras


def cargar_modelos(ruta_yolo, ruta_cnn, ruta_solver=None):
    modelo_yolo = YOLO(ruta_yolo)
    modelo_cnn  = keras.models.load_model(ruta_cnn)
    modelo_solver = keras.models.load_model(ruta_solver) if ruta_solver else None
    return modelo_yolo, modelo_cnn, modelo_solver


def encontrar_esquinas(img_rgb, box_yolo):
    x1, y1, x2, y2 = box_yolo
    margen = 20
    H, W   = img_rgb.shape[:2]
    rx1, ry1 = max(0, x1-margen), max(0, y1-margen)
    rx2, ry2 = min(W, x2+margen), min(H, y2+margen)
    recorte  = img_rgb[ry1:ry2, rx1:rx2]
    gray     = cv2.cvtColor(recorte, cv2.COLOR_RGB2GRAY)
    blur     = cv2.GaussianBlur(gray, (5, 5), 0)
    binaria  = cv2.adaptiveThreshold(blur, 255,
                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                   cv2.THRESH_BINARY_INV, 11, 2)
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None
    contorno_mayor = max(contornos, key=cv2.contourArea)
    if cv2.contourArea(contorno_mayor) < 0.3 * (rx2-rx1) * (ry2-ry1):
        return None
    epsilon = 0.02 * cv2.arcLength(contorno_mayor, True)
    approx  = cv2.approxPolyDP(contorno_mayor, epsilon, True)
    intentos = 0
    while len(approx) != 4 and intentos < 10:
        epsilon *= 1.2
        approx   = cv2.approxPolyDP(contorno_mayor, epsilon, True)
        intentos += 1
    if len(approx) != 4:
        rect   = cv2.minAreaRect(contorno_mayor)
        approx = cv2.boxPoints(rect).reshape(-1, 1, 2).astype(int)
    puntos = approx.reshape(-1, 2).astype(np.float32)
    puntos[:, 0] += rx1
    puntos[:, 1] += ry1
    return puntos


def ordenar_esquinas(puntos):
    puntos  = puntos.reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sumas   = puntos.sum(axis=1)
    diffs   = np.diff(puntos, axis=1).flatten()
    ordered[0] = puntos[np.argmin(sumas)]
    ordered[2] = puntos[np.argmax(sumas)]
    ordered[1] = puntos[np.argmin(diffs)]
    ordered[3] = puntos[np.argmax(diffs)]
    return ordered


def warp_perspectiva(img_rgb, esquinas, tamaño=450):
    origen  = ordenar_esquinas(esquinas)
    destino = np.float32([[0,0],[tamaño,0],[tamaño,tamaño],[0,tamaño]])
    M       = cv2.getPerspectiveTransform(origen, destino)
    return cv2.warpPerspective(img_rgb, M, (tamaño, tamaño))


def segmentar_celdas(grid_warped, tamaño_celda=50, margen=4, largo_linea=15, banda_borde=7):
    """
    largo_linea: longitud mínima (px, en la celda ya redimensionada a 28x28)
    para considerar una racha de píxeles como resto de línea de cuadrícula.
    banda_borde: solo se borran esas rachas si caen dentro de esta franja
    pegada al borde de la celda (donde aparecen los restos de la cuadrícula).
    Un dígito "1" es básicamente una línea vertical centrada: si se borrara
    cualquier racha larga sin importar dónde esté, el "1" desaparece casi
    entero (confirmado: su área pasaba de ~0.13-0.15 a ~0.04-0.06, por
    debajo del umbral de hay_digito). Restringir el borrado al borde
    mantiene el "1" intacto y sigue limpiando las líneas de cuadrícula.
    """
    gray   = cv2.cvtColor(grid_warped, cv2.COLOR_RGB2GRAY)
    celdas = []
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (largo_linea, 1))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, largo_linea))
    for i in range(9):
        for j in range(9):
            y1c = i * tamaño_celda
            y2c = (i + 1) * tamaño_celda
            x1c = j * tamaño_celda
            x2c = (j + 1) * tamaño_celda
            celda = gray[y1c:y2c, x1c:x2c]
            margen_izq = margen + 3 if j == 0 else margen
            margen_arr = margen + 3 if i == 0 else margen
            celda_sin_bordes = celda[margen_arr:tamaño_celda-margen,
                                     margen_izq:tamaño_celda-margen]
            celda_28  = cv2.resize(celda_sin_bordes, (28, 28))
            celda_bin = cv2.adaptiveThreshold(
                celda_28, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 4)
            kernel    = np.ones((2, 2), np.uint8)
            celda_bin = cv2.morphologyEx(celda_bin, cv2.MORPH_OPEN, kernel)

            lineas_h  = cv2.morphologyEx(celda_bin, cv2.MORPH_OPEN, kh)
            lineas_v  = cv2.morphologyEx(celda_bin, cv2.MORPH_OPEN, kv)
            lineas    = cv2.bitwise_or(lineas_h, lineas_v)
            mascara_borde = np.zeros_like(lineas)
            mascara_borde[:banda_borde, :]  = 255
            mascara_borde[-banda_borde:, :] = 255
            mascara_borde[:, :banda_borde]  = 255
            mascara_borde[:, -banda_borde:] = 255
            lineas    = cv2.bitwise_and(lineas, mascara_borde)
            celda_bin = cv2.bitwise_and(celda_bin, cv2.bitwise_not(lineas))

            celdas.append(celda_bin.astype("float32") / 255.0)
    return celdas


def hay_digito(celda_norm, umbral_area=0.06):
    area_bruta = float(np.sum(celda_norm > 0.5) / (28 * 28))
    return area_bruta > umbral_area, area_bruta


def es_valido(tablero, fila, col, num):
    if num in tablero[fila]: return False
    if num in [tablero[i][col] for i in range(9)]: return False
    fi = (fila // 3) * 3
    ci = (col  // 3) * 3
    for i in range(fi, fi + 3):
        for j in range(ci, ci + 3):
            if tablero[i][j] == num: return False
    return True


def resolver_sudoku(tablero, _contador=None, limite=200_000):
    """
    Backtracking clásico. Con pocas pistas y/o tableros contradictorios
    (típico cuando el OCR lee mal) la búsqueda sin heurística puede
    explotar combinatoriamente antes de concluir que no hay solución.
    'limite' acota el número de llamadas para no colgar la app.
    """
    if _contador is None:
        _contador = [0]
    _contador[0] += 1
    if _contador[0] > limite:
        return False
    for i in range(9):
        for j in range(9):
            if tablero[i][j] == 0:
                for num in range(1, 10):
                    if es_valido(tablero, i, j, num):
                        tablero[i][j] = num
                        if resolver_sudoku(tablero, _contador, limite): return True
                        tablero[i][j] = 0
                return False
    return True


def limpiar_conflictos(tablero):
    """
    Si el OCR ha leído dos veces el mismo dígito en la misma fila, columna
    o caja, el tablero es contradictorio de raíz y ningún solver (clásico
    o neuronal) puede completarlo sin tocar las pistas. Esta función deja
    solo una ocurrencia de cada conflicto (las demás pasan a vacío) para
    que el resto del tablero sí se pueda intentar resolver.

    Devuelve (tablero_limpio, hubo_conflictos).
    """
    tab = tablero.copy()
    hubo_conflictos = False

    for fila in range(9):
        vistos = {}
        for col in range(9):
            num = tab[fila][col]
            if num == 0: continue
            if num in vistos:
                tab[fila][col] = 0
                hubo_conflictos = True
            else:
                vistos[num] = col

    for col in range(9):
        vistos = {}
        for fila in range(9):
            num = tab[fila][col]
            if num == 0: continue
            if num in vistos:
                tab[fila][col] = 0
                hubo_conflictos = True
            else:
                vistos[num] = fila

    for bi in range(0, 9, 3):
        for bj in range(0, 9, 3):
            vistos = set()
            for i in range(bi, bi + 3):
                for j in range(bj, bj + 3):
                    num = tab[i][j]
                    if num == 0: continue
                    if num in vistos:
                        tab[i][j] = 0
                        hubo_conflictos = True
                    else:
                        vistos.add(num)

    return tab, hubo_conflictos


def resolver_sudoku_neural(tablero, modelo_solver, max_pasos=81):
    """
    Fallback tolerante a errores de OCR: usa el CNN solver (entrenado con
    1M de partidas, entrada = tablero 81 celdas, salida = softmax 10 clases
    por celda) para rellenar el tablero celda a celda.

    En cada paso se enmascaran las probabilidades con los dígitos que
    todavía son válidos por fila/columna/caja (el argmax crudo del modelo
    NO respeta las reglas del sudoku) y se rellena la celda vacía con
    mayor confianza entre los dígitos válidos. Si alguna celda vacía se
    queda sin ningún dígito válido, el camino no tiene salida y se aborta.

    Devuelve el tablero resuelto (numpy 9x9) o None si no consigue completarlo.
    """
    tab = tablero.copy()

    for _ in range(max_pasos):
        vacias = [(i, j) for i in range(9) for j in range(9) if tab[i][j] == 0]
        if not vacias:
            break

        entrada = tab.reshape(1, 81).astype("float32")
        pred = modelo_solver.predict(entrada, verbose=0)[0]  # (81, 10)

        mejor_celda, mejor_digito, mejor_conf = None, None, -1.0
        for (i, j) in vacias:
            validos = [n for n in range(1, 10) if es_valido(tab, i, j, n)]
            if not validos:
                return None  # callejón sin salida para este camino
            probs = pred[i * 9 + j]
            digito = max(validos, key=lambda n: probs[n])
            conf = float(probs[digito])
            if conf > mejor_conf:
                mejor_celda, mejor_digito, mejor_conf = (i, j), digito, conf

        i, j = mejor_celda
        tab[i][j] = mejor_digito

    if np.any(tab == 0):
        return None

    for i in range(9):
        for j in range(9):
            digito = tab[i][j]
            tab[i][j] = 0
            if not es_valido(tab, i, j, digito):
                tab[i][j] = digito
                return None
            tab[i][j] = digito

    return tab


def imagen_a_matriz(ruta_imagen, modelo_yolo, modelo_cnn):
    img     = cv2.imread(ruta_imagen)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = modelo_yolo(ruta_imagen, verbose=False)
    if len(results[0].boxes) == 0:
        return None, None, "No se detectó la cuadrícula"
    box  = results[0].boxes[0]
    if float(box.conf[0]) < 0.5:
        return None, None, "Confianza YOLO demasiado baja"
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    esquinas = encontrar_esquinas(img_rgb, (x1, y1, x2, y2))
    if esquinas is not None:
        grid_warped = warp_perspectiva(img_rgb, esquinas)
    else:
        esq_fb      = np.float32([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
        grid_warped = warp_perspectiva(img_rgb, esq_fb)
    celdas = segmentar_celdas(grid_warped)
    matriz = np.zeros((9, 9), dtype=int)
    for idx, celda in enumerate(celdas):
        i, j = idx // 9, idx % 9
        tiene, _ = hay_digito(celda)
        if not tiene: continue
        entrada  = celda.reshape(1, 28, 28, 1)
        pred     = modelo_cnn.predict(entrada, verbose=0)[0]
        pred[0]  = 0.0
        pred     = pred / pred.sum()
        digito   = int(np.argmax(pred))
        if float(pred[digito]) >= 0.70:
            matriz[i][j] = digito
    return matriz, grid_warped, f"{int(np.sum(matriz > 0))} dígitos detectados"


def mostrar_resultado(ruta_imagen, modelo_yolo, modelo_cnn, modelo_solver=None):
    matriz, grid_warped, resumen = imagen_a_matriz(ruta_imagen, modelo_yolo, modelo_cnn)
    if matriz is None:
        return None, None, None, resumen
    if int(np.sum(matriz > 0)) < 17:
        return None, matriz, None, f"Solo {int(np.sum(matriz > 0))} pistas. Mínimo 17."

    matriz_base, hubo_conflictos = limpiar_conflictos(matriz)
    mensaje_extra = " (se ignoró una pista duplicada/contradictoria)" if hubo_conflictos else ""

    matriz_resuelta = matriz_base.copy()
    if not resolver_sudoku(matriz_resuelta):
        matriz_resuelta = None
        if modelo_solver is not None:
            matriz_resuelta = resolver_sudoku_neural(matriz_base, modelo_solver)
            mensaje_extra += " (corregido con IA, puede contener errores)"
        if matriz_resuelta is None:
            return None, matriz, None, "Sin solución. Algún dígito mal leído."

    overlay = grid_warped.copy()
    h, w    = overlay.shape[:2]
    cell_h, cell_w = h // 9, w // 9
    for i in range(9):
        for j in range(9):
            if matriz[i][j] != 0: continue  # pista original: ya se ve en la foto, no la repintamos
            if matriz_resuelta[i][j] == 0: continue
            cx = j * cell_w + cell_w // 2
            cy = i * cell_h + cell_h // 2
            cv2.putText(overlay, str(matriz_resuelta[i][j]),
                        (cx - 8, cy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2, cv2.LINE_AA)
    return overlay, matriz, matriz_resuelta, f"Resuelto ({int(np.sum(matriz > 0))} pistas){mensaje_extra}"
