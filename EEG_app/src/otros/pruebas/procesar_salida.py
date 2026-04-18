import pydirectinput
import time

# Variable global para saber qué tecla estamos apretando actualmente
estado_actual = None

def ejecutar_accion_continua(clase_predicha, confianza):
    global estado_actual
    
    # 1. Definir el mapeo
    mapeo = {
        "left_hand": 'a',      # En juegos, 'A' es izquierda
        "right_hand": 'd',     # En juegos, 'D' es derecha
        "feet": 'w',           # 'W' para avanzar (o 'space' para saltar)
        "rest": None           # Clase descanso (importante para soltar teclas)
    }

    # 2. Filtrar por confianza (si es baja, tratamos como descanso)
    tecla_deseada = mapeo.get(clase_predicha)
    if confianza < 60.0: 
        tecla_deseada = None

    # 3. Lógica de cambio de estado
    if tecla_deseada == estado_actual:
        # Si seguimos pensando lo mismo, no hacemos nada (la tecla sigue apretada)
        pass
        
    else:
        # A) Soltamos la tecla anterior si había una
        if estado_actual is not None:
            print(f"Soltando {estado_actual}")
            pydirectinput.keyUp(estado_actual)
        
        # B) Apretamos la nueva tecla si no es descanso
        if tecla_deseada is not None:
            print(f"Presionando {tecla_deseada}")
            pydirectinput.keyDown(tecla_deseada)
        
        # Actualizamos el estado
        estado_actual = tecla_deseada


def __main__():
    ejecutar_accion_continua('feet', 90)
    ejecutar_accion_continua('left_hand', 90)
    ejecutar_accion_continua('right_hand', 90)
    ejecutar_accion_continua('right_hand', 90)
    ejecutar_accion_continua('rest', 90)
    return 0

if __name__ == "__main__":
    __main__()