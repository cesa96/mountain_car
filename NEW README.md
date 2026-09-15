# MountainCar-v0 con Q-Learning
## Análisis Experimental y Optimización de Hiperparámetros

## Autor



## Introducción

Este proyecto documenta el proceso de aprendizaje por refuerzo (Reinforcement Learning) aplicado al entorno MountainCar-v0 utilizando el algoritmo Q-Learning tabular.



El propósito principal consiste en analizar el comportamiento del algoritmo Q-Learning bajo distintas configuraciones de hiperparámetros y determinar cuáles producen el mejor desempeño dentro del entorno.

---

# Objetivos

## Objetivo General

Implementar y evaluar un agente Q-Learning capaz de resolver el entorno MountainCar-v0 mediante la optimización de hiperparámetros.

## Objetivos Específicos

- Comprender el funcionamiento interno de Q-Learning.
- Analizar el impacto de los hiperparámetros sobre el desempeño del agente.
- Realizar experimentación sistemática mediante búsqueda aleatoria (Random Search).
- Comparar resultados utilizando métricas objetivas.
- Identificar la mejor configuración encontrada.

---

# Descripción del Problema

MountainCar-v0 es un entorno clásico de Reinforcement Learning.

Un vehículo con potencia insuficiente debe alcanzar una bandera ubicada en la cima de una montaña.

La potencia del motor no es suficiente para subir directamente la pendiente, por lo que el agente debe aprender una estrategia indirecta que consiste en desplazarse inicialmente en sentido contrario para acumular energía potencial y posteriormente utilizar dicha energía para alcanzar el objetivo.

---

# Espacio de Estados

El entorno proporciona dos variables continuas:

| Variable | Descripción | Rango |
|-----------|-------------|--------|
| Position | Posición horizontal del vehículo | [-1.2, 0.6] |
| Velocity | Velocidad del vehículo | [-0.07, 0.07] |

Por tratarse de variables continuas, se requiere una discretización para aplicar Q-Learning tabular.

---

# Espacio de Acciones

El agente puede ejecutar tres acciones:

| Acción | Valor |
|----------|--------|
| Acelerar izquierda | 0 |
| Sin aceleración | 1 |
| Acelerar derecha | 2 |

---

# Función de Recompensa

Cada paso ejecutado recibe una recompensa:

```text
-1
```

independientemente de la acción realizada.

El episodio termina cuando:

```text
position >= 0.5
```

o cuando se alcanza el límite máximo de pasos permitido.

Debido a esta formulación:

- Recompensas menos negativas indican mejor desempeño.
- Llegar a la meta en menos pasos genera mejores resultados.
- Un valor cercano a -110 suele considerarse una política satisfactoria.

---

# Fundamentos Teóricos

## Aprendizaje por Refuerzo

El aprendizaje por refuerzo permite que un agente aprenda mediante interacción continua con un entorno.

Durante el entrenamiento:

1. El agente observa un estado.
2. Selecciona una acción.
3. Recibe una recompensa.
4. Observa un nuevo estado.
5. Actualiza su conocimiento.

Este proceso se repite hasta construir una política capaz de maximizar la recompensa acumulada.

-
# Discretización del Espacio de Estados

El primer ejercicio para qlearning es discretizar el espacio de estados, de tal manera que se puedan almacenar en una qtable con el respectivo peso de las acciones con el fin de poder realizar el entrenamiento del algoritmo.  El neumero de estados está definido por el hiperparámetro n_bins que nos va a indicar el tamaño de la qtable. Por ejemplo por defecto viene un valor de 20,  lo que nos genera una tabla 20*20

Por ejemplo:

| n_bins | Estados |
|----------|----------|
| 20 | 400 |
| 30 | 900 |
| 40 | 1600 |
| 50 | 2500 |
| 60 | 3600 |

Cada celda corresponde a un estado discreto dentro de la Q-Table.

---

# Exploración y Explotación

El segundo paso es  implementar una política ε-greedy.

La política funciona de la siguiente forma:

- Con probabilidad ε:
  - Acción aleatoria.
- Con probabilidad 1−ε:
  - Mejor acción según la Q-Table.

Durante el entrenamiento ε disminuye gradualmente para favorecer la explotación del conocimiento adquirido.

---

# Hiperparámetros Analizados

Para la evaluación de los hiperparámetros, se tomaron  rangos para los siguientes hiperparámetros y de forma aleatoria se generaron 50 combinaciónes
## Número de divisiones

```python
n_bins ∈ {20,30,40,50,60}
```

## Learning Rate

```python
lr ∈ {0.01,0.03,0.05,0.1,0.2}
```

## Discount Factor

```python
gamma ∈ [0.95, 0.999]
```

## Decaimiento de Epsilon

```python
epsilon_decay ∈ [0.995, 0.99999]
```

## Parámetros Fijos

```python
epsilon_start = 1.0
epsilon_end = 0.01
episodes = 20000
```

---

# Metodología Experimental

Se implementó una búsqueda aleatoria (Random Search).

Para cada experimento:

1. Generar hiperparámetros aleatoriamente.
2. Configurar variables de ambiente.
3. Eliminar entrenamiento anterior.
4. Reentrenar desde cero.
5. Evaluar la política aprendida.
6. Registrar métricas.
7. Almacenar resultados en CSV.
8. Seleccionar la mejor configuración encontrada.

Se realizaron:

```text
50 experimentos independientes
```

---

# Variables de Ambiente Utilizadas

```bash
MOUNTAIN_CAR_N_BINS
MOUNTAIN_CAR_LR
MOUNTAIN_CAR_GAMMA
MOUNTAIN_CAR_EPSILON_START
MOUNTAIN_CAR_EPSILON_END
MOUNTAIN_CAR_EPSILON_DECAY
```

---

# Flujo de Ejecución

Para cada experimento se ejecutaron los siguientes comandos:

```bash
uv run mountaincar delete qlearning
```

```bash
uv run mountaincar train qlearning --episodes 20000
```

```bash
uv run mountaincar load qlearning --eval
```

---

# Métricas de Evaluación

Las métricas registradas fueron:

## Mean Reward

Promedio de recompensa obtenido durante la evaluación.

Ejemplo:

```text
Mean reward: -155.00
```

## Standard Deviation

Variabilidad observada durante la evaluación.

Ejemplo:

```text
+/- 7.55
```

## Success Rate

Número de episodios que alcanzaron la bandera.

Ejemplo:

```text
Reached the flag: 10/10 episodes
```

---

# Criterio de Selección

La mejor configuración corresponde al mayor valor de:

```text
Mean Reward
```


---

# Resultados

## Top 10 Configuraciones

## Top 10 Configuraciones

| Rank | Mean Reward | Std Dev | Flag Success | n_bins | LR | Gamma | Epsilon Decay |
|------|------------:|---------:|--------------|--------:|----:|-------:|--------------:|
| 1 | -126.8 | 14.16 | 10/10 | 40 | 0.20 | 0.979296 | 0.995309 |
| 2 | -126.9 | 32.72 | 10/10 | 20 | 0.05 | 0.951064 | 0.995264 |
| 3 | -128.2 | 17.17 | 10/10 | 50 | 0.10 | 0.988664 | 0.999001 |
| 4 | -130.7 | 22.50 | 10/10 | 40 | 0.10 | 0.998349 | 0.997976 |
| 5 | -131.5 | 19.40 | 10/10 | 40 | 0.10 | 0.992125 | 0.995694 |
| 6 | -131.7 | 10.33 | 10/10 | 20 | 0.10 | 0.998924 | 0.996496 |
| 7 | -131.9 | 19.48 | 10/10 | 30 | 0.20 | 0.984473 | 0.999229 |
| 8 | -132.2 | 9.99 | 10/10 | 30 | 0.20 | 0.955363 | 0.996858 |
| 9 | -133.3 | 22.28 | 10/10 | 20 | 0.03 | 0.958155 | 0.997789 |
| 10 | -136.1 | 1.58 | 10/10 | 20 | 0.10 | 0.997180 | 0.997980 |


---

# Mejor Configuración

## Hiperparámetros

```text
n_bins = 40
lr = 0.20
gamma = 0.979296
epsilon_decay = 0.995309
```

## Resultado

```text
Mean Reward = -126.8
Reached Flag =. 10/10
```

-
# Procesos del Q-learning

## Arquitectura General del Agente

Los pasos generales del agente son los siguientes:

1. (position, velocity) 
2. discretización
3. (i,j)
4. Q-table
5. Acción


## Discretización del Espacio de Estados

1. Se recibe la posición y la velocidd continua
2. Se normalizan
3. Se divide en el número de bins definidos. 
4. Se buscan los indices dentro de la qtable para la posición y la velocidad. 
5. Se ajusta la información de la qtable. 

## Selección acción e-greedy

1. Se calcula el valor aleatorio. 
2.  Se valida si se va a tomar la mejor acción actual o se va a seleccionar aleatoriamente la accion
3. Se retorna la acción. 


## Entrenamiento opr Episodios

1. Iniciar Episodio
2. Resetear el entorno
3. Estado inicial
4. Ejecuta Pasos
5. Ejecuta hasta terminar (Ciclo)
6. Actualiza epsilon
7. Existen más episodios (Volver al inicio)












# Referencias

Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd Edition).

Gymnasium Documentation:
https://gymnasium.farama.org/

Repositorio base:
https://github.com/emiliomunozai/mountain_car