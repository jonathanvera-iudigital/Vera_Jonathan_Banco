# Cajas de un banco: simulación de colas M/M/1

Actividad didáctica 2-M3: el banco tiene 3 cajas para retiros y pagos y quiere saber si debe dedicar
**1 caja a retiros y 2 a pagos, o al revés**, y si necesita una cuarta. Se simula un día de 8 horas con
usuarios aleatorios, se repite en réplicas y se comparan las configuraciones con las mismas llegadas.


## Cómo ejecutarlo

```bash
pip install -r requirements.txt
python -m pytest -q                      # 14 pruebas
python simulacion_banco.py demo          # un día narrado, usuario por usuario
python simulacion_banco.py todo          # los 5 puntos + 1.000 réplicas (≈ 1 min); escribe resultados/
python docs/generar_informe.py           # regenera el informe .docx con las cifras de resultados/
```

Opciones: `--seed 42` (por defecto), `--replicas 10` (por defecto, el mínimo del enunciado), `--salida resultados`.
Con la misma semilla los resultados se repiten.

## Modelo

| Elemento | Decisión |
|---|---|
| Jornada | 480 min. Llegadas hasta el minuto 480; quien está dentro se termina de atender |
| Usuario | acción (70 % retiro, 30 % pago) y tipo (Tabla 2, condicional a la acción) → 8 clases |
| Llegada | el hueco antes de cada usuario es Exp(media de llegada de su clase), Tabla 1 |
| Servicio | Exp(media de servicio de su clase), Tabla 1; igual velocidad en todas las cajas |
| Caja | M/M/1 independiente, FIFO, un servidor; desplazamiento despreciable |
| Reparto | al azar entre las cajas que le sirven (sensibilidad: la de menor cola) |
| Comparación | cada réplica genera un día y **ese mismo día** se atiende con todas las configuraciones |

**Interpretación de las tablas.** Ocho flujos de Poisson independientes darían 4,25 llegadas/min (sobrecarga
total y la Tabla 2 sobraría). Se sortea la clase y luego el tiempo hasta su llegada con la media de esa clase:
2,40 min entre llegadas (~200 usuarios/día), servicio medio 3,20 min, 1,33 cajas de trabajo sobre 3.
Como el hueco depende de la clase, la llegada no es Poisson puro; la teoría (Pollaczek-Khinchine) se usa solo
como referencia aproximada.

## Resultados (semilla 42)

Tablas y figuras completas en [`resultados/`](resultados/); consola en `resultados/consola.txt`.

**1. Tiempo de atención por caja (3 mixtas, 10 réplicas).** Caja 2: 3,37 min (menor), caja 3: 3,47 min
(mayor). La diferencia no es significativa (ANOVA p = 0,92): las cajas son idénticas y las diferencias son azar.

**2. Promedio de usuarios por día** (`p2_promedio_por_tipo.csv`): 32,5 retiros rápidos, 54,0 normales, 25,4
lentos, 25,3 muy lentos; 5,5 pagos rápidos, 12,6 normales, 18,2 lentos, 24,8 muy lentos (≈ 198 por día).

**3. Usuarios por tipo y réplica** (`p3_usuarios_por_tipo_y_replica.csv`, `p3_extremos_por_tipo.csv`): totales
de 174 (réplica 9, la de menos usuarios) a 212. Mínimo por tipo en la columna `replica_min`.

**Réplicas necesarias** (`replicas_necesarias.csv`): para estimar la espera con ±0,5 min (95 %) hacen falta
~97 réplicas (3 mixtas), 74 (1 R + 2 P) y 168 (2 R + 1 P). Con 10 solo se separan las configuraciones de
cuatro cajas, por eso las decisiones usan **1.000 réplicas** (`validacion_1000_replicas.csv`):

| Configuración | Espera media | Retiros | Pagos | Esperan > 5 min | Utilización |
|---|---|---|---|---|---|
| 3 mixtas (actual) | 3,41 | 3,49 | 3,23 | 22,0 % | 45 % |
| 1 retiros + 2 pagos | 5,09 | 6,10 | 2,59 | 32,6 % | 45 % |
| 2 retiros + 1 pago | 3,86 | 1,60 | 8,93 | 21,0 % | 45 % |
| 4 mixtas | 2,16 | 2,21 | 2,04 | 14,5 % | 33 % |
| 2 retiros + 2 pagos | 1,91 | 1,58 | 2,62 | 12,9 % | 33 % |

**4. ¿Cuarta caja?** Con 3 mixtas no es indispensable (espera 3,4 min, ocupación 45 %; criterio propio: ≤ 5 min),
aunque una cuarta recorta la espera 37 %. Si se especializan las cajas, sí hace falta: con 3 cajas exclusivas
un tipo de usuario supera los 5 min (retiros 6,1 o pagos 8,9).

**5. ¿Cuántas exclusivas?** Entre las dos opciones del banco, **2 retiros + 1 pago** (espera 3,86 frente a
5,09 min); el costo es que los pagos esperan 8,9 min. Con una cuarta caja, la mejor es 2 + 2 (1,91 min).
Ninguna de las dos opciones con 3 cajas supera a las 3 mixtas con reparto aleatorio (3,41 min).

**Sensibilidad.** Si los usuarios de las cajas mixtas eligen la caja con menos gente, la espera con 3 mixtas
baja a ~1,1 min: el problema de las mixtas es el reparto aleatorio, no el número de cajas.

**Validación.** Teoría de colas (M/G/1 por caja): 3 mixtas 3,39 min frente a 3,41 simulados; 1 R + 2 P: 5,06
frente a 5,09. Las 14 pruebas cubren datos del enunciado, una caja FIFO calculada a mano, cajas exclusivas,
reproducibilidad y cercanía a la teoría.

## Supuestos y límites

- El enunciado no fija un criterio de servicio aceptable; se usó espera media ≤ 5 min y utilización ≤ 85 %.
- La lectura de las tablas de llegada (arriba) es una interpretación razonada del enunciado.
- No se modelan costos por caja ni demanda variable durante el día.

## Estructura

```
simulacion_banco.py        modelo, experimentos y reporte (comentado)
tests/                     14 pruebas con pytest
resultados/                CSV, figuras y consola
```
