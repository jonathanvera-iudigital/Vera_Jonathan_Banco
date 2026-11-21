# -*- coding: utf-8 -*-
"""Simulación de las cajas de un banco (colas M/M/1) — Actividad didáctica 2-M3.

Pregunta del banco: con 3 cajas, ¿conviene dedicar 1 a retiros y 2 a pagos, o al revés?
¿Hace falta una cuarta caja?

MODELO (resumen; el detalle está en el README y en el informe)
  * Un día = 480 min (8 h). Las llegadas se generan hasta el minuto 480; los usuarios que ya
    están en el banco a esa hora se terminan de atender (el banco no echa a nadie).
  * Cada usuario: acción (70 % retiro / 30 % pago) y tipo (rápido, normal, lento, muy lento) con
    las probabilidades de la Tabla 2. El tiempo desde la llegada anterior es Exponencial con la
    media de llegada de su tipo (Tabla 1) y su tiempo de servicio es Exponencial con la media de
    servicio de su tipo (Tabla 1).
  * Cada caja es una cola M/M/1 independiente (FIFO, un servidor). Velocidad de atención igual
    para todas; el desplazamiento es despreciable.
  * Configuraciones comparadas sobre EXACTAMENTE los mismos usuarios de cada réplica (números
    aleatorios comunes), de modo que las diferencias se deben a la configuración y no al azar.

Uso:  python simulacion_banco.py <comando> [--seed N] [--replicas N] [--salida carpeta]
Comandos: demo, todo.
"""
import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")  # figuras a archivo, sin ventana
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ===============================
# Parámetros del enunciado
# ===============================
JORNADA_MIN = 480.0   # 8 horas de operación
P_RETIRO = 0.70       # 70 % de los usuarios hace retiros; el resto, consignaciones o pagos
SEED = 42
N_REPLICAS = 10       # mínimo exigido por el enunciado
OUT_DIR = "resultados"

TIPOS = ["Rápido", "Normal", "Lento", "Muy lento"]
ACCIONES = ["Retiro", "Pago"]

# Tabla 1: (media de servicio, media de llegada) en minutos, por (acción, tipo).
MEDIA_SERVICIO = {
    ("Retiro", "Rápido"): 1, ("Retiro", "Normal"): 2, ("Retiro", "Lento"): 3, ("Retiro", "Muy lento"): 4,
    ("Pago", "Rápido"): 3, ("Pago", "Normal"): 3, ("Pago", "Lento"): 5, ("Pago", "Muy lento"): 7,
}
MEDIA_LLEGADA = {
    ("Retiro", "Rápido"): 1, ("Retiro", "Normal"): 2, ("Retiro", "Lento"): 3, ("Retiro", "Muy lento"): 3,
    ("Pago", "Rápido"): 1, ("Pago", "Normal"): 2, ("Pago", "Lento"): 3, ("Pago", "Muy lento"): 4,
}
# Tabla 2: probabilidad del tipo DENTRO de cada acción (cada acción suma 1).
PROB_TIPO = {
    "Retiro": [0.23, 0.40, 0.17, 0.20],
    "Pago":   [0.10, 0.20, 0.30, 0.40],
}

# Las 8 combinaciones (acción, tipo) en orden fijo; el índice 0..7 identifica la clase de usuario.
CLASES = [(a, t) for a in ACCIONES for t in TIPOS]
SERV_CLASE = np.array([MEDIA_SERVICIO[c] for c in CLASES], dtype=float)
LLEG_CLASE = np.array([MEDIA_LLEGADA[c] for c in CLASES], dtype=float)
ES_RETIRO = np.array([c[0] == "Retiro" for c in CLASES])
# Probabilidad conjunta de cada clase = P(acción) * P(tipo | acción).
PROB_CLASE = np.array([(P_RETIRO if a == "Retiro" else 1 - P_RETIRO) * PROB_TIPO[a][TIPOS.index(t)]
                       for a, t in CLASES])

# Configuraciones: nombre -> lista con la especialidad de cada caja ("R", "P" o "M" = mixta).
CONFIGS: Dict[str, List[str]] = {
    "3 mixtas (actual)":        ["M", "M", "M"],
    "1 retiros + 2 pagos":      ["R", "P", "P"],
    "2 retiros + 1 pago":       ["R", "R", "P"],
    "4 mixtas":                 ["M", "M", "M", "M"],
    "2 retiros + 2 pagos":      ["R", "R", "P", "P"],
    "1 retiros + 3 pagos":      ["R", "P", "P", "P"],
    "2 retiros + 1 pago + 1 mixta": ["R", "R", "P", "M"],
}


# ===============================
# Generación de un día (los usuarios)
# ===============================
@dataclass
class Dia:
    """Usuarios de un día: todo vectorizado, ordenado por hora de llegada."""
    llegada: np.ndarray    # minuto de llegada
    clase: np.ndarray      # índice 0..7 en CLASES
    servicio: np.ndarray   # minutos de servicio que necesitará

    @property
    def es_retiro(self) -> np.ndarray:
        return ES_RETIRO[self.clase]


def generar_dia(rng: np.random.Generator, jornada: float = JORNADA_MIN) -> Dia:
    """Genera los usuarios que llegan en [0, jornada].

    El hueco antes de cada usuario es Exponencial con la media de llegada de SU clase; se generan
    de más y se recorta en la jornada (el hueco depende de la clase, así que se sortea la clase primero).
    """
    llegadas, clases = [], []
    t = 0.0
    while True:
        k = int(rng.choice(len(CLASES), p=PROB_CLASE))
        t += rng.exponential(LLEG_CLASE[k])
        if t > jornada:
            break
        llegadas.append(t)
        clases.append(k)
    clase = np.array(clases, dtype=int)
    servicio = rng.exponential(SERV_CLASE[clase]) if len(clase) else np.array([])
    return Dia(np.array(llegadas), clase, servicio)


# ===============================
# Simulación de las cajas (M/M/1 en paralelo)
# ===============================
def simular_cajas(dia: Dia, especialidades: Sequence[str], rng: np.random.Generator,
                  enrutamiento: str = "aleatorio") -> pd.DataFrame:
    """Atiende un día con las cajas dadas y devuelve una fila por usuario.

    Cada caja es FIFO con un servidor: inicio = max(llegada, instante en que la caja se libera).
    Enrutamiento: el usuario elige entre las cajas que le sirven (R, P o mixta M):
      * "aleatorio": al azar entre ellas (cada caja es un M/M/1 independiente; es el modelo del enunciado)
      * "menor_cola": la caja con menos usuarios en ese momento (sensibilidad)
    """
    n_cajas = len(especialidades)
    libre = np.zeros(n_cajas)                       # instante en que cada caja queda libre
    pendientes: List[List[float]] = [[] for _ in range(n_cajas)]  # salidas aún por ocurrir (para menor_cola)
    filas = []
    for i in range(len(dia.llegada)):
        t, ret = dia.llegada[i], bool(dia.es_retiro[i])
        letra = "R" if ret else "P"
        validas = [j for j, e in enumerate(especialidades) if e in (letra, "M")]
        if enrutamiento == "menor_cola":
            def en_sistema(j):
                pendientes[j] = [s for s in pendientes[j] if s > t]
                return len(pendientes[j])
            carga = [en_sistema(j) for j in validas]
            minimos = [j for j, c in zip(validas, carga) if c == min(carga)]
            caja = int(rng.choice(minimos))
        else:
            caja = int(rng.choice(validas))
        inicio = max(t, libre[caja])
        salida = inicio + dia.servicio[i]
        libre[caja] = salida
        pendientes[caja].append(salida)
        filas.append((i, caja, t, inicio, salida))
    df = pd.DataFrame(filas, columns=["usuario", "caja", "llegada", "inicio", "salida"])
    df["clase"] = dia.clase
    df["es_retiro"] = dia.es_retiro
    df["servicio"] = dia.servicio
    df["espera"] = df["inicio"] - df["llegada"]
    df["sistema"] = df["salida"] - df["llegada"]
    return df


def metricas(df: pd.DataFrame, especialidades: Sequence[str], jornada: float = JORNADA_MIN) -> dict:
    """Métricas de una réplica: globales, por tipo de acción y por caja."""
    n_cajas = len(especialidades)
    m = {"usuarios": len(df),
         "Wq": df["espera"].mean(), "W": df["sistema"].mean(), "S": df["servicio"].mean(),
         "espera_max": df["espera"].max(),
         "pct_espera_mayor_5": 100 * (df["espera"] > 5).mean(),
         "cierre": df["salida"].max()}
    # L y Lq por la ley de Little sobre la jornada de apertura (Σ tiempos / duración)
    m["Lq"] = df["espera"].sum() / jornada
    m["L"] = df["sistema"].sum() / jornada
    for nombre, mask in (("ret", df["es_retiro"]), ("pag", ~df["es_retiro"])):
        sub = df[mask]
        m[f"Wq_{nombre}"] = sub["espera"].mean() if len(sub) else np.nan
        m[f"W_{nombre}"] = sub["sistema"].mean() if len(sub) else np.nan
        m[f"n_{nombre}"] = len(sub)
    ocupado = df.groupby("caja")["servicio"].sum()
    for j in range(n_cajas):
        sub = df[df["caja"] == j]
        m[f"caja{j+1}_n"] = len(sub)
        m[f"caja{j+1}_S"] = sub["servicio"].mean() if len(sub) else np.nan
        m[f"caja{j+1}_Wq"] = sub["espera"].mean() if len(sub) else np.nan
        m[f"caja{j+1}_W"] = sub["sistema"].mean() if len(sub) else np.nan
        m[f"caja{j+1}_rho"] = ocupado.get(j, 0.0) / jornada
    m["rho"] = np.mean([m[f"caja{j+1}_rho"] for j in range(n_cajas)])
    return m


# ===============================
# Experimento: réplicas de un día para todas las configuraciones
# ===============================
def correr_replicas(n_rep: int = N_REPLICAS, seed: int = SEED, enrutamiento: str = "aleatorio",
                    configs: Dict[str, List[str]] = CONFIGS):
    """Devuelve (dias, resultados): los días generados y {config: [DataFrame por réplica]}.

    Cada réplica usa un día propio (semilla = seed + r) y ese MISMO día se atiende con todas las
    configuraciones (números aleatorios comunes), así la comparación entre ellas es pareada.
    """
    dias = []
    resultados: Dict[str, List[pd.DataFrame]] = {c: [] for c in configs}
    for r in range(n_rep):
        rng_dia = np.random.default_rng([seed, r, 0])
        dia = generar_dia(rng_dia)
        dias.append(dia)
        for nombre, esp in configs.items():
            rng_ruta = np.random.default_rng([seed, r, 1])  # misma semilla de ruteo por réplica
            resultados[nombre].append(simular_cajas(dia, esp, rng_ruta, enrutamiento))
    return dias, resultados


def tabla_replicas(resultados: Dict[str, List[pd.DataFrame]], configs=CONFIGS) -> pd.DataFrame:
    """Una fila por (configuración, réplica) con todas las métricas."""
    filas = []
    for nombre, lista in resultados.items():
        for r, df in enumerate(lista, start=1):
            filas.append({"config": nombre, "replica": r, **metricas(df, configs[nombre])})
    return pd.DataFrame(filas)


def resumen_ic(serie: pd.Series, confianza: float = 0.95):
    """Media, desviación e intervalo de confianza t sobre las réplicas."""
    x = serie.dropna().to_numpy(float)
    n = len(x)
    media, s = x.mean(), x.std(ddof=1) if n > 1 else 0.0
    h = stats.t.ppf(0.5 + confianza / 2, n - 1) * s / math.sqrt(n) if n > 1 else 0.0
    return media, s, media - h, media + h


def replicas_necesarias(piloto: Sequence[float], error: float, confianza: float = 0.95) -> int:
    """Réplicas n tales que la semi-amplitud del IC sea ≤ error: n = (t·s/error)²  (con t de la piloto)."""
    x = np.asarray(piloto, float)
    s = x.std(ddof=1)
    n0 = len(x)
    n = n0
    for _ in range(50):  # iterar porque t depende de n
        n_nuevo = math.ceil((stats.t.ppf(0.5 + confianza / 2, max(n - 1, 1)) * s / error) ** 2)
        if n_nuevo == n:
            break
        n = max(n_nuevo, 2)
    return max(n, n0 if s == 0 else 2)


# ===============================
# Teoría para validar el simulador
# ===============================
def teoria_mm1_por_caja(config: Sequence[str]) -> pd.DataFrame:
    """Aproximación analítica: cada caja recibe Poisson-split de su flujo (M/G/1 con Pollaczek-Khinchine).

    Sirve como referencia ORDEN DE MAGNITUD: las llegadas del enunciado no son exactamente Poisson
    (el hueco depende del tipo), por eso la comparación con la simulación es aproximada.
    """
    tasa_clase = PROB_CLASE / (PROB_CLASE @ LLEG_CLASE)   # usuarios/min de cada clase
    filas = []
    n_R = config.count("R"); n_P = config.count("P"); n_M = config.count("M")
    for j, e in enumerate(config):
        lam = 0.0
        for k in range(len(CLASES)):
            ret = ES_RETIRO[k]
            validas = n_M + (n_R if ret else n_P)
            if e == "M" or (e == "R" and ret) or (e == "P" and not ret):
                lam += tasa_clase[k] / validas
        if lam == 0:
            filas.append((j + 1, 0, 0, 0, 0)); continue
        # momentos del servicio (exponencial por clase) de los usuarios que llegan a esta caja
        pesos = np.array([tasa_clase[k] / (n_M + (n_R if ES_RETIRO[k] else n_P))
                          if (e == "M" or (e == "R" and ES_RETIRO[k]) or (e == "P" and not ES_RETIRO[k])) else 0.0
                          for k in range(len(CLASES))]) / lam
        es1 = (pesos * SERV_CLASE).sum()
        es2 = (pesos * 2 * SERV_CLASE ** 2).sum()          # E[S²] de una exponencial = 2·media²
        rho = lam * es1
        wq = lam * es2 / (2 * (1 - rho)) if rho < 1 else math.inf
        filas.append((j + 1, lam, rho, wq, wq + es1))
    return pd.DataFrame(filas, columns=["caja", "lambda_por_min", "rho", "Wq_teorico", "W_teorico"])


# ===============================
# Figuras y reportes
# ===============================
COLORES = {"3 mixtas (actual)": "#6b7280", "1 retiros + 2 pagos": "#d97706", "2 retiros + 1 pago": "#2563eb",
           "4 mixtas": "#16a34a", "2 retiros + 2 pagos": "#9333ea", "1 retiros + 3 pagos": "#dc2626",
           "2 retiros + 1 pago + 1 mixta": "#0891b2"}


def figura_comparacion(tab: pd.DataFrame, ruta: str, nombres=None):
    """Barras de espera media (con IC 95 %) por configuración, global y por tipo de acción."""
    nombres = nombres or list(CONFIGS)
    fig, ejes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    for ax, col, tit in zip(ejes, ["Wq", "Wq_ret", "Wq_pag"], ["Todos los usuarios", "Retiros", "Pagos"]):
        for i, n in enumerate(nombres):
            m, _, lo, hi = resumen_ic(tab[tab.config == n][col])
            ax.bar(i, m, yerr=[[m - lo], [hi - m]], color=COLORES.get(n, "#888"), capsize=4)
            ax.text(i, hi + 0.15, f"{m:.1f}", ha="center", fontsize=8)
        ax.set_xticks(range(len(nombres)))
        ax.set_xticklabels([n.replace(" + ", "\n+ ").replace(" (", "\n(") for n in nombres], fontsize=7)
        ax.set_title(tit)
        ax.grid(axis="y", alpha=0.3)
    ejes[0].set_ylabel("Espera promedio en cola (min)")
    fig.suptitle("Espera promedio por configuración (media de réplicas, IC 95 %)")
    fig.tight_layout()
    fig.savefig(ruta, dpi=140)
    plt.close(fig)


def figura_cajas(tab: pd.DataFrame, ruta: str):
    """Tiempo promedio de atención (servicio) por caja en la configuración actual."""
    n = "3 mixtas (actual)"
    fig, ax = plt.subplots(figsize=(6, 4))
    for j in (1, 2, 3):
        m, _, lo, hi = resumen_ic(tab[tab.config == n][f"caja{j}_S"])
        ax.bar(j, m, yerr=[[m - lo], [hi - m]], capsize=5, color=["#2563eb", "#16a34a", "#d97706"][j - 1])
        ax.text(j, hi + 0.05, f"{m:.2f}", ha="center")
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["Caja 1", "Caja 2", "Caja 3"])
    ax.set_ylabel("Tiempo promedio de atención (min)")
    ax.set_title("Tiempo de atención por caja (3 mixtas, 10 réplicas)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(ruta, dpi=140); plt.close(fig)


def figura_usuarios(conteos: pd.DataFrame, ruta: str):
    """Usuarios por clase en cada réplica (mapa de calor)."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    datos = conteos.to_numpy()
    im = ax.imshow(datos, cmap="Blues", aspect="auto")
    ax.set_xticks(range(datos.shape[1])); ax.set_xticklabels(conteos.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(datos.shape[0])); ax.set_yticklabels(conteos.index)
    for i in range(datos.shape[0]):
        for j in range(datos.shape[1]):
            ax.text(j, i, int(datos[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if datos[i, j] > datos.max() * 0.6 else "black")
    ax.set_title("Usuarios por tipo en cada réplica (tabla del punto 3)")
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(ruta, dpi=140); plt.close(fig)


def conteos_por_clase(dias: List[Dia]) -> pd.DataFrame:
    """Usuarios de cada una de las 8 clases en cada réplica (filas = réplicas)."""
    filas = []
    for d in dias:
        c = np.bincount(d.clase, minlength=len(CLASES))
        filas.append(c)
    cols = [f"{a} {t}" for a, t in CLASES]
    df = pd.DataFrame(filas, columns=cols, index=range(1, len(dias) + 1))
    df.index.name = "replica"
    return df


# ===============================
# Informe de los 5 puntos del enunciado
# ===============================
def umbral_aceptable() -> dict:
    """Criterios para decidir si el servicio es aceptable (supuesto del trabajo; el enunciado no los fija)."""
    return {"Wq_max": 5.0, "rho_max": 0.85, "espera_max_prom": 30.0}


def correr_todo(n_rep: int, seed: int, salida: str):
    os.makedirs(salida, exist_ok=True)
    dias, res = correr_replicas(n_rep, seed)
    tab = tabla_replicas(res)
    tab.to_csv(os.path.join(salida, "metricas_por_replica.csv"), index=False)
    conteos = conteos_por_clase(dias)
    conteos.to_csv(os.path.join(salida, "p3_usuarios_por_tipo_y_replica.csv"))

    # ----- Punto 1: tiempo promedio de atención por caja
    act = tab[tab.config == "3 mixtas (actual)"]
    p1 = []
    for j in (1, 2, 3):
        for var, etq in (("S", "Atención (servicio)"), ("Wq", "Espera en cola"), ("W", "Tiempo en el sistema")):
            m, s, lo, hi = resumen_ic(act[f"caja{j}_{var}"])
            p1.append({"caja": j, "medida": etq, "media": m, "desv": s, "ic95_inf": lo, "ic95_sup": hi})
        p1.append({"caja": j, "medida": "Usuarios por día",
                   **dict(zip(["media", "desv", "ic95_inf", "ic95_sup"], resumen_ic(act[f"caja{j}_n"])))})
        p1.append({"caja": j, "medida": "Utilización",
                   **dict(zip(["media", "desv", "ic95_inf", "ic95_sup"], resumen_ic(act[f"caja{j}_rho"])))})
    p1 = pd.DataFrame(p1)
    p1.to_csv(os.path.join(salida, "p1_cajas.csv"), index=False)

    # ----- Punto 2: promedio de usuarios por tipo (todas las cajas)
    p2 = pd.DataFrame({"promedio": conteos.mean(), "desv": conteos.std(ddof=1),
                       "esperado_teorico": PROB_CLASE * (JORNADA_MIN / (PROB_CLASE @ LLEG_CLASE))})
    p2.index.name = "tipo"
    p2.to_csv(os.path.join(salida, "p2_promedio_por_tipo.csv"))

    # ----- Punto 3: réplica con menos usuarios por tipo
    p3 = pd.DataFrame({"min": conteos.min(), "replica_min": conteos.idxmin(), "max": conteos.max(),
                       "replica_max": conteos.idxmax()})
    totales = conteos.sum(axis=1)
    p3.to_csv(os.path.join(salida, "p3_extremos_por_tipo.csv"))

    # ----- Puntos 4 y 5: comparar configuraciones
    cmp_cols = ["Wq", "W", "Lq", "L", "rho", "espera_max", "pct_espera_mayor_5", "Wq_ret", "Wq_pag"]
    filas = []
    for n in CONFIGS:
        sub = tab[tab.config == n]
        fila = {"config": n}
        for c in cmp_cols:
            m, _, lo, hi = resumen_ic(sub[c])
            fila[c], fila[c + "_lo"], fila[c + "_hi"] = m, lo, hi
        filas.append(fila)
    cmp = pd.DataFrame(filas)
    cmp.to_csv(os.path.join(salida, "p45_comparacion_configuraciones.csv"), index=False)

    # Comparación pareada contra la configuración actual (diferencia de Wq por réplica, IC t)
    base = tab[tab.config == "3 mixtas (actual)"].set_index("replica")["Wq"]
    pareada = []
    for n in CONFIGS:
        if n == "3 mixtas (actual)":
            continue
        d = tab[tab.config == n].set_index("replica")["Wq"] - base
        m, _, lo, hi = resumen_ic(d)
        pareada.append({"config": n, "dif_Wq_vs_actual": m, "ic95_inf": lo, "ic95_sup": hi})
    pd.DataFrame(pareada).to_csv(os.path.join(salida, "p45_pareada_vs_actual.csv"), index=False)

    # Réplicas necesarias para estimar Wq (actual) con error ±0,5 min al 95 %
    n_nec = {n: replicas_necesarias(tab[tab.config == n]["Wq"], 0.5) for n in CONFIGS}
    pd.Series(n_nec, name="replicas_para_error_0.5min").to_csv(os.path.join(salida, "replicas_necesarias.csv"))

    # Sensibilidad: menor_cola y validación con 1000 réplicas
    _, res_mc = correr_replicas(n_rep, seed, "menor_cola")
    tab_mc = tabla_replicas(res_mc)
    tab_mc.to_csv(os.path.join(salida, "sensibilidad_menor_cola.csv"), index=False)
    _, res_1k = correr_replicas(1000, seed + 1)
    tab_1k = tabla_replicas(res_1k)
    val = tab_1k.groupby("config")[["Wq", "Wq_ret", "Wq_pag", "W", "Lq", "L", "rho", "pct_espera_mayor_5",
                                    "espera_max", "cierre", "usuarios"]].mean()
    val = val.loc[list(CONFIGS)]
    val.to_csv(os.path.join(salida, "validacion_1000_replicas.csv"))
    # ¿Las diferencias entre cajas idénticas son más que azar? ANOVA sobre las 10 réplicas (punto 1)
    anova = stats.f_oneway(*[act[f"caja{j}_S"] for j in (1, 2, 3)])
    pd.Series({"F": anova.statistic, "p_valor": anova.pvalue}, name="anova_servicio_entre_cajas").to_csv(
        os.path.join(salida, "p1_anova.csv"))
    teo = {n: teoria_mm1_por_caja(esp) for n, esp in CONFIGS.items()}
    pd.concat(teo, names=["config"]).reset_index(level=0).to_csv(
        os.path.join(salida, "teoria_por_caja.csv"), index=False)

    # Figuras
    figura_comparacion(tab, os.path.join(salida, "comparacion_configuraciones.png"),
                       ["3 mixtas (actual)", "1 retiros + 2 pagos", "2 retiros + 1 pago", "4 mixtas",
                        "2 retiros + 2 pagos"])
    figura_cajas(tab, os.path.join(salida, "p1_atencion_por_caja.png"))
    figura_usuarios(conteos, os.path.join(salida, "p3_usuarios_por_tipo.png"))
    return dict(dias=dias, tab=tab, conteos=conteos, p1=p1, p2=p2, p3=p3, totales=totales, cmp=cmp,
                pareada=pd.DataFrame(pareada), n_nec=n_nec, tab_mc=tab_mc, val=val, teo=teo, anova=anova)


def imprimir(r: dict):
    pd.set_option("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.3f}".format)
    print("\n== Punto 1: tiempo por caja (3 mixtas, media de réplicas) ==")
    print(r["p1"].pivot(index="caja", columns="medida", values="media"))
    print("\n== Punto 2: promedio de usuarios por tipo ==")
    print(r["p2"])
    print("\n== Punto 3: usuarios por tipo y réplica ==")
    print(r["conteos"])
    print(r["p3"])
    print("\nTotal de usuarios por réplica:", r["totales"].tolist())
    print("\n== Puntos 4-5: comparación de configuraciones ==")
    print(r["cmp"][["config", "Wq", "Wq_ret", "Wq_pag", "W", "Lq", "rho", "espera_max", "pct_espera_mayor_5"]])
    print("\nDiferencia pareada de Wq frente a la actual:")
    print(r["pareada"])
    print("\nRéplicas necesarias (error ±0,5 min, 95 %):", r["n_nec"])
    print(f"\nANOVA del tiempo de atención entre cajas: F = {r['anova'].statistic:.3f}, p = {r['anova'].pvalue:.3f}")
    print("\n== Validación con 1000 réplicas (media) ==")
    print(r["val"])
    print("\n== Sensibilidad: enrutamiento 'menor_cola' (10 réplicas, Wq medio) ==")
    print(r["tab_mc"].groupby("config")[["Wq", "Wq_ret", "Wq_pag"]].mean().loc[list(CONFIGS)])


def demo(seed: int):
    """Un día narrado, usuario por usuario, con la configuración actual (primeros 15 usuarios)."""
    dia = generar_dia(np.random.default_rng([seed, 0, 0]))
    df = simular_cajas(dia, ["M", "M", "M"], np.random.default_rng([seed, 0, 1]))
    print(f"Día de ejemplo: {len(df)} usuarios.  (llegada, caja, inicio, salida en minutos)")
    for _, f in df.head(15).iterrows():
        a, t = CLASES[int(f.clase)]
        print(f"  #{int(f.usuario)+1:>3} {a:<6}{t:<10} llega {f.llegada:6.1f}  caja {int(f.caja)+1}  "
              f"inicia {f.inicio:6.1f}  espera {f.espera:5.1f}  sale {f.salida:6.1f}")
    m = metricas(df, ["M", "M", "M"])
    print(f"\nEspera media {m['Wq']:.2f} min | tiempo en sistema {m['W']:.2f} min | utilización {m['rho']:.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("comando", choices=["demo", "todo"])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--replicas", type=int, default=N_REPLICAS)
    ap.add_argument("--salida", default=OUT_DIR)
    a = ap.parse_args(argv)
    if a.comando == "demo":
        demo(a.seed)
    else:
        imprimir(correr_todo(a.replicas, a.seed, a.salida))


if __name__ == "__main__":
    main()
