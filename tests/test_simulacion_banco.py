# -*- coding: utf-8 -*-
"""Pruebas de la simulación del banco: datos del enunciado, lógica de colas y reproducibilidad."""
import numpy as np
import pytest

import simulacion_banco as sb


def test_probabilidades_suman_uno():
    assert sb.PROB_CLASE.sum() == pytest.approx(1.0)
    assert sb.PROB_CLASE[sb.ES_RETIRO].sum() == pytest.approx(0.70)  # 70 % retiros
    for acc in sb.ACCIONES:
        assert sum(sb.PROB_TIPO[acc]) == pytest.approx(1.0)


def test_tabla1_transcrita_bien():
    assert sb.MEDIA_SERVICIO[("Pago", "Muy lento")] == 7
    assert sb.MEDIA_LLEGADA[("Retiro", "Muy lento")] == 3   # el enunciado repite 3 en retiro lento y muy lento
    assert sb.MEDIA_LLEGADA[("Pago", "Muy lento")] == 4


def test_dia_ordenado_y_dentro_de_la_jornada():
    d = sb.generar_dia(np.random.default_rng(1))
    assert np.all(np.diff(d.llegada) > 0)
    assert d.llegada.max() <= sb.JORNADA_MIN
    assert len(d.llegada) == len(d.clase) == len(d.servicio)


def test_media_de_llegada_y_servicio_coinciden_con_teoria():
    rng = np.random.default_rng(7)
    n = [len(sb.generar_dia(rng).llegada) for _ in range(300)]
    esperado = sb.JORNADA_MIN / (sb.PROB_CLASE @ sb.LLEG_CLASE)   # ≈ 200 usuarios por día
    assert np.mean(n) == pytest.approx(esperado, rel=0.03)


def test_una_caja_fifo_a_mano():
    # Tres usuarios: llegan a 0, 1 y 10; servicios de 5, 3 y 2 minutos, una sola caja mixta.
    d = sb.Dia(np.array([0.0, 1.0, 10.0]), np.array([0, 0, 0]), np.array([5.0, 3.0, 2.0]))
    df = sb.simular_cajas(d, ["M"], np.random.default_rng(0))
    assert df["inicio"].tolist() == [0.0, 5.0, 10.0]
    assert df["espera"].tolist() == [0.0, 4.0, 0.0]
    assert df["salida"].tolist() == [5.0, 8.0, 12.0]


def test_cajas_exclusivas_respetan_el_tipo():
    d = sb.generar_dia(np.random.default_rng(3))
    df = sb.simular_cajas(d, ["R", "P", "P"], np.random.default_rng(3))
    assert set(df[df.es_retiro]["caja"]) == {0}          # los retiros solo van a la caja 1
    assert set(df[~df.es_retiro]["caja"]) <= {1, 2}      # los pagos solo a las cajas 2 y 3


def test_sin_solapes_en_la_misma_caja():
    d = sb.generar_dia(np.random.default_rng(5))
    df = sb.simular_cajas(d, ["M", "M", "M"], np.random.default_rng(5))
    for _, g in df.groupby("caja"):
        g = g.sort_values("inicio")
        assert np.all(g["inicio"].to_numpy()[1:] >= g["salida"].to_numpy()[:-1] - 1e-9)
    assert (df["espera"] >= 0).all()


def test_todos_son_atendidos_y_conservacion():
    d = sb.generar_dia(np.random.default_rng(9))
    df = sb.simular_cajas(d, ["R", "R", "P"], np.random.default_rng(9))
    assert len(df) == len(d.llegada)
    assert df["servicio"].sum() == pytest.approx(d.servicio.sum())


def test_reproducible_con_la_misma_semilla():
    _, a = sb.correr_replicas(3, seed=11)
    _, b = sb.correr_replicas(3, seed=11)
    assert a["4 mixtas"][2].equals(b["4 mixtas"][2])


def test_misma_gente_en_todas_las_configuraciones():
    dias, res = sb.correr_replicas(2, seed=4)
    for nombre in sb.CONFIGS:
        assert res[nombre][0]["llegada"].tolist() == dias[0].llegada.tolist()


def test_cuarta_caja_reduce_la_espera():
    _, res = sb.correr_replicas(30, seed=21)
    tab = sb.tabla_replicas(res)
    assert tab[tab.config == "4 mixtas"]["Wq"].mean() < tab[tab.config == "3 mixtas (actual)"]["Wq"].mean()


def test_simulacion_cercana_a_la_teoria_pollaczek_khinchine():
    _, res = sb.correr_replicas(300, seed=2, configs={"3 mixtas (actual)": ["M", "M", "M"]})
    sim = sb.tabla_replicas(res, {"3 mixtas (actual)": ["M", "M", "M"]})["Wq"].mean()
    teo = sb.teoria_mm1_por_caja(["M", "M", "M"])["Wq_teorico"].mean()
    assert sim == pytest.approx(teo, rel=0.15)


def test_replicas_necesarias_crece_si_se_pide_mas_precision():
    x = [4.1, 5.0, 3.2, 6.1, 4.4, 5.5, 3.9, 4.8, 5.2, 4.0]
    assert sb.replicas_necesarias(x, 0.25) > sb.replicas_necesarias(x, 0.5) >= len(x) // 2


def test_resumen_ic_contiene_la_media():
    import pandas as pd
    m, s, lo, hi = sb.resumen_ic(pd.Series([1.0, 2.0, 3.0, 4.0]))
    assert lo < m < hi and m == pytest.approx(2.5)
