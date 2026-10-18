import argparse
import logging
import re
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from bloques_1_2 import (FONDO, TEXTO, TEXTO2, AZUL, AZUL_CLARO, GRIS, CUADRICULA,
                         TERRACOTA, configurar_fuente, cargar_series)

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

# ------------------------------------------------------------------ paleta
JADE = "#2A7364"        # mediación: modelo ganador y estimación
JADE_CLARO = "#9CC5BB"  # mismo jade aclarado: error medio (MAE) del ganador
GRIS_CLARO = "#C9CFD6"  # gris aclarado: error medio (MAE) del resto

# Tamaño (pulgadas) del espacio de cada gráfica en el lienzo de 11 × 17 in
TAM_ESTACIONALIDAD = (10.0, 1.8)
TAM_CUADRANTE = (10.0, 1.7)
TAM_MODELOS = (4.7, 2.1)
TAM_PROYECCION = (5.1, 2.1)

MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
NOMBRES = {
    "Total_Viajeros": "Total",
    "Via_Aerea": "Vía aérea",
    "Via_Terrestre": "Vía terrestre",
    "Via_Maritima": "Vía marítima",
    "Pais_Guatemala": "Guatemala",
    "Pais_El Salvador": "El Salvador",
    "Pais_Estados Unidos de América": "EE. UU.",
}
MARCAS = {"Via_Maritima": "*", "Pais_Guatemala": "†"}   # llamadas al pie del cuadrante
SEMILLA = 42   # las bandas de Prophet se simulan: con semilla fija salen siempre iguales


def neg(texto: str) -> str:
    """Signo menos tipográfico en lugar del guion."""
    return texto.replace("-", "−")


def guardar(fig, salida: Path, nombre: str):
    salida.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(salida / f"{nombre}.{ext}", dpi=300, facecolor=FONDO)
    plt.close(fig)


# ------------------------------------------------------------------ datos
def leer_metricas_lab1(ruta: Path) -> pd.DataFrame:
    """Tablas de modelos por serie, tal como las imprimió el Lab 1 (detalle_por_serie.txt)."""
    num = r"([\d.]+(?:e[+-]?\d+)?)"
    patron = re.compile(rf"^\s*(.+?)\s+{num}\s+{num}\s+{num}\s+(NaN|[\d.]+)\s+(NaN|[\d.]+)\s*$")
    filas, serie, en_tabla = [], None, False
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        if linea.startswith("SERIE:"):
            serie, en_tabla = linea.split(":", 1)[1].strip(), False
        elif linea.startswith("4.h-4.j"):
            en_tabla = True
        elif linea.startswith("4.k"):
            en_tabla = False
        elif en_tabla and (m := patron.match(linea)):
            modelo, mae, rmse, mape, aic, bic = m.groups()
            filas.append((serie, modelo, float(mae), float(rmse), float(mape),
                          np.nan if aic == "NaN" else float(aic),
                          np.nan if bic == "NaN" else float(bic)))
    return pd.DataFrame(filas, columns=["serie", "modelo", "MAE", "RMSE", "MAPE_%", "AIC", "BIC"])


def indice_estacional(serie: pd.Series) -> pd.Series:
    """Promedio de cada mes 2009–2019 sobre el promedio general (mes promedio = 100)."""
    pre = serie[:"2019-12"]
    return pre.groupby(pre.index.month).mean() / pre.mean() * 100


def tabla_cuadrante(s: pd.DataFrame, resumen: pd.DataFrame) -> pd.DataFrame:
    """Las cuatro medidas del inciso 5 del Lab 1, una fila por serie."""
    a2019 = s[s.index.year == 2019].mean()
    return pd.DataFrame({
        "Fs": resumen["F_estacional"],                            # fuerza estacional (STL)
        "Ft": resumen["F_tendencia"],                             # fuerza de tendencia (STL)
        "CV": s.std() / s.mean() * 100,                           # volatilidad: desv./media
        "Caida": (a2019 - s.loc["2020-04-01"]) / a2019 * 100,     # abril 2020 vs. mes prom. 2019
    }).loc[s.columns]


def ajustar_prophet(train: pd.Series, test: pd.Series) -> pd.DataFrame:
    """Prophet con la configuración del Lab 1; bandas de 80 y 95 % de la simulación posterior."""
    from prophet import Prophet
    pr = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                 seasonality_mode="multiplicative")
    pr.fit(pd.DataFrame({"ds": train.index, "y": train.values}))
    futuro = pr.make_future_dataframe(periods=len(test), freq="MS")
    pred = pr.predict(futuro).tail(len(test))
    np.random.seed(SEMILLA)
    muestras = pr.predictive_samples(futuro)["yhat"][-len(test):]
    out = pd.DataFrame({"yhat": pred["yhat"].values}, index=test.index)
    for nivel in (80, 95):
        cola = (100 - nivel) / 2
        out[f"lo{nivel}"], out[f"hi{nivel}"] = np.percentile(muestras, [cola, 100 - cola], axis=1)
    return out


# ------------------------------------------------------------------ Bloque 3a
def grafica_estacionalidad(s: pd.DataFrame, salida: Path, tam=TAM_ESTACIONALIDAD):
    """Barras del índice estacional del total (izq.) y puntos aérea vs. terrestre (der.)."""
    w, h = tam
    idx = indice_estacional(s["Total_Viajeros"])
    aerea = indice_estacional(s["Via_Aerea"])
    terr = indice_estacional(s["Via_Terrestre"])

    fig = plt.figure(figsize=tam)
    fig.patch.set_facecolor(FONDO)
    ax = fig.add_axes([0.04, 0.12, 0.58, 0.68])
    ax2 = fig.add_axes([0.70, 0.12, 0.28, 0.655])
    for a in (ax, ax2):
        a.set_facecolor(FONDO)
        for lado in ("top", "right", "left"):
            a.spines[lado].set_visible(False)
        a.spines["bottom"].set_color(GRIS)

    fig.text(0.04, 0.95, "Índice estacional del total de ingresos, 2009–2019 (mes promedio = 100)",
             fontsize=10.5, fontweight="bold", color=TEXTO, va="center")

    # ---- barras: desviación respecto al mes promedio ----
    meses = np.arange(1, 13)
    desv = idx.values - 100
    colores = [AZUL if v >= 103 else GRIS for v in idx.values]
    ax.bar(meses, desv, bottom=100, width=0.68, color=colores, zorder=3)
    ax.axhline(100, color=TEXTO2, lw=1.0, zorder=4)
    ax.set_ylim(68, 160)
    ax.set_xlim(0.4, 12.6)
    ax.set_xticks(meses)
    ax.set_xticklabels(MESES)
    ax.set_yticks([80, 100, 120])
    ax.grid(axis="y", color=CUADRICULA, lw=0.8, zorder=0)
    ax.tick_params(axis="x", labelsize=10.5, colors=TEXTO2, length=0, pad=4)
    ax.tick_params(axis="y", labelsize=10, colors=TEXTO2, length=0)
    CLAVE = {2, 3, 4, 7, 9, 12}   # solo se rotulan los meses que cuentan la historia
    for m, v in zip(meses, idx.values):
        if m not in CLAVE:
            continue
        pico = v >= 103
        ax.text(m, v + 1.5 if v >= 100 else v - 1.5, neg(f"{v - 100:+.0f} %"), ha="center",
                va="bottom" if v >= 100 else "top", fontsize=10,
                fontweight="bold" if pico or v < 85 else "normal",
                color=TEXTO if pico or v < 85 else TEXTO2)

    def evento(x, y, texto, ha="center"):
        ax.text(x, y, texto, ha=ha, va="bottom", fontsize=10.5, fontweight="bold",
                color=AZUL, linespacing=1.2)

    evento(3.5, 121, "Semana Santa\n(mar–abr)")
    evento(7.3, 121, "Vacaciones\nBono 14 (jul)")
    evento(11.15, 118, "Aguinaldo,\nNavidad y\nfin de año", ha="right")

    # ---- puntos: ¿qué vía sube en cada pico? ----
    filas = [("Semana Santa (mar–abr)", (aerea[3] + aerea[4]) / 2, (terr[3] + terr[4]) / 2),
             ("Vacaciones (jul)", aerea[7], terr[7]),
             ("Fin de año (dic)", aerea[12], terr[12])]
    sep = 1.6
    ax2.set_xlim(78, 148)
    ax2.set_ylim(-0.8, sep * (len(filas) - 1) + 0.55)
    ax2.invert_yaxis()
    ax2.set_yticks([])
    ax2.set_xticks([80, 100, 120, 140])
    ax2.tick_params(axis="x", labelsize=10, colors=TEXTO2, length=0, pad=4)
    ax2.axvline(100, color=TEXTO2, lw=1.0, zorder=2)
    ax2.grid(axis="x", color=CUADRICULA, lw=0.8, zorder=0)
    ax2.set_axisbelow(True)
    for i, (nombre, va_, vt) in enumerate(filas):
        y = i * sep + 0.3
        ax2.text(78.5, y - 0.36, nombre, fontsize=10, color=TEXTO, va="bottom", ha="left", zorder=6,
                 bbox=dict(facecolor=FONDO, edgecolor="none", pad=0.8))
        ax2.plot([min(va_, vt), max(va_, vt)], [y] * 2, color=GRIS_CLARO, lw=2.2, zorder=3)
        ax2.scatter([va_], [y], s=50, color=AZUL, zorder=5)
        ax2.scatter([vt], [y], s=50, color=JADE, marker="D", zorder=5)
        izq_es_aerea = va_ < vt
        ax2.text(min(va_, vt) - 2.4, y, f"{(va_ if izq_es_aerea else vt):.0f}",
                 ha="right", va="center", fontsize=10, fontweight="bold",
                 color=AZUL if izq_es_aerea else JADE)
        ax2.text(max(va_, vt) + 2.4, y, f"{(vt if izq_es_aerea else va_):.0f}",
                 ha="left", va="center", fontsize=10, fontweight="bold",
                 color=JADE if izq_es_aerea else AZUL)

    fig.text(0.70, 0.95, "¿Qué vía empuja cada pico?", fontsize=10.5, fontweight="bold",
             color=TEXTO, va="center")
    fig.text(0.70, 0.835, "●", fontsize=10, color=AZUL, va="center")
    fig.text(0.717, 0.835, "aérea", fontsize=10, color=AZUL, fontweight="bold", va="center")
    fig.text(0.79, 0.835, "◆", fontsize=10, color=JADE, va="center")
    fig.text(0.807, 0.835, "terrestre", fontsize=10, color=JADE, fontweight="bold", va="center")

    guardar(fig, salida, "bloque3_estacionalidad")
    return idx, aerea, terr


# ------------------------------------------------------------------ Bloque 3b
MOSTRAR = ["Via_Aerea", "Via_Terrestre", "Pais_El Salvador", "Pais_Estados Unidos de América"]


def grafica_cuadrante(q: pd.DataFrame, salida: Path, tam=TAM_CUADRANTE):
    """Tabla visual con barras: una columna por pregunta, una fila por serie comparable.

    El ganador se decide sobre las series elegibles (todas menos el total y la vía marítima).
    Se dibujan solo las series comparables (MOSTRAR); la marítima y Guatemala, cuyas series se
    cortan, se mencionan en la nota al pie con su cifra."""
    w, h = tam
    q = q.drop(index="Total_Viajeros")
    fig = plt.figure(figsize=tam)
    fig.patch.set_facecolor(FONDO)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis("off")

    elegibles = [c for c in q.index if c != "Via_Maritima"]
    filas = [c for c in q.index if c in MOSTRAR]
    columnas = [  # (columna, pregunta, medida, color del ganador, formato del valor)
        ("Fs", "¿La más estacional?", "fuerza estacional (0 a 1)", AZUL, lambda v: f"{v:.2f}"),
        ("Ft", "¿La que más crece?", "fuerza de tendencia (0 a 1)", AZUL, lambda v: f"{v:.2f}"),
        ("CV", "¿La más volátil?", "variación sobre la media", AZUL, lambda v: f"{v:.0f} %"),
        ("Caida", "¿La más golpeada en pandemia?", "caída de abr 2020 vs. 2019", TERRACOTA, lambda v: f"{v:.1f} %"),
    ]
    x_nom, x0, ancho_col, ancho_barra = 0.05, 1.55, 1.9, 1.2
    y_fila0, alto_fila = 0.60, 0.205
    escala = {"Fs": 1.0, "Ft": 1.0, "CV": 100.0, "Caida": 100.0}   # todas parten de cero

    resp = {}
    for j, (col, pregunta, medida, c_gan, fmt) in enumerate(columnas):
        x = x0 + j * ancho_col
        top = q.loc[elegibles, col].max()
        ganadores = [c for c in elegibles if q.loc[c, col] >= top - 1e-9]
        resp[col] = ganadores
        nombres = " y ".join(NOMBRES[c] for c in ganadores)
        ax.text(x, 0.10, pregunta, fontsize=10, color=TEXTO2, va="center")
        ax.text(x, 0.285, nombres, fontsize=12.5, fontweight="bold", color=c_gan, va="center")
        ax.text(x, 0.45, medida, fontsize=9.5, color=TEXTO2, va="center")
        for i, c in enumerate(filas):
            y = y_fila0 + i * alto_fila + alto_fila / 2
            v = q.loc[c, col]
            gana = c in ganadores
            color = c_gan if gana else AZUL_CLARO
            largo = max(v / escala[col], 0) * ancho_barra
            ax.add_patch(Rectangle((x, y - 0.05), largo, 0.10, color=color, lw=0))
            ax.text(x + largo + 0.07, y, neg(fmt(v)), fontsize=10, va="center",
                    fontweight="bold" if gana else "normal", color=TEXTO if gana else TEXTO2)

    for i, c in enumerate(filas):
        y = y_fila0 + i * alto_fila + alto_fila / 2
        ax.text(x_nom, y, NOMBRES[c], fontsize=10.5, va="center", color=TEXTO)
    yl = y_fila0 + 2 * alto_fila   # separador entre vías y países
    ax.plot([x_nom, w - 0.05], [yl, yl], color=CUADRICULA, lw=1.0)
    gan_ft = resp["Ft"][0]
    sig = q.loc[[c for c in elegibles if c != gan_ft], "Ft"].idxmax()
    ax.text(x_nom, 1.56, f"Marítima (serie hasta 2016; estacionalidad {q.loc['Via_Maritima', 'Fs']:.2f}) y Guatemala "
            f"(hasta 2022) no compiten: series cortadas. Tendencia: {NOMBRES[gan_ft].lower()} "
            f"{q.loc[gan_ft, 'Ft']:.2f} apenas supera a {NOMBRES[sig].lower()} {q.loc[sig, 'Ft']:.2f}.",
            fontsize=9, color=TEXTO2, va="center")

    guardar(fig, salida, "bloque3_cuadrante")
    return resp


# ------------------------------------------------------------------ Bloque 4a
def grafica_modelos(met: pd.DataFrame, salida: Path, tam=TAM_MODELOS):
    """Error de cada algoritmo sobre el test del total: RMSE (oscuro) y MAE (claro)."""
    t = met[met["serie"] == "Total_Viajeros"].sort_values("RMSE").reset_index(drop=True)
    etiquetas = {
        "Prophet": "Prophet",
        "Holt-Winters multiplicativo": "Holt-Winters multiplicativo",
        "Suavizamiento exp. simple": "Suavizamiento exp. simple",
        "Seasonal naive": "Seasonal naive",
        "Holt-Winters aditivo": "Holt-Winters aditivo",
    }
    w, h = tam
    fig = plt.figure(figsize=tam)
    fig.patch.set_facecolor(FONDO)
    ax = fig.add_axes([0.45, 0.05, 0.43, 0.69])
    ax.set_facecolor(FONDO)
    n = len(t)
    sarima = None
    for i, r in t.iterrows():
        gana = i == 0
        c_rmse, c_mae = (JADE, JADE_CLARO) if gana else (GRIS, GRIS_CLARO)
        ax.barh(i - 0.19, r["RMSE"] / 1e3, height=0.34, color=c_rmse, zorder=3)
        ax.barh(i + 0.19, r["MAE"] / 1e3, height=0.34, color=c_mae, zorder=3)
        t1 = ax.text(r["RMSE"] / 1e3 + 5, i, f"{r['RMSE'] / 1e3:.0f}", va="center", fontsize=9.5,
                     fontweight="bold", color=JADE if gana else TEXTO)
        ax.annotate(f" · {r['MAE'] / 1e3:.0f}", xy=(1, 0.5), xycoords=t1, va="center", ha="left",
                    fontsize=9.5, color=JADE if gana else TEXTO2)
        nombre = etiquetas.get(r["modelo"])
        if nombre is None:   # SARIMA: su AIC/BIC va en la nota al pie
            nombre = "SARIMA"
            sarima = (r["AIC"], r["BIC"])
        ax.text(-6, i, nombre, ha="right", va="center", fontsize=10,
                fontweight="bold" if gana else "normal", color=JADE if gana else TEXTO)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlim(0, 340)
    ax.set_yticks([])
    ax.set_xticks([])
    for lado in ("top", "right", "left", "bottom"):
        ax.spines[lado].set_visible(False)
    ax.axvline(0, color=TEXTO2, lw=1.0, zorder=4)

    fig.text(0.02, 0.94, "Error de pronóstico en el periodo de prueba", fontsize=10.5,
             fontweight="bold", color=TEXTO, va="center")
    fig.text(0.02, 0.835, "■", fontsize=10, color=GRIS, va="center")
    fig.text(0.055, 0.835, "RMSE", fontsize=9.5, color=TEXTO2, va="center")
    fig.text(0.16, 0.835, "■", fontsize=10, color=GRIS_CLARO, va="center")
    fig.text(0.195, 0.835, "MAE · miles de viajeros por mes", fontsize=9.5,
             color=TEXTO2, va="center")
    guardar(fig, salida, "bloque4_modelos")
    return t


# ------------------------------------------------------------------ Bloque 4b
def grafica_proyeccion(s: pd.DataFrame, f: pd.DataFrame, salida: Path, tam=TAM_PROYECCION):
    """Predicción de Prophet sobre el test, con abanico de 80 y 95 % ('Estimación')."""
    w, h = tam
    T = s["Total_Viajeros"] / 1e3
    fig = plt.figure(figsize=tam)
    fig.patch.set_facecolor(FONDO)
    ax = fig.add_axes([0.075, 0.25, 0.905, 0.47])
    ax.set_facecolor(FONDO)

    f = (f / 1e3).clip(lower=0)   # no puede haber viajeros negativos
    inicio = f.index[0]
    ax.fill_between(f.index, f["lo95"], f["hi95"], color=JADE, alpha=0.14, lw=0, zorder=1)
    ax.fill_between(f.index, f["lo80"], f["hi80"], color=JADE, alpha=0.28, lw=0, zorder=2)
    ax.axvline(inicio, color=GRIS, lw=1, ls=(0, (3, 3)), zorder=1)
    ent = T["2019":][:inicio - pd.offsets.MonthBegin(1)]
    ax.plot(ent.index, ent.values, color=AZUL_CLARO, lw=1.4, zorder=3)
    real = T[inicio:]
    ax.plot(real.index, real.values, color=AZUL, lw=1.8, zorder=4)
    ax.plot(f.index, f["yhat"], color=JADE, lw=2.2, zorder=5)

    ax.set_ylim(0, 540)
    ax.set_xlim(pd.Timestamp("2018-12-01"), pd.Timestamp("2028-02-01"))
    ax.set_yticks([0, 200, 400])
    ax.set_xticks([pd.Timestamp(f"{a}-01-01") for a in (2020, 2022, 2024, 2026)])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", color=CUADRICULA, lw=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(GRIS)
    ax.tick_params(axis="both", labelsize=10, colors=TEXTO2, length=0, pad=4)
    ax.tick_params(axis="x", length=3, color=GRIS)

    # etiquetas directas
    fin = f.index[-1] + pd.Timedelta(days=30)
    ax.text(fin, real.iloc[-1], "Real", color=AZUL, fontweight="bold", fontsize=10, va="center")
    ax.text(fin, f["yhat"].iloc[-1], "Estimación", color=JADE, fontweight="bold", fontsize=10, va="center")
    ax.text(inicio - pd.Timedelta(days=40), 512, "Entrenamiento", ha="right", va="center",
            fontsize=10, color=TEXTO2)
    ax.text(inicio + pd.Timedelta(days=40), 512, "Prueba", ha="left", va="center", fontsize=10, color=TEXTO2)

    # encabezado de la gráfica
    fig.text(0.02, 0.94, "Modelo ganador, Prophet: real vs. estimación", fontsize=10.5,
             fontweight="bold", color=TEXTO, va="center")
    for x, alfa, txt in ((0.02, 0.28, "banda del 80 %"), (0.29, 0.14, "banda del 95 %")):
        fig.patches.append(Rectangle((x, 0.80), 0.03, 0.05, transform=fig.transFigure,
                                     facecolor=JADE, alpha=alfa + 0.05, lw=0))
        fig.text(x + 0.04, 0.825, txt, fontsize=9.5, color=JADE, fontweight="bold", va="center")
    fig.text(0.98, 0.825, "miles de viajeros por mes", fontsize=9.5, color=TEXTO2, va="center", ha="right")

    # lectura honesta: cuántos meses reales cayeron dentro de cada banda
    dentro = {n: np.mean((real.values >= f[f"lo{n}"].values) & (real.values <= f[f"hi{n}"].values))
              for n in (80, 95)}
    fig.text(0.02, 0.05, f"Solo {dentro[80] * 100:.0f} % de los meses reales cayó en la banda de 80 %.",
             fontsize=9.5, color=TEXTO, va="center", fontweight="bold")
    guardar(fig, salida, "bloque4_proyeccion")
    return dentro


# ------------------------------------------------------------------ cifras
def juntar_cifras(idx, aerea, terr, q, resp, met, tabla_total, dentro, rep, resumen) -> pd.DataFrame:
    f = []
    f.append(("B3", "Índice estacional de diciembre (mes promedio = 100)", idx[12], neg(f"{idx[12] - 100:+.0f} %")))
    f.append(("B3", "Índice estacional de febrero", idx[2], neg(f"{idx[2] - 100:+.0f} %")))
    f.append(("B3", "Índice estacional de septiembre", idx[9], neg(f"{idx[9] - 100:+.0f} %")))
    f.append(("B3", "Índice estacional de marzo", idx[3], neg(f"{idx[3] - 100:+.0f} %")))
    f.append(("B3", "Índice estacional de abril", idx[4], neg(f"{idx[4] - 100:+.0f} %")))
    f.append(("B3", "Índice estacional de julio", idx[7], neg(f"{idx[7] - 100:+.0f} %")))
    f.append(("B3", "Índice estacional de agosto", idx[8], neg(f"{idx[8] - 100:+.0f} %")))
    f.append(("B3", "Índice vía aérea, julio", aerea[7], f"{aerea[7]:.0f}"))
    f.append(("B3", "Índice vía terrestre, julio", terr[7], f"{terr[7]:.0f}"))
    f.append(("B3", "Índice vía aérea, diciembre", aerea[12], f"{aerea[12]:.0f}"))
    f.append(("B3", "Índice vía terrestre, diciembre", terr[12], f"{terr[12]:.0f}"))
    f.append(("B3", "Índice vía aérea, marzo–abril", (aerea[3] + aerea[4]) / 2, f"{(aerea[3] + aerea[4]) / 2:.0f}"))
    f.append(("B3", "Índice vía terrestre, marzo–abril", (terr[3] + terr[4]) / 2, f"{(terr[3] + terr[4]) / 2:.0f}"))
    f.append(("B3", "Índice vía aérea, agosto", aerea[8], f"{aerea[8]:.0f}"))
    f.append(("B3", "Índice vía terrestre, agosto", terr[8], f"{terr[8]:.0f}"))
    etiquetas = {"Fs": "más estacional", "Ft": "mayor tendencia", "CV": "más volátil", "Caida": "más golpeada por la pandemia"}
    for col, ganadores in resp.items():
        for c in ganadores:
            f.append(("B3", f"Serie {etiquetas[col]}", q.loc[c, col], NOMBRES[c]))
    for c in q.index:
        f.append(("B3", f"{NOMBRES[c]}: fuerza estacional / tendencia / CV % / caída abr 2020 %",
                  q.loc[c, "Fs"], f"{q.loc[c, 'Fs']:.2f} / {q.loc[c, 'Ft']:.2f} / {q.loc[c, 'CV']:.1f} / {q.loc[c, 'Caida']:.1f}"))
    ganadores_modelo = tabla_total.iloc[0]
    f.append(("B4", f"RMSE del mejor modelo ({ganadores_modelo['modelo']}), Lab 1", ganadores_modelo["RMSE"],
              f"{ganadores_modelo['RMSE'] / 1e3:.0f} mil"))
    f.append(("B4", f"MAE del mejor modelo ({ganadores_modelo['modelo']}), Lab 1", ganadores_modelo["MAE"],
              f"{ganadores_modelo['MAE'] / 1e3:.0f} mil"))
    for _, r in tabla_total.iterrows():
        f.append(("B4", f"Total · {r['modelo']}: MAE / RMSE / MAPE %", r["RMSE"],
                  f"{r['MAE']:.0f} / {r['RMSE']:.0f} / {r['MAPE_%']:.1f}"))
    mejores = met.loc[met.groupby("serie")["RMSE"].idxmin()]
    cuenta = mejores["modelo"].value_counts()
    for modelo, n in cuenta.items():
        f.append(("B4", f"Series (de {len(mejores)}) donde {modelo} fue el mejor", n, f"{n} de {len(mejores)}"))
    f.append(("B4", "Series donde SARIMA fue el mejor", int(mejores["modelo"].str.startswith("SARIMA").sum()),
              f"{int(mejores['modelo'].str.startswith('SARIMA').sum())} de {len(mejores)}"))
    f.append(("B4", "Meses reales dentro de la banda de 80 %", dentro[80], f"{dentro[80] * 100:.0f} %"))
    f.append(("B4", "Meses reales dentro de la banda de 95 %", dentro[95], f"{dentro[95] * 100:.0f} %"))
    f.append(("B4", "RMSE de Prophet reproducido para la gráfica (Lab 1: "
              f"{ganadores_modelo['RMSE']:.0f})", rep["RMSE"], f"{rep['RMSE']:.0f}"))
    f.append(("B4", "MAE de Prophet reproducido para la gráfica (Lab 1: "
              f"{ganadores_modelo['MAE']:.0f})", rep["MAE"], f"{rep['MAE']:.0f}"))
    f.append(("B4", "Promedio mensual real en el periodo de prueba", rep["media_test"], f"{rep['media_test'] / 1e3:.0f} mil"))
    f.append(("B4", "Error medio (MAE) de Prophet sobre el promedio del test", ganadores_modelo["MAE"] / rep["media_test"] * 100,
              f"{ganadores_modelo['MAE'] / rep['media_test'] * 100:.0f} %"))
    return pd.DataFrame(f, columns=["bloque", "cifra", "valor", "texto_infografia"])


# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datos", default="datos", type=Path)
    p.add_argument("--salida", default="salidas", type=Path)
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")   # la consola de Windows (cp1252) no imprime «−»

    familia = configurar_fuente()
    s = cargar_series(a.datos)
    resumen = pd.read_csv(a.datos / "resumen_inciso4.csv", index_col="serie")
    met = leer_metricas_lab1(a.datos / "detalle_por_serie.txt")
    a.salida.mkdir(parents=True, exist_ok=True)
    met.to_csv(a.salida / "metricas_modelos_lab1.csv", index=False)

    # control: el mejor modelo por RMSE de la tabla leída coincide con el resumen del Lab 1
    mejor_leido = met.loc[met.groupby("serie")["RMSE"].idxmin()].set_index("serie")["modelo"]
    for serie, modelo in resumen["mejor_modelo"].items():
        assert mejor_leido[serie] == modelo, f"{serie}: {mejor_leido[serie]} != {modelo}"

    idx, aerea, terr = grafica_estacionalidad(s, a.salida)
    q = tabla_cuadrante(s, resumen)
    resp = grafica_cuadrante(q, a.salida)
    tabla_total = grafica_modelos(met, a.salida)

    tr = pd.read_csv(a.datos / "train_series.csv", parse_dates=["Fecha"], index_col="Fecha")["Total_Viajeros"]
    te = pd.read_csv(a.datos / "test_series.csv", parse_dates=["Fecha"], index_col="Fecha")["Total_Viajeros"]
    f = ajustar_prophet(tr, te)
    dentro = grafica_proyeccion(s, f, a.salida)
    err = te.values - f["yhat"].values
    rep = {"RMSE": float(np.sqrt(np.mean(err ** 2))), "MAE": float(np.mean(np.abs(err))),
           "media_test": float(te.mean())}

    cifras = juntar_cifras(idx, aerea, terr, q, resp, met, tabla_total, dentro, rep, resumen)
    cifras.to_csv(a.salida / "cifras_bloques_3_4.csv", index=False)
    print(f"Fuente usada: {familia}\n")
    print(cifras[["bloque", "cifra", "texto_infografia"]].to_string(index=False))
    print(f"\nGráficas guardadas en {a.salida}/bloque3_*.(svg|pdf|png) y bloque4_*.(svg|pdf|png)")


if __name__ == "__main__":
    main()
