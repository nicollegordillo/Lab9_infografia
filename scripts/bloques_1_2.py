import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
import pandas as pd

# ------------------------------------------------------------------ paleta
FONDO = "#FAF8F3"      # marfil
TEXTO = "#1F2933"      # carbón
TEXTO2 = "#52606D"     # texto secundario
AZUL = "#1D5B8C"       # dominante: dato real
AZUL_CLARO = "#9DBAD3" # mismo azul aclarado: serie mensual de fondo
GRIS = "#9AA5B1"       # contexto
CUADRICULA = "#E4E7EB"
TERRACOTA = "#B9471A"  # tónico: pandemia / anomalía

# Tamaño (pulgadas) del espacio del panorama en el lienzo de 11 × 17 in
TAM_PANORAMA = (10.0, 2.2)

# ------------------------------------------------------------------ fuente
def configurar_fuente():
    nombres = {f.name for f in font_manager.fontManager.ttflist}
    familia = "Inter" if "Inter" in nombres else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": familia,
        "svg.fonttype": "none",   # texto como texto en el SVG -> editable en Figma
        "pdf.fonttype": 42,       # fuentes TrueType incrustadas en el PDF
        "axes.unicode_minus": False,
    })
    return familia


# ------------------------------------------------------------------ datos
def cargar_series(carpeta: Path) -> pd.DataFrame:
    tr = pd.read_csv(carpeta / "train_series.csv", parse_dates=["Fecha"], index_col="Fecha")
    te = pd.read_csv(carpeta / "test_series.csv", parse_dates=["Fecha"], index_col="Fecha")
    return pd.concat([tr, te]).sort_index()


def calcular_cifras(s: pd.DataFrame) -> pd.DataFrame:
    T = s["Total_Viajeros"]
    anual = T.groupby(T.index.year).sum()
    prom_2019 = T["2019"].mean()
    abr_2020 = T["2020-04"].iloc[0]
    vias = s[["Via_Aerea", "Via_Terrestre", "Via_Maritima"]].sum()
    pre = T[:"2019-12"]
    indice = pre.groupby(pre.index.month).mean() / pre.mean() * 100
    pico_fecha, pico_valor = T[:"2019-12"].idxmax(), T[:"2019-12"].max()

    filas = [
        ("B1", "Total de ingresos ene 2009 – jun 2026", T.sum(), f"{T.sum()/1e6:.1f} M"),
        ("B1", "Caída abril 2020 vs mes promedio 2019 (%)", (abr_2020 / prom_2019 - 1) * 100,
         f"{(abr_2020 / prom_2019 - 1) * 100:.0f} %"),
        ("B1", "Participación vía terrestre (%)", vias["Via_Terrestre"] / vias.sum() * 100,
         f"{vias['Via_Terrestre'] / vias.sum() * 100:.0f} %"),
        ("B1", "Participación vía aérea (%)", vias["Via_Aerea"] / vias.sum() * 100,
         f"{vias['Via_Aerea'] / vias.sum() * 100:.0f} %"),
        ("B1", "Índice estacional de diciembre 2009–2019 (mes promedio = 100)", indice[12],
         f"+{indice[12] - 100:.0f} %"),
        ("B2", "Ingresos 2009", anual[2009], f"{anual[2009]/1e6:.2f} M"),
        ("B2", "Ingresos 2019", anual[2019], f"{anual[2019]/1e6:.2f} M"),
        ("B2", "Crecimiento 2009→2019 (veces)", anual[2019] / anual[2009], f"×{anual[2019] / anual[2009]:.1f}"),
        ("B2", f"Récord mensual ({pico_fecha:%b %Y})", pico_valor, f"{pico_valor/1e3:.0f} mil"),
        ("B2", "Abril 2020", abr_2020, f"{abr_2020/1e3:.0f} mil"),
        ("B2", "Mes promedio 2019", prom_2019, f"{prom_2019/1e3:.0f} mil"),
        ("B2", "Ingresos 2022", anual[2022], f"{anual[2022]/1e6:.2f} M"),
        ("B2", "2022 como % de 2019", anual[2022] / anual[2019] * 100, f"{anual[2022] / anual[2019] * 100:.0f} %"),
        ("B2", "Ingresos 2023 (cambio de registro)", anual[2023], f"{anual[2023]/1e6:.2f} M"),
        ("B2", "Ingresos 2025", anual[2025], f"{anual[2025]/1e6:.2f} M"),
    ]
    return pd.DataFrame(filas, columns=["bloque", "cifra", "valor", "texto_infografia"])


# ------------------------------------------------------------------ gráfica
def grafica_panorama(s: pd.DataFrame, salida: Path, ancho_in=10.0, alto_in=3.0):
    """Línea mensual + promedio mensual de cada año (escalones).

    Se usa el promedio de cada año en lugar de un promedio móvil porque el móvil
    "adelanta" la caída de 2020 hacia 2019 y suaviza el cambio de registro de 2023.
    """
    T = s["Total_Viajeros"] / 1000                     # miles de viajeros
    prom_anual = T.groupby(T.index.year).mean()        # mes promedio de cada año

    fig, ax = plt.subplots(figsize=(ancho_in, alto_in))
    fig.patch.set_facecolor(FONDO)
    ax.set_facecolor(FONDO)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.97, bottom=0.11)

    # Pandemia: banda de marzo 2020 a diciembre 2021
    ini, fin = pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-31")
    ax.axvspan(ini, fin, color=TERRACOTA, alpha=0.10, lw=0, zorder=0)
    ax.text(ini + (fin - ini) / 2, 512, "Pandemia", ha="center", va="top",
            fontsize=11, color=TERRACOTA, fontweight="bold")

    # Cambio de registro: enero 2023
    quiebre = pd.Timestamp("2023-01-01")
    ax.axvline(quiebre, color=GRIS, lw=1, ls=(0, (3, 3)), zorder=1)

    # Serie mensual (fina) y promedio de cada año (escalones gruesos)
    ax.plot(T.index, T.values, color=AZUL_CLARO, lw=1.0, zorder=2)
    for anio, v in prom_anual.items():
        x0 = pd.Timestamp(f"{anio}-01-01")
        x1 = min(pd.Timestamp(f"{anio}-12-31"), T.index.max() + pd.offsets.MonthEnd(0))
        ax.plot([x0, x1], [v, v], color=AZUL, lw=2.8, solid_capstyle="butt", zorder=3)

    # Ejes
    ax.set_ylim(0, 520)
    ax.set_yticks([0, 100, 200, 300, 400, 500])
    ax.set_yticklabels(["0", "100", "200", "300", "400", "500"])
    ax.set_xlim(pd.Timestamp("2008-12-01"), pd.Timestamp("2027-01-15"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2, month=1, day=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", color=CUADRICULA, lw=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(GRIS)
    ax.tick_params(axis="both", labelsize=11, colors=TEXTO2, length=0, pad=4)
    ax.tick_params(axis="x", length=3, color=GRIS)

    # ---- anotaciones (texto directo, sin leyenda) ----
    def nota(x, y, lineas, ha="left", color=TEXTO):
        """Primera línea en negrita; las demás en gris. y = borde superior."""
        txt = ax.text(x, y, lineas[0], fontsize=11, fontweight="bold", color=color, ha=ha, va="top")
        if len(lineas) > 1:   # resto de líneas 15 pt debajo, sin importar la escala del eje
            ax.annotate("\n".join(lineas[1:]), xy=(x, y), xytext=(0, -15), textcoords="offset points",
                        fontsize=11, color=TEXTO2, ha=ha, va="top", linespacing=1.3)
        return txt

    def guia(x0, y0, x1, y1):
        ax.plot([x0, x1], [y0, y1], color=GRIS, lw=0.8, zorder=4)

    Ttot = s["Total_Viajeros"]
    a09, a19, a22 = (Ttot[str(a)].sum() / 1e6 for a in (2009, 2019, 2022))

    # 1. Década de crecimiento
    nota(pd.Timestamp("2009-03-01"), 505, [f"Una década de crecimiento: ×{a19 / a09:.1f}"])

    # 2. Récord de diciembre 2019
    pico_f = T[:"2019-12"].idxmax(); pico_v = T[:"2019-12"].max()
    ax.scatter([pico_f], [pico_v], s=40, color=AZUL, zorder=5, edgecolor=FONDO, linewidth=1.5)
    ax.text(pico_f - pd.Timedelta(days=60), pico_v + 12, f"Récord: {pico_v:.0f} mil · dic 2019",
            fontsize=11, fontweight="bold", color=TEXTO, ha="right", va="center")

    # 3. Cierre de fronteras: mínimo de abril 2020
    min_f = pd.Timestamp("2020-04-01"); min_v = T[min_f]
    caida = (min_v / T["2019"].mean() - 1) * 100
    ax.scatter([min_f], [min_v], s=40, color=TERRACOTA, zorder=5, edgecolor=FONDO, linewidth=1.5)
    nota(pd.Timestamp("2019-08-01"), 168,
         ["Mar 2020: cierre de fronteras",
          f"abril cae a {min_v:.0f} mil ({caida:.0f} %)".replace("-", "−")], ha="right", color=TERRACOTA)
    guia(pd.Timestamp("2019-09-01"), 62, min_f - pd.Timedelta(days=20), min_v + 4)

    # 4. Rebote 2022 (pico de diciembre 2022)
    r_f = T["2022"].idxmax(); r_v = T[r_f]
    nota(pd.Timestamp("2023-03-01"), 505, [f"2022: rebote a {a22:.2f} M", f"{a22 / a19 * 100:.0f} % de 2019"])
    guia(pd.Timestamp("2023-02-15"), 485, r_f + pd.Timedelta(days=10), r_v + 8)

    # 5. Cambio de registro 2023
    nota(pd.Timestamp("2023-03-01"), 158,
         ["2023: cambio de registro", "deja de contar aparte a", "guatemaltecos del exterior"])

    # 6. Etiquetas directas de las dos marcas (en lugar de leyenda)
    ax.plot([pd.Timestamp("2009-03-01"), pd.Timestamp("2009-11-01")], [72, 72], color=AZUL, lw=2.8)
    ax.text(pd.Timestamp("2010-01-15"), 72, "mes promedio de cada año", fontsize=11, color=AZUL,
            fontweight="bold", va="center")
    ax.plot([pd.Timestamp("2009-03-01"), pd.Timestamp("2009-11-01")], [34, 34], color=AZUL_CLARO, lw=1.2)
    ax.text(pd.Timestamp("2010-01-15"), 34, "cada mes", fontsize=11, color=TEXTO2, va="center")

    salida.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(salida / f"bloque2_panorama.{ext}", dpi=300, facecolor=FONDO)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datos", default="datos", type=Path)
    p.add_argument("--salida", default="salidas", type=Path)
    a = p.parse_args()

    familia = configurar_fuente()
    s = cargar_series(a.datos)
    cifras = calcular_cifras(s)
    a.salida.mkdir(parents=True, exist_ok=True)
    cifras.to_csv(a.salida / "cifras_bloques_1_2.csv", index=False)
    print(f"Fuente usada: {familia}\n")
    print(cifras[["bloque", "cifra", "texto_infografia"]].to_string(index=False))
    grafica_panorama(s, a.salida, *TAM_PANORAMA)
    print(f"\nGráfica guardada en {a.salida}/bloque2_panorama.(svg|pdf|png)")


if __name__ == "__main__":
    main()
