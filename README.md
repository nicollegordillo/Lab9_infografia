# Laboratorio 9 · Infografía

Infografía estática de una página que comunica a un directivo de INGUAT los hallazgos del
[Laboratorio 1 · Series de Tiempo](https://github.com/djuarez-2017510/Lab1_Serie_tiempo): qué nos enseñan 17 años de datos
migratorios sobre el turismo en Guatemala y qué debemos anticipar.

## Autores

- Humberto Alexander de la Cruz · 23735
- Daniel Oswaldo Juárez · 23709
- Nicolle Alexandra Gordillo · 22246

## Estructura

```
Lab9_Infografia/
├── datos/            Series del Lab 1 que usan los scripts (entrenamiento, prueba y resumen de modelos)
├── scripts/          Código que genera las cifras y los gráficos de la infografía
├── salidas/          Gráficos (SVG, PDF, PNG) y tablas de cifras generadas por los scripts
└── infografia/       Versiones finales: PNG 3300 × 5100 px y PDF 11 × 17 in
```

## Cómo reproducir los gráficos

Requisitos: Python 3.9+ y la fuente [Inter](https://rsms.me/inter/) instalada.

```bash
pip install -r requirements.txt
```

```bash
python scripts/bloques_1_2.py --datos datos --salida salidas
python scripts/bloques_3_4.py --datos datos --salida salidas
```

`bloques_1_2.py` imprime las cifras de los KPIs y las anotaciones, las guarda en
`salidas/cifras_bloques_1_2.csv` y exporta la gráfica del panorama histórico
(`salidas/bloque2_panorama.svg|pdf|png`) con el tamaño exacto de su espacio en el lienzo.

`bloques_3_4.py` hace lo mismo para los bloques 3 y 4: exporta cuatro gráficas
(`bloque3_estacionalidad`, `bloque3_cuadrante`, `bloque4_modelos`, `bloque4_proyeccion`, cada una en
SVG, PDF y PNG) y guarda todas las cifras del texto en `salidas/cifras_bloques_3_4.csv`.
Las métricas de los modelos (MAE, RMSE, AIC, BIC) se leen tal cual de `datos/detalle_por_serie.txt`,
la salida del Laboratorio 1, y se vuelcan a `salidas/metricas_modelos_lab1.csv`. Solo la línea y las
bandas de la proyección se vuelven a calcular, ajustando Prophet con la configuración del Lab 1
(semilla fija para las bandas). El Prophet reproducido da RMSE ≈ 119 500 contra 119 081 del Lab 1
(≈ 0.4 % de diferencia según la versión de la librería); las tablas usan siempre la cifra del Lab 1.


## Avance por bloque

| Bloque | Contenido | Script | Estado |
| --- | --- | --- | --- |
| 1 | Titular y 3 KPIs | `scripts/bloques_1_2.py` | Cifras listas |
| 2 | Panorama histórico 2009–2026 | `scripts/bloques_1_2.py` | Gráfica lista |
| 3 | Estacionalidad y cuadrante comparativo | `scripts/bloques_3_4.py` | Gráficas listas |
| 4 | Comparación de modelos y proyección | `scripts/bloques_3_4.py` | Gráficas listas |
| 5 | Recomendaciones y ficha técnica | — (texto escrito en Figma) | Listo |


## Diseño

- Lienzo: 11 × 17 in (en Figma, 792 × 1224 px; se exporta a 3300 × 5100 px).
- Fuente: Inter.
- Paleta: azul Atitlán `#1D5B8C` (dominante), verde jade `#2A7364` (mediación),
  terracota `#B9471A` (tónico, solo para acentos), gris `#9AA5B1`, fondo marfil `#FAF8F3`,
  texto carbón `#1F2933`.
- Las decisiones completas están en `documento/`.

## Enlaces

- Archivo de Figma: https://www.figma.com/design/fSaKd8XPndQOdS1Gn4oQs5/Lab-9-Infograf%C3%ADa?node-id=0-1&t=VpGnMuz8jD6zRJ7C-1

- Documento de explicación compartido: https://docs.google.com/document/d/1kGOwHA4xdSMlhN0x5vzvBCzt4zQ2TrcNBB34gj4XA4Y/edit?tab=t.0 

## Fuente de los datos

Base de Migración 2009–junio 2026, INGUAT / IGM (turistas y excursionistas), procesada en el
Laboratorio 1. Los datos se comparten con fines estrictamente académicos: este repositorio es
privado y no debe publicarse sin consulta previa al catedrático.
