# Laboratorio 9 · Infografía

**Universidad del Valle de Guatemala** · Facultad de Ingeniería · CC3084 Data Science · Semestre II-2026

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

`bloques_1_2.py` imprime las cifras de los KPIs y las anotaciones, las guarda en
`salidas/cifras_bloques_1_2.csv` y exporta la gráfica del panorama histórico
(`salidas/bloque2_panorama.svg|pdf|png`) con el tamaño exacto de su espacio en el lienzo.

## Avance por bloque

| Bloque | Contenido | Script | Estado |
| --- | --- | --- | --- |
| 1 | Titular y 4 KPIs | `scripts/bloques_1_2.py` | Cifras listas |
| 2 | Panorama histórico 2009–2026 | `scripts/bloques_1_2.py` | Gráfica lista |
| 3 | Estacionalidad y cuadrante comparativo | — | Pendiente |
| 4 | Comparación de modelos y proyección | — | Pendiente |
| 5 | Recomendaciones y ficha técnica | — | Pendiente |

## Diseño

- Lienzo: 11 × 17 in (en Figma, 792 × 1224 px; se exporta a 3300 × 5100 px).
- Fuente: Inter.
- Paleta: azul Atitlán `#1D5B8C` (dominante), verde jade `#2A7364` (mediación),
  terracota `#B9471A` (tónico, solo para acentos), gris `#9AA5B1`, fondo marfil `#FAF8F3`,
  texto carbón `#1F2933`.
- Las decisiones completas están en `documento/`.

## Enlaces

- Archivo de Figma: _(agregar enlace)_
- Documento de explicación compartido: _(agregar enlace)_

## Fuente de los datos

Base de Migración 2009–junio 2026, INGUAT / IGM (turistas y excursionistas), procesada en el
Laboratorio 1. Los datos se comparten con fines estrictamente académicos: este repositorio es
privado y no debe publicarse sin consulta previa al catedrático.
