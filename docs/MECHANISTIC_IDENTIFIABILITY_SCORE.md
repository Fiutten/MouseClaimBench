# Mechanistic Identifiability Score

## Propósito

El `Mechanistic Identifiability Score` (MIS) no mide si un modelo predice bien
en abstracto. Mide si una afirmación mecanística parcial es defendible con la
evidencia disponible.

La motivación viene de un riesgo concreto del proyecto: una señal neural puede
ser reproducible, predictible y aun así no identificar el mecanismo anatómico o
dinámico que queremos evaluar.

## Definición operacional

El MIS se compone de tres bloques no intercambiables:

1. `reproducibility`: el target empírico es estable entre sesiones, animales o
   particiones independientes.
2. `topology_specificity`: el modelo que conserva la hipótesis anatómica supera
   controles nulos razonables, como grafos permutados, desconectados o
   transpuestos recalibrados.
3. `directed_identifiability`: el target contiene información suficiente para
   resolver dirección, latencia, lead-lag o una firma dinámica equivalente.

Cada bloque contiene criterios preregistrados. Cada criterio tiene:

- nombre;
- valor observado;
- umbral;
- dirección de comparación;
- decisión binaria;
- puntuación normalizada descriptiva en `[0, 1]`.

La decisión global es conjuntiva:

```text
MIS pasa si y solo si pasan todos los bloques.
```

Esto es deliberado. Un resultado excelente en reproducibilidad no puede
compensar un fallo en topología o dirección, porque esos bloques responden
preguntas científicas distintas.

## Interpretación

`passed = true` significa que el modelo/target supera los requisitos mínimos
para una afirmación mecanística parcial dentro de la definición usada.

`passed = false` no significa que el dataset sea inútil ni que el modelo no
prediga. Significa que el paquete modelo-target no soporta la afirmación
mecanística concreta.

## Casos incluidos

### Benchmark sintético con verdad conocida

El script `mousebrainbench-synthetic-mis` genera cuatro casos:

- `directed_truth`: señal dirigida con latencias regionales conocidas.
- `common_drive_nonidentifiable`: señal reproducible y predictible, pero sin
  estructura dirigida identificable.
- `topology_without_direction`: señal con estructura regional específica, pero
  sin latencias/lead-lag resolubles.
- `direction_without_topology_specificity`: señal con timing dirigido, pero sin
  predictor topológicamente específico.

El resultado esperado es que solo `directed_truth` pase MIS. Esto comprueba que
el score no confunde reproducibilidad, topología parcial o timing parcial con
una afirmación mecanística completa.

Artefacto:

```text
results/synthetic_identifiability_benchmark.json
```

### Allen VBN como caso negativo real

El script `mousebrainbench-allen-mis` aplica MIS a los resultados sellados de
Fase 2c, Fase 3 y Fase 4.

Resultado actual:

- `reproducibility`: pasa.
- `topology_specificity`: falla.
- `directed_identifiability`: falla.

Conclusión:

```text
reproducible_target_without_mechanistic_identifiability
```

Este resultado es útil porque evita una afirmación excesiva: Allen VBN contiene
un target evocado reproducible, pero este target no debe usarse como evidencia
de conectividad anatómico-dinámica dirigida.

Artefacto:

```text
results/allen_vbn_mechanistic_identifiability_score.json
```

## Límites

El MIS no es todavía un estándar externo. Es una propuesta interna formalizada
para hacer falsables nuestras afirmaciones. Para convertirlo en contribución Q1
necesitamos aplicarlo a más de un régimen:

- un caso negativo real, ya cubierto por Allen VBN;
- un caso positivo/predictivo moderno, candidato: Sensorium/Dynamic Sensorium;
- opcionalmente un caso estructura-función de alta resolución, candidato:
  MICrONS en escala piloto.

## Sensibilidad

El script `mousebrainbench-q1-sensitivity` comprueba que las conclusiones
actuales no dependen de un único umbral:

- Allen VBN debe permanecer negativo bajo perturbaciones razonables de los
  umbrales.
- Sensorium 2022 static debe conservar evidencia parcial de fiabilidad y
  topografía.
- Dynamic Sensorium debe reportarse como evidencia predictiva con ganancias NN
  pequeñas, no como mecanismo.

Artefacto:

```text
results/q1_sensitivity/summary.json
```

## MIS 2.0: calibración sintética ampliada

La siguiente línea metodológica del proyecto no sustituye el MIS usado en el
manuscrito enviado. Lo extiende con un benchmark sintético de calibración para
estimar falsos positivos, falsos negativos y estabilidad de bloques bajo verdad
conocida.

Comando:

```text
mousebrainbench-mis2-synthetic
```

Artefactos:

```text
results/mis2_synthetic_calibration/summary.json
results/mis2_synthetic_calibration/summary.md
results/mis2_threshold_sensitivity/summary.json
results/mis2_threshold_sensitivity/summary.md
```

La suite incluye escenarios positivos y negativos:

- verdad dirigida limpia;
- verdad dirigida con más ruido;
- verdad dirigida con pocas sesiones;
- verdad dirigida de baja señal/ruido;
- common drive altamente reproducible;
- topología sin dirección;
- dirección sin topología específica;
- predicción sin topología verdadera;
- common drive ruidoso.

La regla crítica se mantiene: un caso no mecanístico no debe pasar MIS aunque
sea reproducible o predictivo. Si aparece un falso positivo en esta suite, la
decisión correcta no es relajar la interpretación, sino revisar el gate. Los
falsos negativos en escenarios de baja señal se interpretan como conservadurismo
del gate y deben cuantificarse antes de proponer MIS 2.0 como contribución
general.

El comando `mousebrainbench-mis2-sensitivity` añade esa cuantificación mediante
un barrido de ruido, tamaño muestral y perfiles de umbral. Cada celda se
clasifica como:

- `safe`: falsos positivos cero y falsos negativos bajos;
- `conservative`: falsos positivos cero y falsos negativos altos;
- `dangerous`: falsos positivos no nulos con sensibilidad aparentemente buena;
- `unstable`: falsos positivos y falsos negativos problemáticos.

La condición mínima para seguir desarrollando MIS 2.0 es que los escenarios
negativos diseñados no entren en regiones `dangerous` o `unstable`.

El protocolo de desarrollo queda documentado en
[MIS2_CALIBRATION_PROTOCOL.md](MIS2_CALIBRATION_PROTOCOL.md).
