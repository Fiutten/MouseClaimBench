# Fase 5: Sensorium/Dynamic Sensorium predictivo frente a MIS

## Objetivo

Usar Sensorium/Dynamic Sensorium como caso moderno de predicción neural para
comprobar una afirmación concreta:

```text
un modelo puede alcanzar alta correlación predictiva sin quedar justificado como
modelo mecanístico.
```

Esta fase no intenta superar el leaderboard de Sensorium. Su función es conectar
MouseBrainBench con benchmarks modernos de IA/neurociencia y evaluar si el MIS
aporta información que la correlación estándar no captura.

## Datos oficiales

Sensorium 2022 usa imágenes naturales estáticas; Dynamic Sensorium 2023 usa
vídeos. Ambos publican datos en GIN como zips por ratón, con estructura:

```text
data/images/      # Sensorium 2022
data/videos/      # Dynamic Sensorium 2023
data/responses/
data/behavior/
data/pupil_center/
meta/trials/tiers.npy
meta/trials/frame_image_id.npy
meta/trials/trial_idx.npy
meta/neurons/
```

Los datos no se versionan en Git. Además, Dynamic Sensorium declara licencia
`CC BY-NC-ND 4.0`, por lo que cualquier análisis publicable requiere revisar y
respetar explícitamente sus condiciones.

## Implementación actual

Loader:

```text
mousebrainbench/data/loaders/sensorium.py
```

Benchmark:

```text
mousebrainbench/benchmarks/sensorium_predictive_mis.py
```

Smoke sintético:

```text
mousebrainbench/benchmarks/sensorium_synthetic_smoke.py
```

El loader soporta la estructura documentada de Sensorium sin depender del stack
oficial de PyTorch. Extrae descriptores simples de estímulo y colapsa respuestas
estáticas/dinámicas a una matriz `trial x neuron` para una primera evaluación
ligera.

## Baselines actuales

1. `mean_response`: media neuronal del conjunto de entrenamiento.
2. `stimulus_ridge`: regresión ridge sobre descriptores deterministas del
   estímulo.
3. `scrambled_stimulus_ridge`: mismo modelo con estímulos de entrenamiento
   permutados.

Estos baselines son deliberadamente simples. No compiten con SOTA; sirven para
verificar el marco antes de añadir modelos visuales profundos.

## Resultado smoke actual

Artefacto:

```text
results/sensorium_predictive_mis_benchmark.json
```

Resultado:

- correlación `stimulus_ridge`: `0.9938`;
- correlación `mean_response`: `0.1353`;
- correlación `scrambled_stimulus_ridge`: `0.1308`;
- fiabilidad por repeticiones: `0.9888`;
- reproducibilidad: pasa;
- especificidad de estímulo: pasa;
- identificabilidad mecanística: falla;
- decisión: `predictive_signal_requires_extra_mechanistic_evidence`.

Interpretación: el sistema demuestra exactamente la separación que buscamos.
Puede haber predicción fuerte y específica de estímulo, pero eso no autoriza una
afirmación mecanística si no hay estructura, intervención, OOD o restricciones
causales adicionales.

## Cómo correr sobre un ratón oficial

Tras descargar y descomprimir un zip oficial:

```bash
mousebrainbench-sensorium-mis /path/to/unzipped/sensorium_mouse \
  --modality static \
  --max-trials 1000 \
  --eval-tier validation \
  --output results/sensorium_real_mouse_mis.json
```

Para Dynamic Sensorium:

```bash
mousebrainbench-sensorium-mis /path/to/unzipped/dynamic_mouse \
  --modality dynamic \
  --max-trials 1000 \
  --eval-tier validation \
  --output results/dynamic_sensorium_real_mouse_mis.json
```

`--max-trials` es intencional en la primera ejecución real: queremos validar
formato, memoria y tiempos antes de procesar un ratón completo.

Por defecto se evalúa `validation`/`val`. En los datasets de competición
Sensorium 2022 las tiers `test` y `final_test` no deben usarse para calcular
correlación directa porque sus respuestas están retenidas o zeroed para el
protocolo de leaderboard.

## Criterio de avance

Avanzamos a descarga real si:

1. el loader sigue pasando tests;
2. el smoke conserva la separación predicción/MIS;
3. hay espacio suficiente para al menos un zip oficial;
4. se acepta explícitamente que el primer objetivo no es SOTA, sino evaluación
   metodológica.

Tras un primer ratón real, los siguientes pasos son:

1. medir rendimiento de baselines transparentes;
2. añadir una comparación con splits OOD si el dataset lo permite;
3. decidir si merece introducir un modelo visual profundo preentrenado o baseline
   oficial;
4. solo entonces discutir una reclamación Q1 de IA.

La primera ejecución real con tres zips Sensorium 2022 está documentada en
[PHASE5_SENSORIUM_REAL_RESULTS.md](PHASE5_SENSORIUM_REAL_RESULTS.md).
