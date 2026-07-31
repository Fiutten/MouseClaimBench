# Fase 5c: adaptador calibrado para Dynamic Sensorium

## Objetivo

El baseline dinámico inicial era deliberadamente débil: una regresión ridge
directa sobre descriptores de vídeo colapsados. Perdía frente al predictor
medio en Dynamic Sensorium, aunque superaba al control con estímulos permutados.

El objetivo de esta fase es añadir un adaptador predictivo más razonable sin
convertirlo en una caja negra ni usar el tier `oracle` para ajustar
hiperparámetros.

## Adaptador implementado

Se añadió `calibrated_residual_ridge` en
`mousebrainbench/surrogate/sensorium_adapters.py`.

La idea es:

1. calcular la respuesta media de entrenamiento;
2. modelar solo el residual dependiente del estímulo;
3. seleccionar por CV interna en `train`:
   - regularización ridge `alpha`;
   - escala residual `beta`;
4. predecir en held-out como:

```text
prediction = train_mean_response + beta * ridge_residual(video_features)
```

Esto evita que un residual ruidoso degrade la predicción respecto al predictor
medio, que era el fallo del ridge directo en vídeo.

La grid usada fue:

```text
alpha in {0.1, 0.3, 1, 3, 10, 30, 100}
beta  in linspace(0, 1.5, 61)
```

La selección se hace solo con `train`; `oracle` queda como evaluación held-out.

## Comando reproducible

```bash
.venv/bin/python -m mousebrainbench.benchmarks.sensorium_predictive_mis \
  data/dynamic_sensorium/extracted/<mouse-dir> \
  --modality dynamic \
  --eval-tier oracle \
  --adapter calibrated_residual_ridge \
  --adapter-alpha-grid 0.1,0.3,1,3,10,30,100 \
  --git-revision ff333fd \
  --output results/dynamic_sensorium_adapter/<mouse>_oracle_calibrated_residual_mis.json
```

## Resultados agregados

Artefacto agregado:

```text
results/dynamic_sensorium_adapter/summary_dynamic_sensorium2023_calibrated_residual_mis.json
```

| Métrica | n | Mediana | Mínimo | Máximo |
|---|---:|---:|---:|---:|
| Adapter correlation | `5` | `0.4759` | `0.3902` | `0.5009` |
| Adapter Δ mean | `5` | `0.0222` | `-0.0013` | `0.0330` |
| Adapter Δ scrambled | `5` | `0.0377` | `0.0327` | `0.0445` |
| Adapter selected as best stimulus model | `5` | `5/5` | `5/5` | `5/5` |
| Positive Δ mean | `5` | `4/5` | `4/5` | `4/5` |
| Reliability estimable | `5` | `0/5` | `0/5` | `0/5` |

Resultados individuales:

| Mouse | Mean | Adapter | Δ mean | Δ scrambled | alpha | beta |
|---|---:|---:|---:|---:|---:|---:|
| `dynamic29515` | `0.4238` | `0.4224` | `-0.0013` | `0.0377` | `100` | `0.575` |
| `dynamic29623` | `0.3680` | `0.3902` | `0.0222` | `0.0327` | `100` | `0.625` |
| `dynamic29647` | `0.4472` | `0.4801` | `0.0330` | `0.0445` | `100` | `0.675` |
| `dynamic29712` | `0.4530` | `0.4759` | `0.0230` | `0.0337` | `100` | `0.600` |
| `dynamic29755` | `0.4806` | `0.5009` | `0.0203` | `0.0415` | `100` | `0.725` |

## Interpretación crítica

Este resultado es positivo, pero no debe exagerarse.

Lo que sí demuestra:

1. El marco permite mejorar un caso negativo sin cambiar la métrica.
2. La mejora frente a mean es positiva en 4/5 ratones y positiva en mediana.
3. La mejora frente al control scrambled es positiva en 5/5 ratones.
4. El adaptador fue seleccionado con CV interna en `train`, no ajustado usando
   `oracle`.
5. El MIS sigue evitando una conclusión mecanística fuerte.

Lo que no demuestra:

1. No es un modelo SOTA Sensorium.
2. No resuelve el problema de fiabilidad, porque `oracle` no tiene vídeos
   repetidos exactos.
3. No aporta por sí solo mecanismo biológico.
4. No reemplaza un modelo temporal profundo u oficial.

## Decisión

**Avance positivo moderado.** Ya no estamos ante un caso puramente negativo:
existe una mejora predictiva reproducible en la mayoría de ratones y robusta
frente a scrambled. Para una contribución fuerte, el siguiente paso debe ser
comparar este adaptador contra un modelo temporal más competente y, en paralelo,
buscar una fuente de evaluación con repeticiones, OOD o restricciones
estructurales.

## Extensión 5d: descriptor temporal transparente

Se añadió `feature_mode=temporal_filterbank` en
`mousebrainbench/data/loaders/sensorium.py`. Este modo conserva:

- ventanas temporales gruesas del vídeo;
- energía de movimiento frame-a-frame;
- modulación de baja frecuencia de luminancia y contraste;
- pooling espacial por ventanas temporales.

No es un modelo profundo ni SOTA Sensorium. Es un baseline temporal auditable
para probar si la información temporal explícita aporta señal antes de pasar a
un modelo más pesado.

Comando reproducible:

```bash
.venv/bin/python scripts/run_dynamic_sensorium_temporal_filterbank.py --force
```

Artefacto:

```text
results/dynamic_sensorium_adapter/summary_dynamic_sensorium2023_temporal_filterbank_mis.json
```

Resultado agregado:

| Métrica | Valor |
|---|---:|
| Ratones | `5` |
| Temporal mejora frente a summary | `4/5` |
| Mediana temporal - summary | `0.00847` |
| Mediana temporal - mean | `0.03114` |
| Mediana temporal - scrambled | `0.05168` |

## Extensión 5e: OOD con respuestas liberadas

El README oficial de Dynamic Sensorium enlaza una release legacy con cinco
ratones y respuestas OOD liberadas. Se validaron los cinco ratones visibles con
`unzip -t`:

```text
dynamic29156-11-10
dynamic29228-2-10
dynamic29234-6-9
dynamic29513-3-5
dynamic29514-2-9
```

Los hashes SHA256 y los intentos parciales reemplazados están documentados en
`docs/DATA_PROVENANCE.md`. El loader corregido excluye automáticamente trials
sin respuesta porque alinea por ID de archivo. Los tiers `live_test_main`,
`live_test_bonus`, `final_test_main` y `final_test_bonus` tienen respuestas no
nulas.

Comando reproducible:

```bash
for root in data/dynamic_sensorium_ood/extracted/dynamic*-Video-*; do
  .venv/bin/python scripts/run_dynamic_sensorium_ood_probe.py "$root"
done
.venv/bin/python scripts/summarize_dynamic_sensorium_ood.py
```

Artefacto agregado:

```text
results/dynamic_sensorium_ood/summary_dynamic_sensorium_legacy_ood_temporal_comparison.json
```

| Métrica OOD | Valor |
|---|---:|
| Ratones válidos | `5` |
| Temporal mejora frente a summary | `5/5` |
| Temporal mejora frente a mean | `4/5` |
| Temporal mejora frente a scrambled | `5/5` |
| Mediana summary correlation | `0.41113` |
| Mediana temporal correlation | `0.42178` |
| Mediana temporal - summary | `0.01106` |
| Mediana temporal - mean | `0.02501` |
| Mediana temporal - scrambled | `0.05062` |
| Reliability estimable | `0/5` |
| MIS passed | `0/5` |

Interpretación: este es un resultado positivo moderado de generalización OOD,
pero no una conclusión mecanística. El descriptor temporal mejora frente a
summary y scrambled en los cinco ratones, y frente al predictor medio en cuatro
de cinco. `reliability_estimable=false` y el MIS completo no pasa; la
contribución defendible es que el marco separa explícitamente predicción, OOD e
identificabilidad.

## Extensión 5f: adaptador temporal SVD

Se añadió `temporal_svd_residual_ridge` como baseline temporal más fuerte sin
introducir todavía deep learning. El adaptador aprende una subbase SVD de los
descriptores `temporal_filterbank` usando solo `train`, predice residuales con
ridge y calibra componentes, `alpha` y `beta` por CV interna.

Comando reproducible para la cohorte principal:

```bash
.venv/bin/python scripts/run_dynamic_sensorium_temporal_svd.py \
  --extracted-root data/dynamic_sensorium/extracted \
  --output-dir results/dynamic_sensorium_temporal_svd/main \
  --summary-output results/dynamic_sensorium_temporal_svd/summary_dynamic_sensorium2023_temporal_svd.json \
  --eval-tier oracle \
  --alpha-grid 1,10,100 \
  --git-revision 439d9ae \
  --force
```

Comando reproducible para OOD legacy:

```bash
.venv/bin/python scripts/run_dynamic_sensorium_temporal_svd.py \
  --extracted-root data/dynamic_sensorium_ood/extracted \
  --output-dir results/dynamic_sensorium_temporal_svd/ood \
  --summary-output results/dynamic_sensorium_temporal_svd/summary_dynamic_sensorium_legacy_ood_temporal_svd.json \
  --eval-tier live_test_main \
  --eval-tier live_test_bonus \
  --eval-tier final_test_main \
  --eval-tier final_test_bonus \
  --alpha-grid 1,10,100 \
  --git-revision 439d9ae \
  --ood-gate \
  --force
```

Resultado:

| Cohorte | n | SVD > mean | SVD > scrambled | Mediana SVD - mean | SVD > temporal previo |
|---|---:|---:|---:|---:|---:|
| Dynamic Sensorium 2023 `oracle` | `5` | `4/5` | `5/5` | `0.03889` | `3/5` |
| Legacy OOD liberado | `5` | `5/5` | `5/5` | `0.03091` | `5/5` |

Interpretación crítica: el SVD temporal aporta una mejora incremental defendible,
especialmente en OOD, pero sigue siendo evidencia predictiva. El MIS permanece
negativo (`0/5` y `0/5`) porque no existe fiabilidad por repetición ni una
restricción estructural/causal activa.

### Cierre de cohorte OOD completa

Al completar `dynamic29228-2-10` y `dynamic29513-3-5`, la comparación SVD frente
al temporal-filterbank previo quedó:

| Mouse | Temporal filterbank | Temporal SVD | SVD - filterbank |
|---|---:|---:|---:|
| `dynamic29156-11-10` | `0.39149` | `0.39884` | `0.00735` |
| `dynamic29228-2-10` | `0.42886` | `0.43471` | `0.00586` |
| `dynamic29234-6-9` | `0.32300` | `0.36030` | `0.03729` |
| `dynamic29513-3-5` | `0.42178` | `0.42657` | `0.00479` |
| `dynamic29514-2-9` | `0.42881` | `0.43319` | `0.00438` |

SVD supera al temporal-filterbank en `5/5` ratones, con mediana `+0.00586`.
La mejora es consistente pero pequeña; el valor metodológico está en que el
benchmark detecta una mejora predictiva OOD sin permitir que esa mejora se
confunda con identificabilidad mecanística.
