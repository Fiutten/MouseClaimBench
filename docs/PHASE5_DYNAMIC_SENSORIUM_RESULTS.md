# Fase 5b: resultados reales Dynamic Sensorium 2023

## Descarga

Se descargó la cohorte pública visible en GIN Dynamic Sensorium 2023:
cinco zips oficiales de vídeo. Esta descarga no debe confundirse con una
participación en la competición Sensorium 2023: los tiers `live_test_*` y
`final_test_*` contienen respuestas retenidas/zeroed en esta release pública.

| Dataset | Tamaño zip | SHA-256 |
|---|---:|---|
| `dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20` | `9.5G` | `24da060fe55e060f8aa67d5c23c04a5d9868f84487eba05a4daa24e8c722dba0` |
| `dynamic29623-4-9-Video-9b4f6a1a067fe51e15306b9628efea20` | `8.9G` | `937a7046e911782c65011b51bb489798f93c7879bfcb2e318ccf62174f7e2dda` |
| `dynamic29647-19-8-Video-9b4f6a1a067fe51e15306b9628efea20` | `9.9G` | `860592ab060d40a03be0c270da19a26cf63f751d31b1588ba99b89ff8e55171c` |
| `dynamic29712-5-9-Video-9b4f6a1a067fe51e15306b9628efea20` | `9.7G` | `be7c9661aa652efbe00005b15b32a5cb4a8862a999dfcaa4bb99f9dec6457ddb` |
| `dynamic29755-2-8-Video-9b4f6a1a067fe51e15306b9628efea20` | `10G` | `fcbbaa6c714a0e8f77f3cfaf50f435232036b21391cae998f336ac6b38b73516` |

Los cinco zips pasaron `unzip -t` sin errores. Los datos no se versionan en Git.

Uso local:

```text
data/dynamic_sensorium/raw       48G
data/dynamic_sensorium/extracted 94G
```

## Estructura observada

| Dataset | Trials | Neuronas | Tiers |
|---|---:|---:|---|
| `dynamic29515` | `711` | `7863` | `train=348`, `oracle=58`, `live_test_main=56`, `live_test_bonus=58`, `final_test_main=57`, `final_test_bonus=134` |
| `dynamic29623` | `682` | `7908` | `train=329`, `oracle=56`, `live_test_main=53`, `live_test_bonus=57`, `final_test_main=56`, `final_test_bonus=131` |
| `dynamic29647` | `711` | `8202` | `train=354`, `oracle=60`, `live_test_main=60`, `live_test_bonus=58`, `final_test_main=59`, `final_test_bonus=120` |
| `dynamic29712` | `739` | `7939` | `train=359`, `oracle=60`, `live_test_main=60`, `live_test_bonus=80`, `final_test_main=60`, `final_test_bonus=120` |
| `dynamic29755` | `712` | `8122` | `train=354`, `oracle=59`, `live_test_main=60`, `live_test_bonus=60`, `final_test_main=60`, `final_test_bonus=119` |

Cada trial contiene vídeos `height x width x frames`, respuestas
`neurons x frames`, comportamiento `2 x frames` y pupil center `2 x frames`.
Las respuestas de `train` y `oracle` contienen `NaN` para muestras faltantes; el
loader usa media temporal ignorando `NaN` y convierte neuronas totalmente
faltantes a cero para mantener matrices finitas. Esto queda documentado en
`mousebrainbench/data/loaders/sensorium.py`.

## Tiers evaluables

Los tiers `live_test_*` y `final_test_*` tienen respuestas públicas cero y no se
usan para métricas directas. El tier `oracle` contiene respuestas no nulas y se
usa como evaluación pública reproducible.

Se comprobó por hash de archivo que `oracle` no contiene vídeos repetidos
exactos en ninguno de los cinco ratones:

| Dataset | Oracle trials | Vídeos únicos | Grupos repetidos |
|---|---:|---:|---:|
| `dynamic29515` | `58` | `58` | `0` |
| `dynamic29623` | `56` | `56` | `0` |
| `dynamic29647` | `60` | `60` | `0` |
| `dynamic29712` | `60` | `60` | `0` |
| `dynamic29755` | `59` | `59` | `0` |

Por tanto, `reliability=0.0` en estos artefactos no significa baja fiabilidad
neural. Significa `reliability_estimable=false`: no hay repeticiones exactas
con respuestas públicas para estimar split-half reliability con el mecanismo
actual.

## Baseline usado

Se reutiliza `stimulus_ridge` con modalidad `dynamic`. El descriptor de vídeo es
intencionadamente barato:

- estadísticos globales de intensidad;
- dinámica global de medias por frame;
- descriptor espacial pooled `8 x 8` sobre el vídeo medio;
- covariables conductuales y pupilares como contexto opcional.

Este baseline colapsa el tiempo. No es un modelo moderno de vídeo ni un modelo
oficial Sensorium. Su función científica aquí es actuar como prueba de estrés
negativa: si el marco es honesto, debe detectar cuándo una representación
simple no basta.

## Resultados agregados

Artefacto agregado:

```text
results/dynamic_sensorium_real/summary_dynamic_sensorium2023_oracle_mis.json
```

| Métrica | n | Mediana | Mínimo | Máximo |
|---|---:|---:|---:|---:|
| Best predictive correlation | `5` | `0.4094` | `0.1969` | `0.4380` |
| Best Δ mean | `5` | `-0.0435` | `-0.1710` | `-0.0311` |
| Best Δ scrambled | `5` | `0.0521` | `0.0053` | `0.0978` |
| Reliability estimable | `5` | `0/5` | `0/5` | `0/5` |
| MIS score | `5` | `0.2222` | `0.1228` | `0.2222` |

Comparación directa con Sensorium 2022 estático:

| Lectura | Sensorium 2022 estático | Dynamic Sensorium 2023 |
|---|---:|---:|
| Median Best Δ mean | `0.0948` | `-0.0435` |
| Median Best Δ scrambled | `0.0684` | `0.0521` |
| Reliability estimable | sí, `0.6420` mediana en `test` repetido | no, `0/5` ratones |

## Interpretación crítica

Dynamic Sensorium 2023 funciona como caso negativo útil para nuestra propuesta:

1. El dataset se descargó, validó y puede leerse de forma reproducible.
2. El tier público `oracle` permite evaluar respuesta neuronal no nula, pero no
   permite estimar fiabilidad por repeticiones exactas.
3. El descriptor temporal colapsado supera a menudo al control con estímulos
   permutados, lo que indica cierta información de estímulo.
4. Sin embargo, no supera al predictor medio de entrenamiento. Por tanto no
   debe presentarse como modelo predictivo fuerte.
5. El MIS global falla correctamente: no hay reproducibilidad estimable, no hay
   ganancia frente a mean y no hay evidencia estructural, causal u OOD.

Esto fortalece la tesis metodológica del proyecto: el marco distingue entre
correlación aparente, especificidad frente a controles, reproducibilidad
estimable e identificabilidad mecanística. Para un resultado Q1 fuerte, el
siguiente paso no es inflar esta prueba, sino añadir un adaptador a un modelo
temporal competitivo u oficial Sensorium y volver a evaluar si la mejora
predictiva cambia o no la conclusión mecanística.

## Seguimiento: adaptador calibrado

Se implementó un adaptador `calibrated_residual_ridge` como siguiente paso
positivo. El adaptador selecciona regularización y escala residual usando solo
CV interna en `train`, y después evalúa `oracle`.

Resultado agregado:

```text
results/dynamic_sensorium_adapter/summary_dynamic_sensorium2023_calibrated_residual_mis.json
```

Resumen:

- mediana de correlación del adaptador: `0.4759`;
- mediana `Δ mean`: `+0.0222`;
- mediana `Δ scrambled`: `+0.0377`;
- `Δ mean` positivo en `4/5` ratones;
- `Δ scrambled` positivo en `5/5` ratones;
- fiabilidad sigue no estimable en `oracle`.

La documentación completa está en
[PHASE5_DYNAMIC_SENSORIUM_ADAPTER.md](PHASE5_DYNAMIC_SENSORIUM_ADAPTER.md).
