# Fase 5: resultados reales Sensorium 2022

## Descarga

Se descargó la cohorte completa disponible en GIN Sensorium 2022: siete zips
oficiales. La petición inicial de ampliar con 17 ratones nuevos no es posible
dentro de este recurso concreto; Sensorium 2022 contiene 7 ratones en total.

| Dataset | Rol | Tamaño zip | SHA-256 |
|---|---|---:|---|
| `static21067-10-18-GrayImageNet-94c6ff995dac583098847cfecd43e7b6` | pretraining mouse | `423M` | `2a3a0a7516855f3b305fc000e8987b609d6917c89804c911b76300fc66cfc802` |
| `static22846-10-16-GrayImageNet-94c6ff995dac583098847cfecd43e7b6` | pretraining mouse | `397M` | `c5baf086e58ef303ebe62cb49da35b73c496f974cc990e973c3f104309cef6da` |
| `static23343-5-17-GrayImageNet-94c6ff995dac583098847cfecd43e7b6` | pretraining mouse | `393M` | `8145d4b3f5d7a8aae9df4f04d4d6061215ab9060a47611fb4cdbf8c5ecfb626d` |
| `static23656-14-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6` | pretraining mouse | `411M` | `fb02840192b226c0b7951df6387ff3ebf8236874725fb57ab29212376479c937` |
| `static23964-4-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6` | pretraining mouse | `411M` | `677739254fd338d1a249a2cf2f4f965048c5a8919a254d57d5317d2f998f7481` |
| `static26872-17-20-GrayImageNet-94c6ff995dac583098847cfecd43e7b6` | Sensorium competition mouse 1 | `416M` | `c004994ba0110dcff21273af3e7b759ca30881dabbdc74201e777e89a8ce68d0` |
| `static27204-5-13-GrayImageNet-94c6ff995dac583098847cfecd43e7b6` | Sensorium+ competition mouse 2 | `412M` | `bfafe3589131485ae1a20be643956618e3520c7fbaf25df39e9d7b1ccca9da5d` |

Los siete zips pasaron `unzip -t` sin errores. Los datos no se versionan en Git.

Uso local:

```text
data/sensorium/raw       2.8G
data/sensorium/extracted 7.9G
```

## Estructura observada

| Dataset | Trials | Neuronas | Tiers |
|---|---:|---:|---|
| `static21067` | `5994` | `8372` | `train=4473`, `validation=523`, `test=998` |
| `static22846` | `5997` | `7344` | `train=4498`, `validation=500`, `test=999` |
| `static23343` | `5951` | `7334` | `train=4466`, `validation=496`, `test=989` |
| `static23656` | `5966` | `8107` | `train=4477`, `validation=496`, `test=993` |
| `static23964` | `5983` | `8098` | `train=4490`, `validation=499`, `test=994` |
| `static26872` | `6955` | `7776` | `train=4474`, `validation=496`, `test=990`, `final_test=995` |
| `static27204` | `6959` | `7538` | `train=4471`, `validation=497`, `test=995`, `final_test=996` |

En los dos datasets de competición (`26872`, `27204`), las respuestas de `test`
y `final_test` están zeroed/retenidas, tal como indica el README oficial. Por
eso la evaluación directa usa `validation`. En los cinco pretraining mice,
`test` contiene respuestas reales y 100 estímulos repetidos, por lo que se usa
como caso robusto de fiabilidad.

## Baseline usado

Se usa `stimulus_ridge`, una regresión ridge transparente sobre:

- descriptores globales de intensidad;
- descriptor espacial pooled `8 x 8`;
- control `scrambled_stimulus_ridge` con estímulos de entrenamiento permutados.

Además se calcula `stimulus_context_ridge`, que añade covariables de
comportamiento y posición pupilar cuando no están zeroed. El control
`scrambled_stimulus_context_ridge` conserva esas covariables y permuta solo los
estímulos de entrenamiento. Esto permite distinguir mejora por estado
conductual de mejora atribuible a imagen.

Esto sigue siendo un baseline ligero, no SOTA. Su función es comprobar si el
marco detecta señal predictiva antes de introducir modelos visuales profundos.

## Resultados agregados

Artefacto agregado:

```text
results/sensorium_real/summary_sensorium2022_static_mis.json
```

### Pretraining mice con `test` repetido

| Métrica | n | Mediana | Mínimo | Máximo |
|---|---:|---:|---:|---:|
| Reliability | `5` | `0.6420` | `0.5908` | `0.7299` |
| Best predictive correlation | `5` | `0.3410` | `0.3086` | `0.3514` |
| Best Δ mean | `5` | `0.0948` | `0.0854` | `0.1006` |
| Best Δ scrambled | `5` | `0.0684` | `0.0642` | `0.0852` |
| MIS score | `5` | `0.6667` | `0.6667` | `0.6667` |

### Validation en los siete ratones

| Métrica | n | Mediana | Mínimo | Máximo |
|---|---:|---:|---:|---:|
| Best predictive correlation | `7` | `0.3279` | `0.2725` | `0.3523` |
| Best Δ mean | `7` | `0.1024` | `0.0513` | `0.1040` |
| Best Δ scrambled | `7` | `0.0774` | `0.0594` | `0.0902` |
| MIS score | `7` | `0.3333` | `0.3333` | `0.3333` |

## Resultados individuales iniciales

| Artefacto | Eval tier | Reliability | Mean | Stimulus ridge | Context ridge | Best Δ mean | Best Δ scrambled | MIS |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `static22846_test_mis.json` | `test` | `0.6532` | `0.2404` | `0.2964` | `0.3410` | `0.1006` | `0.0684` | fails |
| `static22846_validation_mis.json` | `validation` | `0.0000` | `0.2301` | `0.2962` | `0.3325` | `0.1024` | `0.0775` | fails |
| `static26872_validation_mis.json` | `validation` | `0.0000` | `0.2212` | `0.2725` | unavailable | `0.0513` | `0.0798` | fails |
| `static27204_validation_mis.json` | `validation` | `0.0000` | `0.2383` | `0.2824` | `0.3152` | `0.0769` | `0.0594` | fails |

La tabla completa está en `summary_sensorium2022_static_mis.json`. La tabla
anterior conserva las primeras ejecuciones para trazabilidad.

`validation` no tiene estímulos repetidos, por eso `reliability=0.0` significa
"no estimable con este cálculo", no ausencia demostrada de fiabilidad neural.
Los cinco pretraining mice tienen `test` repetido y pasan el bloque de
reproducibilidad.

## Interpretación crítica

El caso real ya muestra el patrón que necesitamos para la tesis metodológica:

1. Un baseline visual/contextual simple mejora de forma consistente frente a
   media y frente a control permutado.
2. Las covariables conductuales/pupilares mejoran la predicción cuando están
   disponibles; `26872` las tiene zeroed y se trata correctamente como no
   contextual.
3. En los cinco pretraining mice, el target tiene fiabilidad por repeticiones.
4. La conclusión es robusta en la cohorte completa Sensorium 2022: el bloque de
   reproducibilidad y especificidad predictiva puede pasar, pero el MIS global
   no pasa porque falta evidencia mecanística.
5. Aun así, el MIS no pasa porque falta evidencia mecanística:
   estructura, perturbación, restricción causal u OOD formal.

Esto no es todavía una contribución Q1 fuerte. Es una base empírica válida para
el siguiente paso: introducir un modelo visual más competitivo o un baseline
oficial Sensorium y evaluar si su ganancia predictiva cambia o no la conclusión
mecanística.
