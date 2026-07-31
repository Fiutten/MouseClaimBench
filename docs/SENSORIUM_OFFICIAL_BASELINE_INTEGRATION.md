# Sensorium Official Baseline Integration

## Estado

Se ha integrado el stack oficial de Sensorium 2023 a nivel de smoke test
ejecutable:

- `external/sensorium_2023`: starter kit oficial.
- `external/neuralpredictors_43fa`: source tree del commit de
  `neuralpredictors` referenciado por `requirements.txt`.
- `.venv-sensorium-official`: entorno aislado no versionado.
- `scripts/smoke_sensorium_official_model.py`: forward pass reproducible sobre
  un ratón local de Dynamic Sensorium.

Artefacto:

```text
results/sensorium_official_baseline_audit/official_model_smoke.json
```

Resultado:

```text
official_stack_forward_ok = true
trained_baseline = false
sota_claim = false
```

## Qué sí demuestra

1. El loader oficial `mouse_video_loader` puede leer nuestro directorio local de
   Dynamic Sensorium.
2. Una arquitectura oficial mínima basada en `Factorized3dCore` y readout
   factorised puede instanciarse.
3. El modelo produce un forward pass sobre un batch real:
   - input: `[1, 1, 13, 9, 64]`;
   - response: `[1, 7863, 51]`;
   - output: `[1, 11, 7863]`.

## Qué no demuestra

- No es un baseline oficial entrenado.
- No reproduce el leaderboard Sensorium.
- No es una comparación SOTA.
- No cambia todavía la decisión Q1.

## Bloqueos encontrados

El `requirements.txt` oficial no fue reproducible directamente en este entorno:

1. `pandas==2.0.0` falla bajo Python 3.12/macOS arm64 durante build.
2. El `pyproject.toml` instala `neuralpredictors==0.3.0`, pero esa wheel no
   contiene `neuralpredictors.layers.cores.conv3d`.
3. El código oficial requiere el source tree del commit Git
   `43faededa2d2e76bb904f38a49b9d8b81d287a0a`.
4. `neuralpredictors` usa imports legacy de `collections.Iterable`, por lo que
   el smoke aplica un shim de compatibilidad Python 3.12.

## Reproducibilidad

Para reconstruir el entorno:

```bash
scripts/setup_sensorium_official_env.sh
```

El script descarga fuentes oficiales, crea `.venv-sensorium-official`, instala
dependencias mínimas y ejecuta el smoke test.

## Decisión

El stack oficial queda integrado a nivel técnico, pero para un Q1 fuerte falta
entrenar/evaluar una configuración oficial y generar:

```text
results/sensorium_official_baseline_audit/official_trained_baseline_summary.json
```

Hasta que ese artefacto exista, el baseline NN evaluado sigue siendo
`torch_residual_mlp`, explícitamente no-SOTA.
