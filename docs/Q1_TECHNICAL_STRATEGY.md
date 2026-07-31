# Estrategia técnica para aspirar a Q1

## Diagnóstico honesto

Con Allen VBN solo no tenemos todavía un paper Q1 fuerte de inteligencia
artificial. Tenemos un caso metodológico sólido: un target reproducible que no
es mecanísticamente identificable para conectividad dirigida. Es valioso, pero
demasiado negativo y demasiado neuroinformático si queda aislado.

Para elevarlo, la contribución debe pasar de:

```text
un pipeline que falsifica una hipótesis en Allen VBN
```

a:

```text
un benchmark reproducible que separa predicción, reproducibilidad e
identificabilidad mecanística en modelos parciales de cerebro de ratón.
```

## Línea central propuesta

La contribución defendible sería:

1. Formalizar `Mechanistic Identifiability Score` como criterio falsable.
2. Validarlo en un benchmark sintético con verdad conocida.
3. Mostrar un caso negativo real: Allen VBN.
4. Añadir un caso predictivo moderno: Sensorium/Dynamic Sensorium.
5. Decidir MICrONS solo como piloto estructura-función si aporta una pregunta
   adicional clara.

## Estado actual de los cinco puntos

Actualización 2026-06-19:

La fase de cierre produjo una decisión explícita:

```text
methodological_benchmark_paper_now_q1_requires_external_piece
```

Esto significa que el paquete ya es defendible como benchmark metodológico
reproducible, pero todavía no debe presentarse como Q1 fuerte de IA hasta
integrar un baseline oficial Sensorium o aprobar/ejecutar un piloto MICrONS
estructura-función acotado.

### 1. MIS formalizado

Implementado en:

```text
mousebrainbench/validation/mechanistic_identifiability.py
```

El score es conjuntivo y separa reproducibilidad, especificidad topológica e
identificabilidad dirigida. Esto impide compensar un fallo mecanístico con una
buena correlación funcional.

### 2. Benchmark sintético con verdad conocida

Implementado en:

```text
mousebrainbench/benchmarks/synthetic_identifiability.py
```

Resultado actual:

- `directed_truth`: pasa MIS.
- `common_drive_nonidentifiable`: falla MIS.
- `topology_without_direction`: falla MIS aunque pasa topología.
- `direction_without_topology_specificity`: falla MIS aunque pasa dirección.

Este benchmark protege contra un fallo conceptual importante: declarar mecanismo
cuando solo hay una respuesta común reproducible, una topología parcial o una
firma temporal parcial.

### 3. Allen VBN como caso negativo

Implementado en:

```text
mousebrainbench/benchmarks/allen_vbn_mis.py
```

Resultado actual:

- MIS global: no pasa.
- Reproducibilidad: pasa.
- Especificidad topológica: falla.
- Identificabilidad dirigida: falla.

Interpretación:

```text
reproducible_target_without_mechanistic_identifiability
```

Allen VBN queda cerrado como evidencia negativa real, no como base para seguir
forzando una interpretación anatómica.

### 4. Sensorium/Dynamic Sensorium como caso predictivo moderno

Decisión: entra como siguiente fase técnica si queremos subir ambición.

Razón:

- Sensorium está diseñado como benchmark de predicción neural en ratón.
- Dynamic Sensorium añade estímulos naturales dinámicos y evaluación más cercana
  al problema moderno de modelos predictivos.
- La literatura reciente de digital twins visuales de ratón está creciendo, por
  lo que necesitamos compararnos con ese eje y no solo con Allen VBN.

Uso correcto dentro del proyecto:

1. No empezar entrenando modelos profundos desde cero.
2. Construir primero adaptadores de datos y métricas.
3. Cargar splits y targets oficiales cuando estén disponibles.
4. Evaluar baselines transparentes:
   - media por neurona;
   - ridge/PLS;
   - baseline convolucional mínimo solo si el dataset lo justifica.
5. Aplicar MIS en una variante adaptada:
   - reproducibilidad/predicción;
   - especificidad de entrada visual;
   - estabilidad fuera de distribución;
   - solo hablar de mecanismo si hay perturbación, estructura o dirección
     justificable.

Resultado esperado útil:

- Si Sensorium predice bien pero no pasa una puerta mecanística, refuerza la
  tesis: predicción no equivale a mecanismo.
- Si un subconjunto pasa una puerta de identificabilidad parcial, tendríamos el
  caso positivo que falta para una propuesta Q1 más fuerte.

Estado implementado:

- loader compatible con la estructura oficial de zips Sensorium/Dynamic;
- benchmark predictivo con `mean_response`, `stimulus_ridge` y control
  `scrambled_stimulus_ridge`;
- smoke sintético que demuestra alta correlación sin identificabilidad
  mecanística;
- protocolo en [PHASE5_SENSORIUM_PROTOCOL.md](PHASE5_SENSORIUM_PROTOCOL.md).
- baseline NN local `torch_residual_mlp` evaluado en Dynamic Sensorium principal
  y legacy OOD;
- integración oficial Sensorium 2023 a nivel de smoke test: el loader oficial
  abre datos locales y una arquitectura oficial ejecuta forward pass sobre un
  batch real;
- auditoría oficial Sensorium que concluye:
  `official_sensorium_stack_integrated_training_pending`.

La siguiente acción empírica ya no es descargar más ratones. La pieza que falta
para Q1 es entrenar/evaluar una configuración oficial reproducible del ecosistema
Sensorium y pasarla por MouseBrainBench. El forward-pass smoke test no basta
para claim Q1.

### 5. MICrONS como caso estructura-función

Decisión: no debe entrar como eje principal todavía.

Razón:

- Es científicamente atractivo, pero pesado en datos, dependencias y curación.
- Puede abrir un frente demasiado costoso antes de cerrar Sensorium.
- El riesgo de prometer más de lo que podemos reproducir localmente es alto.

Uso correcto:

MICrONS debe quedar como piloto opcional de estructura-función, con criterios de
entrada estrictos:

1. Usar solo un subconjunto manejable y públicamente documentado.
2. No intentar reconstruir ni descargar todo el recurso.
3. Formular una pregunta estrecha:
   - ¿mejora la estructura local la predicción funcional frente a controles?
   - ¿qué nivel de conectividad local aporta información no capturada por un
     baseline funcional?
4. Abandonar MICrONS si el coste de integración supera el valor diferencial
   frente a Sensorium.

Estado actual:

- puerta MICrONS implementada;
- no hay manifiesto local pequeño;
- decisión: `defer_microns_until_small_manifest_available`.

## Qué hay que hacer ahora

La siguiente fase técnica queda reducida a dos rutas, no a exploración abierta:

1. **Ruta Q1 externa:** entrenar/evaluar un baseline oficial Sensorium en entorno
   reproducible y pasar sus predicciones por MouseBrainBench.
2. **Ruta Q1 estructura-función:** localizar o construir un manifiesto MICrONS
   pequeño que cumpla la puerta `microns_pilot_gate`.

Sin una de esas dos rutas, se debe avanzar a paper metodológico fuerte, no a Q1
de alto impacto en IA.

## Criterio de avance hacia Q1

No avanzamos a escritura Q1 hasta tener al menos:

- un caso sintético con verdad conocida;
- un caso real negativo cerrado;
- un caso real predictivo moderno;
- una comparación clara entre predicción y mecanismo;
- scripts reproducibles para regenerar tablas y artefactos.

Si Sensorium no aporta una diferencia clara, el proyecto sigue siendo publicable
como metodología/neuroinformatics, pero no como Q1 fuerte de IA.

## Literatura que condiciona la estrategia

- Sensorium define un benchmark moderno de predicción neural en corteza visual
  de ratón y, por tanto, es el comparador natural para una línea IA/predicción.
- Dynamic Sensorium extiende ese marco a estímulos dinámicos, más cercano a
  evaluación temporal y generalización fuera de distribución.
- Los trabajos recientes sobre digital twins visuales de ratón elevan el listón:
  no basta con mostrar correlaciones; hay que separar predicción, generalización
  y explicación mecanística.
- MICrONS aporta escala estructura-función local, pero no es un cerebro completo
  de ratón y no debe convertirse en el primer frente de integración salvo piloto.

Referencias de partida:

- Sensorium: <https://arxiv.org/abs/2206.08666>
- Dynamic Sensorium: <https://arxiv.org/abs/2305.19654>
- Sensorium 2023 retrospective: <https://arxiv.org/abs/2407.09100>
- Mouse V1 digital twins: <https://arxiv.org/abs/2605.23122>
- MICrONS cortical inductive biases: <https://arxiv.org/abs/2606.14975>
