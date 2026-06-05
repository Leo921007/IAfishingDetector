# Skills del proyecto

Registro de las *agent skills* del repositorio: lo que está **instalado** y las **candidatas**
descubiertas (sin instalar) para la fase de entrenamiento/validación
(`docs/PLAN_ENTRENAMIENTO.md`). El usuario decide cuáles instalar.

## Instalado

| Skill | Origen | Para qué |
|-------|--------|----------|
| `find-skills` | `vercel-labs/skills` | Descubrir e instalar skills del ecosistema skills.sh (`npx skills find <kw>` / `add`). |

## Convención

- **Ubicación**: las skills del proyecto viven en **`.claude/skills/<nombre>/SKILL.md`**
  (scope de proyecto; el scope global sería `~/.claude/skills/`). `.gitignore` **no** excluye
  `.claude/`, así que se versionan.
- **Archivos reales, no symlinks**: se instalan con `--copy` para poder commitearlas.
- **Manifiesto**: `skills-lock.json` (raíz) registra qué skills hay y su hash, para
  reproducir/actualizar (`npx skills update`). Análogo a un lockfile de dependencias.
- **Invocación**: Claude Code reconoce automáticamente las skills bajo `.claude/skills/`;
  se activan por su `description` cuando la tarea encaja.
- **Instalar una candidata**: `npx skills add <owner/repo@skill> --yes --copy`.

## Candidatas descubiertas (NO instaladas — pendientes de aprobación)

> **Honestidad del registro:** skills.sh está dominado por skills de LLM/agentes y de SaaS.
> **No existe** una skill dedicada a entrenar **YOLO/ultralytics** ni a calcular **métricas de
> detección (mAP/precision/recall) a nivel de evento**. Esa parte del plan (§6 entrenamiento, §7
> arnés de validación) **sigue siendo código propio**, como ya anticipa el Apéndice del plan.
> Lo que sí aporta valor real es **tracking de experimentos**, **datasets** y **export ONNX**.

| # | Skill (`owner/repo@skill`) | Categoría | Qué hace | Encaja con (PLAN) | Instalar | Confianza |
|---|----------------------------|-----------|----------|-------------------|----------|-----------|
| 1 | `wandb/skills@wandb-primary` | Tracking de experimentos | W&B oficial: loguear runs, métricas, hiperparámetros y artefactos/datasets versionados; comparar corridas. | §6 (bake-off A vs B), §7.3 (barrido de umbral), §9 (fases) | `npx skills add wandb/skills@wandb-primary --yes --copy` | **Alta** |
| 2 | `huggingface/skills@hugging-face-trackio` | Tracking de experimentos | Trackio: tracking ligero y local estilo W&B (HF). | §6, §9 | `npx skills add huggingface/skills@hugging-face-trackio --yes --copy` | Media |
| 3 | `huggingface/skills@huggingface-datasets` | Datasets / versionado | Gestionar y versionar datasets vía HF Hub. | §3.4 (split por sesión), §2 (dataset por ubicación) | `npx skills add huggingface/skills@huggingface-datasets --yes --copy` | Media |
| 4 | `jeremylongshore/claude-code-plugins-plus-skills@onnx-converter` | Export ONNX | Convertir modelos a formato ONNX. | §6 (salida `locations/<loc>/detector.onnx`), §2 | `npx skills add jeremylongshore/claude-code-plugins-plus-skills@onnx-converter --yes --copy` | Media |
| 5 | `mlflow/skills@mlflow-onboarding` | Tracking de experimentos | MLflow: tracking de experimentos/modelos (foco fuerte en tracing/observabilidad de LLM, menos en visión). | §9 | `npx skills add mlflow/skills@mlflow-onboarding --yes --copy` | Media-baja |
| 6 | `nvidia/skills@physical-ai-video-data-augmentation` | Augmentation de visión | Augmentation de datos de vídeo (physical AI, NVIDIA). No es fotométrico-para-detección a medida, pero es lo más cercano del registro. | §6 (augmentation fotométrico/ruido), §7.2 (por condición) | `npx skills add nvidia/skills@physical-ai-video-data-augmentation --yes --copy` | Media-baja |
| 7 | `jeremylongshore/claude-code-plugins-plus-skills@data-augmentation-pipeline` | Augmentation (genérico) | Pipeline genérico de data augmentation (no específico de imágenes/detección). | §6 | `npx skills add jeremylongshore/claude-code-plugins-plus-skills@data-augmentation-pipeline --yes --copy` | Baja |
| 8 | `davila7/claude-code-templates@senior-computer-vision` | Asesoría CV (entrenamiento) | Agente asesor de visión por computadora; orientación general, **no** pipeline YOLO a medida. | §6 (asesoría) | `npx skills add davila7/claude-code-templates@senior-computer-vision --yes --copy` | Baja |
| 9 | `modelscope.cn@glmv-grounding` | Etiquetado / grounding | Grounding visual (cajas) con GLM-V; posible apoyo a auto-anotación. | §5 (etiquetado), §3.2 (etiquetador 2 clases) | `npx skills add modelscope.cn@glmv-grounding --yes --copy` | Baja |
| 10 | `seb1n/awesome-ai-agent-skills@data-labeling` | Etiquetado (genérico) | Etiquetado de datos genérico (no bbox de detección a medida). | §5, §3.2 | `npx skills add seb1n/awesome-ai-agent-skills@data-labeling --yes --copy` | Baja |

### Categorías sin candidata útil (resultado honesto del registro)

- **Entrenamiento YOLO/ultralytics (§6)** → **ninguna específica.** `npx skills find ultralytics` no
  devuelve nada; los resultados de `yolo` son del *modo YOLO* (auto-aprobar acciones), no del modelo de
  detección. Solo hay asesoría general de CV (filas 8). El entrenamiento sigue siendo `train_corcho_zona.py`.
- **Métricas/evaluación de detección a nivel de evento (§7, §7.4, §8)** → **ninguna.** `evaluation`,
  `metrics`, `model evaluation` devuelven solo eval de LLM o métricas de negocio; `mAP` colisiona con
  "map" (firecrawl/mapbox). El **arnés de validación a nivel de evento es código propio** (Apéndice).

## Cómo se obtuvo

Descubrimiento con la skill `find-skills` vía `npx skills find <keyword>` sobre las keywords de cada
categoría del plan: `yolo`, `ultralytics`, `object detection`, `training`, `labeling`, `annotation`,
`bounding box`, `augmentation`, `albumentations`, `evaluation`, `metrics`, `mAP`, `model evaluation`,
`onnx`, `inference`, `dataset`, `dvc`, `experiment tracking`, `mlflow`, `wandb`, `computer vision`.
