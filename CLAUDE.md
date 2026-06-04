# Reglas permanentes del proyecto — IAfishingDetector

## Convención de commits y Git

- **Autorización permanente:** puedes ejecutar `git add`, `git commit` y `git push` sin
  pedir confirmación.
- **Commits atómicos:** un commit por cada cambio lógico coherente.
- **Mensajes en español, formato Conventional Commits:**
  `tipo: resumen breve en imperativo`
  - Tipos permitidos: `feat`, `fix`, `refactor`, `perf`, `docs`, `test`, `chore`, `data`, `train`.
  - Cuerpo opcional con bullets concisos del **qué** y el **porqué**.
- **PROHIBIDO** incluir cualquier referencia a Anthropic, Claude, "AI", "Generated with",
  emojis de robot o trailers `Co-Authored-By`. Los commits son **exclusivamente de mi autoría**.
- **No cambies** la configuración `user.name` / `user.email` de git.
- **Antes de cada `push`,** muestra en el chat un resumen de los commits que vas a subir.

## Estructura del repositorio (Etapa R1)

Mantené la **raíz mínima**. En la raíz solo van: `main.py` (entrypoint), `config.yaml`,
`CLAUDE.md`, `README.md`, `.gitignore`, `requirements*.txt` y las carpetas.

- **`pesca/`** — paquete **runtime** (lo importa `main.py`): `config`, `corcho_detector`,
  `bite_trigger`, `splash`, `logging_setup`, `platform_io`, `session`. Imports internos
  `from pesca.<mod> import ...`. `pesca/config.py` resuelve todas las rutas desde la raíz
  (`REPO_ROOT = Path(__file__).resolve().parents[1]`).
- **`tools/`** — herramientas de desarrollo (standalone). **Correr con `python -m tools.<script>`**
  desde la raíz del repo (así `pesca.*` y `tools.*` resuelven). Importan `from pesca.<mod> import ...`.
- **`docs/`** — documentación (`AUDITORIA.md`, `ETAPA*.md`, `PLAN_ENTRENAMIENTO.md`, métricas).
- **`legacy/`** — código/artefactos **muertos** (audio viejo, yolov5). No se borra; no se importa.
- **`.claude/skills/`** — skills de Claude Code (versionadas; se llenan en la Etapa R2).
- **Sin cambios:** `tests/`, `locations/` (modelo+ROI por ubicación), `models/`, `data/`, `dataset/`.

**Reglas:** runtime nuevo → `pesca/`; script de desarrollo nuevo → `tools/` (con `from pesca...`);
documentación → `docs/`. `python main.py` **no cambia**. No metas `.py` sueltos en la raíz.
