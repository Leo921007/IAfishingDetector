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
