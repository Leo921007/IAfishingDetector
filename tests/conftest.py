"""Asegura que la raíz del repo esté en sys.path para importar los módulos del bot."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
