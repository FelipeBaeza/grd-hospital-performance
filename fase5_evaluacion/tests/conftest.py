import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluacion.config  # noqa: F401  (agrega las Fases 3 y 4 al path)
