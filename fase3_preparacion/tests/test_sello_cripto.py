"""Plan 4.5 (H-16): sin la clave no se puede leer el desenlace sellado; con ella se recupera idéntico."""
import pytest

from prep import sello_cripto as sc

CLAVE = "clave-de-prueba-larga"


def test_sellar_y_abrir(tmp_path):
    orig = tmp_path / "desenlace_2025.parquet"
    orig.write_bytes(b"PAR1 datos sinteticos del desenlace " * 50)
    sellado = sc.sellar(orig, CLAVE)
    assert b"datos sinteticos" not in sellado.read_bytes()
    with pytest.raises(sc.SelloError):
        sc.sellar(orig, CLAVE)                                   # no se sobrescribe
    with pytest.raises(sc.SelloError):
        sc.abrir(sellado, tmp_path / "x.parquet", "otra-clave-equivocada")
    abierto = sc.abrir(sellado, tmp_path / "abierto.parquet", CLAVE)
    assert abierto.read_bytes() == orig.read_bytes()


def test_un_sellado_alterado_no_se_abre(tmp_path):
    orig = tmp_path / "d.parquet"
    orig.write_bytes(b"x" * 1000)
    sellado = sc.sellar(orig, CLAVE)
    b = bytearray(sellado.read_bytes())
    b[-1] ^= 1
    sellado.write_bytes(bytes(b))
    with pytest.raises(sc.SelloError, match="no es el registrado"):
        sc.abrir(sellado, tmp_path / "y.parquet", CLAVE)
