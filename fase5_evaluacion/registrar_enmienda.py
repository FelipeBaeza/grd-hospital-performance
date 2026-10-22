"""Registra la enmienda 01 al protocolo de la Fase 5. Se ejecuta desde fase5_evaluacion/.

    python registrar_enmienda.py huellas
        Muestra las huellas que van en la constancia del profesor guía (Parte C). No escribe nada.
    python registrar_enmienda.py firmar --autor "Nombre" --profesor "Nombre" --fecha-firma AAAA-MM-DD [--sin-constancia] [--nota "..."]
        Crea documentos/registro_enmienda_01.json. Exige documentos/constancia_profesor_guia_firmada.pdf, salvo
        --sin-constancia, que lo deja anotado en el registro.
    python registrar_enmienda.py antes-de-correr <script> [<script> ...]
        Antes de la primera corrida: registra la huella de los scripts y comprueba que la enmienda no cambió.

Después de «firmar»: git add documentos/registro_enmienda_01.json && git commit && git tag -a enmienda-01 && git push --follow-tags
"""
import argparse
import sys

from evaluacion import enmienda


def main() -> int:
    ap = argparse.ArgumentParser(description="Registro de la enmienda 01 del protocolo de la Fase 5")
    sub = ap.add_subparsers(dest="accion", required=True)
    sub.add_parser("huellas")
    f = sub.add_parser("firmar")
    f.add_argument("--autor", required=True)
    f.add_argument("--profesor", required=True)
    f.add_argument("--fecha-firma", required=True)
    f.add_argument("--sin-constancia", action="store_true")
    f.add_argument("--nota", default=None, help="texto que queda en el registro, por ejemplo cómo aprobó el profesor")
    c = sub.add_parser("antes-de-correr")
    c.add_argument("scripts", nargs="+")
    a = ap.parse_args()
    try:
        if a.accion == "huellas":
            for ruta, h in enmienda.huellas().items():
                print(f"{ruta}\n  SHA-256: {h}")
        elif a.accion == "firmar":
            print("registrado:", enmienda.firmar(a.autor, a.profesor, a.fecha_firma, sin_constancia=a.sin_constancia, nota=a.nota))
            print("siguiente: git add documentos/registro_enmienda_01.json && git commit -m 'Registra la enmienda 01' "
                  "&& git tag -a enmienda-01 -m 'Enmienda 01 firmada' && git push --follow-tags")
        else:
            print("registrado:", enmienda.antes_de_correr(a.scripts))
    except (enmienda.ErrorEnmienda, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
