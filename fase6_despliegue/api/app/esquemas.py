from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class PerfilDemo(BaseModel):
    """Perfil clínico SINTÉTICO. No se guarda, no se registra y no identifica a nadie."""
    sexo: str
    edad: int = Field(ge=18, le=110)
    tipo_ingreso: str
    tipo_procedencia: str
    diagnostico_principal: str = Field(min_length=3, max_length=3)
    ingreso_fin_de_semana: bool = False
    cuidado_intensivo_al_ingreso: bool = False
    comorbilidades: list[str] = Field(default_factory=list, max_length=31)

    @field_validator("comorbilidades")
    @classmethod
    def _unicas(cls, v: list[str]) -> list[str]:
        return sorted(set(v))
