import operator
from typing import Annotated, TypedDict, Union
from langgraph.graph import StateGraph, END

# 1. Definimos el Estado del Agente
class AgentState(TypedDict):
    sintomas_actuales: str
    especialidad_sugerida: str
    confianza: float
    preguntas_adicionales: list
    finalizado: bool

# 2. Nodo: Captura de Voz (usando tu código genérico)
def nodo_escuchar(state: AgentState):
    # Aquí llamarías a tu función capturar_voz_a_texto()
    nueva_entrada = "Me duele el pecho y el brazo izquierdo" 
    return {"sintomas_actuales": nueva_entrada}

# 3. Nodo: Clasificación (usando tu RNA)
def nodo_clasificar(state: AgentState):
    # Aquí llamarías a rna_predict(state['sintomas_actuales'])
    especialidad, confianza = "Cardiología", 85.0
    return {
        "especialidad_sugerida": especialidad,
        "confianza": confianza,
        "finalizado": confianza > 70.0  # Si es alta, terminamos
    }

# 4. Construcción del Grafo
workflow = StateGraph(AgentState)

# Añadimos los nodos
workflow.add_node("capturar_voz", nodo_escuchar)
workflow.add_node("clasificador_rna", nodo_clasificar)

# Definimos las conexiones (Edges)
workflow.set_entry_point("capturar_voz")
workflow.add_edge("capturar_voz", "clasificador_rna")

# Lógica condicional: ¿Necesitamos más info?
workflow.add_conditional_edges(
    "clasificador_rna",
    lambda x: "finalizar" if x["finalizado"] else "reintentar",
    {
        "finalizar": END,
        "reintentar": "capturar_voz" # Vuelve a preguntar si la confianza es baja
    }
)

app = workflow.compile()