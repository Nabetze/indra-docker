from langchain_openai import ChatOpenAI
import os
from flask import Flask, jsonify, request
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import create_react_agent


## datos de trazabilidad
os.environ["LANGSMITH_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "gcpaiagent"
os.environ["OPENAI_API_KEY"] ="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"


app = Flask(__name__)

@app.route('/agent', methods=['GET'])
def main():
    #Capturamos variables enviadas
    id_agente = request.args.get('idagente')
    msg = request.args.get('msg')
    #datos de configuracion
    DB_URI = os.environ.get(
        "DB_URI",
        "postgres://postgres:1234@34.173.40.179:5432/"
        #"postgresql://usuario:password@0.0.0.0:5432/basededatos?sslmode=disable"
    )
    connection_kwargs = {
        "autocommit": True,
        "prepare_threshold": 0,
    }
    db_query = ElasticsearchStore(
        es_url="http://34.125.83.15:9200",
        es_user="elastic",
        es_password="L9bWwXp639bERknDgMxK",
        index_name="menu-data",
        embedding=OpenAIEmbeddings(model="text-embedding-3-large"))

    # Herramienta RAG
    retriever = db_query.as_retriever()
    tool_rag =retriever.as_tool(
    name ="datos_menu",
    description="Herramienta que tiene la información detallada (nombre, descripcion, precio, ingredientes, disponibilidad y categoria) de todos los productosdel menú"
    )

    stock_actual = {
    "Hamburguesa Clásica": 4,
    "Tacos al Pastor": 0,
    "Ensalada Vegana": 10,
    "Pizza Margarita": 2,
    "Smoothie de Frutas": 0,
    }

    from langchain.agents import tool

    @tool
    def verificar_stock(nombre_plato: str) -> str:
        """
        Verifica la si un plato está disponible en el menú y la cantidad disponible del plato.
        """
        nombre_plato = nombre_plato.strip()
        if nombre_plato in stock_actual:
            cantidad = stock_actual[nombre_plato]
            if cantidad > 0:
                return f"✅ El plato '{nombre_plato}' está disponible. Quedan {cantidad} unidades."
            else:
                return f"❌ Lo siento, el plato '{nombre_plato}' no está disponible actualmente."
        else:
            return f"🤔 No encontré el plato '{nombre_plato}' en el inventario."

    cupones_validos = {
    "FAMILIA20": 0.20,
    "BEBIDA10": 0.10,
    "VEGANO15": 0.15,
    }

    @tool
    def aplicar_cupon(nombre_cupon: str, total: float) -> str:
        """
        Aplica un cupón de descuento al total si es válido.
        """
        nombre_cupon = nombre_cupon.strip().upper()
        if nombre_cupon in cupones_validos:
            descuento = cupones_validos[nombre_cupon]
            nuevo_total = round(total * (1 - descuento), 2)
            return f"✅ Cupón aplicado. Descuento del {int(descuento*100)}%. Total final: ${nuevo_total}"
        else:
            return f"❌ El cupón '{nombre_cupon}' no es válido."

    @tool
    def notificar_camarero(motivo: str = "Asistencia general") -> str:
        """
        Tool para notificar al camarero que el cliente necesita asistencia.
        Puede indicar el motivo: dificultad visual, confusión, asistencia para ordenar, etc.
        """
        import datetime
        import requests

        hora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        payload = {
            "motivo": motivo,
            "hora": hora,
            "estado": "pendiente"
        }

        # Reemplaza con tu URL de webhook.site
        webhook_url = "	https://webhook.site/64914f9d-304a-4ac9-bb0a-6ce251d91f61"

        try:
            requests.post(webhook_url, json=payload)
            return f"✅ Notificación enviada al camarero. Motivo: '{motivo}' a las {hora}."
        except:
            return "⚠️ No se pudo enviar la notificación. Inténtalo de nuevo."

    # Inicializamos la memoria
    with ConnectionPool(
            # Example configuration
            conninfo=DB_URI,
            max_size=20,
            kwargs=connection_kwargs,
    ) as pool:
        checkpointer = PostgresSaver(pool)

        # Inicializamos el modelo
        model = ChatOpenAI(model="gpt-4.1-2025-04-14")

        # Agrupamos las herramientas
        tolkit = [tool_rag, verificar_stock, aplicar_cupon]

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 """
                Eres un asistente virtual amigable y accesible especializado en guiar a personas con discapacidad visual a través del menú de un restaurante.

                Utiliza únicamente las herramientas disponibles para responder: consulta de menú (RAG), verificación de stock, aplicación de cupones y recomendación de platos similares. Si no cuentas con una herramienta específica para responder una pregunta, indícalo de forma clara y sugiere cómo podrías ayudar.

                Tu objetivo es que el cliente se sienta acompañado, informado y en control de su pedido. Sé breve, claro, cálido y evita tecnicismos innecesarios. Eres paciente, empático y actúas como un asistente personal que lo guía en todo momento.

                Sigue esta guía de conversación:

                1. **Saludo y apertura:**
                Da una bienvenida cordial. Pregunta si el cliente ya tiene en mente algún plato o si quiere que le leas opciones del menú por categoría (ej. entradas, principales, postres, vegetarianos, etc.).

                2. **Consulta del menú:**
                Usa la herramienta de menú para brindar información precisa. Describe brevemente cada plato mencionado: nombre, ingredientes principales, precio, y si forma parte de algún combo. Si el cliente tiene preferencias (ej. sin carne, sin lactosa), sugiere opciones que se alineen.

                3. **Verificación de disponibilidad:**
                Antes de confirmar un pedido, verifica si los platos solicitados están disponibles en stock. Si algo no está disponible, sugiere una alternativa similar usando la herramienta de recomendación.

                4. **Aplicación de cupones o promociones:**
                Si el cliente menciona un cupón o desea saber si hay promociones, usa la herramienta para aplicar descuentos o combos y muestra el precio actualizado.

                5. **Resumen del pedido:**
                Repite lo que ha seleccionado el cliente con los precios individuales y el total final. Pregunta si desea añadir algo más antes de cerrar el pedido.

                6. **Cierre de atención:**
                Agradece su tiempo y disponibilidad. Indícale cómo puede proceder para finalizar el pedido (ej. llamar al restaurante, pagar presencialmente o pedir ayuda en el local). Si el cliente necesita ayuda adicional, ofrécesela de inmediato.

                7. **Estilo de comunicación:**
                Sé cercano, conversacional y directo. Mantén frases cortas y claras. Si el cliente no entiende algo o parece confundido, repite o simplifica tu respuesta. Tu prioridad es la comodidad y accesibilidad del cliente.

                 """),
                ("human", "{messages}"),
            ]
        )
        # inicializamos el agente
        agent_executor = create_react_agent(model, tolkit, checkpointer=checkpointer, prompt=prompt)
        # ejecutamos el agente
        config = {"configurable": {"thread_id": id_agente}}
        response = agent_executor.invoke({"messages": [HumanMessage(content=msg)]}, config=config)
        return response['messages'][-1].content


if __name__ == '__main__':
    # La aplicación escucha en el puerto 8080, requerido por Cloud Run
    app.run(host='0.0.0.0', port=8080)