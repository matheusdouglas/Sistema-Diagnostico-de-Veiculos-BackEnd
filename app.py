from dotenv import load_dotenv
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import openai
import logging
import re

load_dotenv()

# Configurar o logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)  # Configurar o CORS

client = openai.api_key = os.getenv("OPENAI_API_KEY")

class VehicleDiagnosticSystem:
    def __init__(self, csv_file):
        self.codes = self.load_codes(csv_file)
        self.diagnostics_history = []

    def load_codes(self, csv_file):
        """Carrega os códigos de falha do arquivo CSV usando pandas."""
        try:
            df = pd.read_csv(csv_file)
            if 'code' in df.columns and 'description' in df.columns:
                codes = dict(zip(df['code'], df['description']))
            else:
                raise KeyError("As colunas 'code' ou 'description' não foram encontradas no arquivo CSV.")
            return codes
        except FileNotFoundError:
            logging.error(f"Arquivo {csv_file} não encontrado.")
            return {}

    def search_code(self, code):
        """Busca a descrição de um código de falha OBD2."""
        description = self.codes.get(code.upper(), None)
        if description:
            return f"Código: {code.upper()}\nDescrição: {description}"
        else:
            return f"Código {code.upper()} não encontrado."

    def diagnose_vehicle(self, customer_complaint, method_choice, dtc_code, related_symptoms, problem_area):
        """Realiza o diagnóstico do veículo baseado nas informações inseridas."""
        diagnostic_methods = [
            "Verificar a Reclamação",
            "Determinar os Sintomas Relacionados",
            "Analisar os Sintomas Relacionados",
            "Isolar a Área do Problema",
            "Reparar a Área do Problema",
            "Certificar que a Operação Está Adequada"
        ]
        selected_method = diagnostic_methods[method_choice - 1]

        dtc_description = self.search_code(dtc_code) if dtc_code else "N/A"

        diagnosis = {
            "Reclamação do Cliente": customer_complaint,
            "Método de Diagnóstico": selected_method,
            "Código de Falha": dtc_code,
            "Descrição do Código": dtc_description,
            "Sintomas Relacionados": related_symptoms,
            "Área do Problema": problem_area
        }
        self.diagnostics_history.append(diagnosis)

        chatbot_suggestion = self.get_chatbot_solution(diagnosis)
        return {
            "diagnosis": diagnosis,
            "suggestion": chatbot_suggestion
        }

    def get_chatbot_solution(self, diagnosis):
        """Chama o chatbot da OpenAI para fornecer uma solução baseada no diagnóstico."""
        assistant = client.beta.assistants.create(
            name="Vehicle Diagnostic Assistant",
            instructions="You are a vehicle diagnostic assistant. Provide the most accurate vehicle problem diagnosis and solutions. In case of questions that are not related to your expertise as a vehicle diagnostic assistant answer just -Eu não posso te ajudar com isso-. Your response should be less than or equal to 300 words.",
            tools=[{"type": "code_interpreter"}],
            model="gpt-4o-mini",
        )

        thread = client.beta.threads.create()

        prompt = (
            f"Cliente reclamou de: {diagnosis['Reclamação do Cliente']}\n"
            f"Método de Diagnóstico: {diagnosis['Método de Diagnóstico']}\n"
            f"Código de Falha: {diagnosis['Código de Falha']}\n"
            f"Descrição do Código: {diagnosis['Descrição do Código']}\n"
            f"Sintomas Relacionados: {diagnosis['Sintomas Relacionados']}\n"
            f"Área do Problema: {diagnosis['Área do Problema']}\n"
            f"Com base nessas informações, quais podem ser as causas do problema e como ele pode ser resolvido?"
        )

        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt,
        )

        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=assistant.id,
            instructions="Please address the user as 'Mecânico'.",
        )

        if run.status == "completed":
            messages = client.beta.threads.messages.list(thread_id=thread.id)
            suggestion = "\n".join([msg.content[0].text.value for msg in messages if msg.content[0].type == "text"])
            # Remove Markdown formatting
            suggestion = re.sub(r"[*_#`~]", "", suggestion)
        else:
            suggestion = "Ocorreu um erro ao processar a solicitação. Tente novamente."

        client.beta.assistants.delete(assistant.id)

        return suggestion

    def view_history(self):
        """Visualiza o histórico de diagnósticos realizados."""
        return self.diagnostics_history

    def generate_report(self):
        """Gera um relatório básico do último diagnóstico realizado."""
        if not self.diagnostics_history:
            return "Nenhum diagnóstico disponível para gerar relatório."

        last_diagnosis = self.diagnostics_history[-1]
        report = "\n--- Relatório de Diagnóstico ---\n"
        for key, value in last_diagnosis.items():
            report += f"{key}: {value}\n"

        with open("diagnostic_report.txt", "w") as file:
            file.write(report)

        return "Relatório gerado e salvo como 'diagnostic_report.txt'"

# Instanciar o sistema
system = VehicleDiagnosticSystem('OBD2.csv')

@app.route('/diagnose', methods=['POST'])
def diagnose_vehicle():
    data = request.json
    logging.info(f"Diagnóstico solicitado com dados: {data}")
    response = system.diagnose_vehicle(
        customer_complaint=data.get('customer_complaint'),
        method_choice=data.get('method_choice'),
        dtc_code=data.get('dtc_code'),
        related_symptoms=data.get('related_symptoms'),
        problem_area=data.get('problem_area')
    )
    logging.info(f"Resposta do diagnóstico: {response}")
    return jsonify(response)

@app.route('/search_code', methods=['GET'])
def search_code():
    code = request.args.get('code')
    logging.info(f"Buscando código OBD2: {code}")
    result = system.search_code(code)
    logging.info(f"Descrição do código encontrado: {result}")
    return jsonify({"result": result})

@app.route('/view_history', methods=['GET'])
def view_history():
    history = system.view_history()
    logging.info(f"Histórico de diagnósticos: {history}")
    return jsonify(history)

@app.route('/codes', methods=['GET'])
def get_all_codes():
    """Retorna a lista de todos os códigos e descrições."""
    logging.info("Solicitação para buscar todos os códigos.")
    codes = [{"code": code, "description": desc} for code, desc in system.codes.items()]
    return jsonify(codes)


@app.route('/generate_report', methods=['GET'])
def generate_report():
    result = system.generate_report()
    logging.info(f"Resultado da geração do relatório: {result}")
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)
