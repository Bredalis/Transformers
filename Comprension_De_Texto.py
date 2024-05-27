
# Librerias

from textwrap import wrap
from transformers import (AutoTokenizer, 
AutoModelForQuestionAnswering, pipeline)  

# Uso de modelo preentrenado

url = "mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
modelo_preentrenado = url

tokenizer = AutoTokenizer.from_pretrained(
  modelo_preentrenado, do_lower_case = False)

# Modelo

modelo = AutoModelForQuestionAnswering.from_pretrained(modelo_preentrenado)
print(f"Modelo: \n{modelo}")

# Tokenizacion

with open("Contexto.txt", "r") as archivo:
  contexto = archivo.read()

pregunta = input("")

codificador = tokenizer.encode_plus(pregunta, contexto, return_tensors = "pt")
input_ids = codificador["input_ids"].tolist()
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

# Mostrar los tokens

for id, token in zip(input_ids[0], tokens):
  print("\n{:<12} {:>6}".format(token, id))

# Ejemplo de inferencia (pregunta - respuesta)

nlp = pipeline("question-answering", model = modelo, tokenizer = tokenizer)
output = nlp({"question": pregunta, "context": contexto})
print(f"\nRespuesta: {output["answer"]}")