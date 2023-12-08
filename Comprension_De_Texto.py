
# Librerias

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from textwrap import wrap

# Uso de modelo preentrenado

modelo_preentrenado = "mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
tokenizer = AutoTokenizer.from_pretrained(modelo_preentrenado, do_lower_case = False)

# Modelo

modelo = AutoModelForQuestionAnswering.from_pretrained(modelo_preentrenado)
print(f"Modelo: \n{modelo}")

# Tokenizacion

contexto = "Yo soy Bredalis y mi banda favorita de kpop es BTS"
pregunta = "¿Que banda de kpop me gusta?"

codificador = tokenizer.encode_plus(pregunta, contexto, return_tensors = "pt")
input_ids = codificador["input_ids"].tolist()
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

# Mostrar los tokens

for id, token in zip(input_ids[0], tokens):
  print("\n{:<12} {:>6}".format(token, id))

# Ejemplo de inferencia (pregunta - respuesta)

nlp = pipeline("question-answering", model = modelo, tokenizer = tokenizer)
output = nlp({"question": pregunta, "context": contexto})
print(f"\nRespuesta: {output}")

# Prueba del modelo

def question_answering(contexto):

  espacio = '-----------------'

  # Mostrar contexto

  print('\nContexto:')
  print(espacio)
  print('\n'.join(wrap(contexto)))

  # Bucle (pregunta - respuesta):

  continua = True

  while continua:

    print('\nPregunta:')
    print(espacio)
    pregunta = str(input())

    continua = pregunta != ''

    if continua:

      print('\nRespuesta:')
      print(espacio)
      print(output['answer'])

contexto = "BTS (en hangul, 방탄소년단; romanización revisada del coreano, Bangtan Sonyeondan; literalmente, «Bulletproof Boy Scouts»), también conocido como Bangtan Boys, es un grupo surcoreano formado en 2010. Está compuesto por siete integrantes: Jin, Suga, J-Hope, RM, Jimin, V y Jungkook, quienes escriben y producen la mayor parte de su material discográfico. A pesar de haber sido creado con un estilo principalmente hip hop, ha llegado a incorporar una gran variedad de géneros en su repertorio musical."

question_answering(contexto)