##  Descrição do modelo
Este é um modelo pré-treinado disponível por meio da biblioteca [TorchXRayVision](https://github.com/mlmed/torchxrayvision). Ele é capaz de detectar 18 doenças de tórax.

## Doenças detectadas
- Atelectasia
- Cardiomegalia
- Cardiomediastino aumentado
- Consolidação
- Derrame pleural (Efusão)
- Edema
- Enfisema
- Espessamento pleural
- Fibrose
- Fratura
- Hérnia
- Infiltração
- Lesão pulmonar
- Massa
- Nódulo
- Opacidade pulmonar
- Pneumonia
- Pneumotórax

## Entrada
- Imagem de raio-x **frontal** de tórax, preferencialmente em formato 224x224

## Saída
- Probabilidade de cada doença detectada com precisão acima do limite definido pelo usuário

## Exemplo de entrada e saída  :
