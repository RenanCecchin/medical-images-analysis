##  Descrição do modelo
Este é um modelo pré-treinado disponível por meio da biblioteca [TorchXRayVision](https://github.com/mlmed/torchxrayvision). Ele é capaz de detectar 17 doenças de tórax.

## Doenças detectadas
- Atelectasia
- Cardiomegalia
- Consolidação
- Derrame pleural (Efusão)
- Edema
- Enfisema
- Espessamento pleural
- Fibrose
- Fratura
- Hérnia
- Infiltração
- Massa
- Nódulo
- Opacidade pulmonar
- Pneumonia
- Pneumotórax

## Entrada
- Imagem de raio-x **frontal** de tórax, preferencialmente em formato 512x512

## Saída
- Probabilidade de cada doença detectada com precisão acima do limite definido pelo usuário

## Exemplo de entrada e saída  :
