##  Descrição do modelo
Este é um modelo pré-treinado disponível por meio da biblioteca [TorchXRayVision](https://github.com/mlmed/torchxrayvision). Ele foi treinado com o dataset [CheXpert](https://arxiv.org/abs/1901.07031) e é capaz de detectar 7 doenças de tórax.

## Doenças detectadas
- Atelectasia
- Cardiomegalia
- Consolidação
- Derrame pleural (Efusão)
- Edema
- Pneumonia
- Pneumotórax

## Entrada
- Imagem de raio-x **frontal** de tórax, preferencialmente em formato 224x224

## Saída
- Probabilidade de cada doença detectada com precisão acima do limite definido pelo usuário

## Exemplo de entrada e saída  :
