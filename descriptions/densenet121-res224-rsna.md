##  Descrição do modelo
Este é um modelo pré-treinado disponível por meio da biblioteca [TorchXRayVision](https://github.com/mlmed/torchxrayvision). Ele foi treinado com o dataset [RSNA Pneumonia Detection Challenge](https://pubs.rsna.org/doi/full/10.1148/ryai.2019180041) e possui a maior precisão em detecção de opacidade pulmonar e pneumonia.

## Doenças detectadas
- Opacidade pulmonar
- Pneumonia

## Entrada
- Imagem de raio-x **frontal** de tórax, preferencialmente em formato 224x224

## Saída
- Probabilidade de cada doença detectada com precisão acima do limite definido pelo usuário

## Exemplo de entrada e saída  :
