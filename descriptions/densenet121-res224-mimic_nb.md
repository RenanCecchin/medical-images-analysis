##  Descrição do modelo
Este é um modelo pré-treinado disponível por meio da biblioteca [TorchXRayVision](https://github.com/mlmed/torchxrayvision). Ele foi treinado com o dataset [MIMIC CXR do MIT](https://github.com/MIT-LCP/mimic-cxr) e é capaz de detectar 11 doenças de tórax.

## Doenças detectadas
- Atelectasia
- Cardiomegalia
- Cardiomediastino aumentado
- Consolidação
- Derrame pleural (Efusão)
- Edema
- Fratura
- Lesão pulmonar
- Opacidade pulmonar
- Pneumonia
- Pneumotórax

## Entrada
- Imagem de raio-x **frontal** de tórax, preferencialmente em formato 224x224

## Saída
- Probabilidade de cada doença detectada com precisão acima do limite definido pelo usuário

## Exemplo de entrada e saída  :
