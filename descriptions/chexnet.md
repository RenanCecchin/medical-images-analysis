##  Descrição do modelo
Este é um modelo pré-treinado disponível [aqui](https://github.com/arnoweng/CheXNet/). Ele é uma reimplementação de um [trabalho de mesmo nome](https://stanfordmlgroup.github.io/projects/chexnet/) feito por pesquisadores da Universidade de Stanford. O modelo é capaz de detectar 14 doenças em imagens de raio-x de tórax.

## Doenças detectadas
- Atelectasia
- Cardiomegalia
- Derrame pleural (Efusão)
- Infiltração
- Massa
- Nódulo
- Pneumonia
- Pneumotórax
- Consolidação
- Edema
- Enfisema
- Fibrose
- Espessamento pleural
- Hérnia

## Entrada
- Imagem de raio-x **frontal** de tórax

## Saída
- Um mapa de calor com as regiões de interesse para as doenças detectadas com confiança superior às restrições do usuário

## Exemplo de entrada e saída  :
