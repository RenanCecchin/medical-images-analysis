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

# Modelo disponível para uso

Exemplo de entrada e saída  :

![exemplo de entrada](https://imgs.search.brave.com/6sR23hyjZGwG5oW0OFo742gr_62WGN_HecZZ2f0yJ_A/rs:fit:860:0:0/g:ce/aHR0cHM6Ly9zdGFu/Zm9yZG1sZ3JvdXAu/Z2l0aHViLmlvL3By/b2plY3RzL2NoZXhu/ZXQvaW1nL2NoZXgt/bWFpbi5zdmc.svg)
