# Aplicação para detecção de partidas de xadrez usando visão computacional

Esse repositório contém todo o software do meu [trabalho de TCC](https://wiki.sj.ifsc.edu.br/index.php/Aplicativo_de_Reconhecimento_de_Lances_de_Xadrez_com_Vis%C3%A3o_Computacional).

> :warning: Este projeto e sua documentação foram criados em um sistema Ubuntu 24.04 e não foram testados em outras distribuições Linux ou outros sistemas operacionais.

## Como funciona?

A aplicação utiliza uma sequência de técnicas de processamento de imagens para isolar a área do tabuleiro de xadrez e mapear todas as suas casas. Em seguida, utiliza um modelo YOLOv11 treinado e hospedado na plataforma Roboflow para detecção e classificação das peças de xadrez presentes no tabuleiro e por fim usa um algoritmo simples baseado nas coordenadas da imagem para determinar em qual casa cada peça está. Este processo é repetido a cada imagem e as posições das peças, bem como os lances jogados, são validados pela biblioteca python-chess.

Ao fim da partida, é apresentado um relatório da partida na linha de comando como o abaixo:

```
======== GAME FINISHED! MOVES: =========
1. e4 e5
2. Nf3 d6
3. d4 Bg4
4. dxe5 Bxf3
5. Qxf3 dxe5
6. Bc4 Nf6
7. Qb3 Qe7
8. Nc3 c6
9. Bg5 b5
10. Nxb5 cxb5
11. Bxb5+ Nbd7
12. O-O-O Rd8
13. Rxd7 Rxd7
14. Rd1 Qe6
15. Bxd7+ Nxd7
16. Qb8+ Nxb8
17. Rd8# 
========================================
```

## Requisitos

* Python 3.12.3
* Instalar o python3-venv:
```
sudo apt install python3-venv
```
* Criar um ambiente virtual python na raiz do repositório e ativá-lo:
```
python3 -m venv .venv
source .venv/bin/activate
```
* Instalar dependências do requirements.txt
```
pip install -r requirements.txt
```

## Como rodar?

Tendo o ambiente virtual python devidamente configurado e ativado como descrito acima, crie um arquivo `.env` na raiz do repositório contendo as seguintes variáveis de ambiente:
```
ROBOFLOW_API_KEY=
ROBOFLOW_PROJECT_ID=
ROBOFLOW_MODEL_VERSION=
```

Embora o projeto no Roboflow seja público, para conseguir testar o modelo criado no decorrer deste trabalho, será necessário ter a API_KEY do projeto no Roboflow, bem como o ID do projeto e a versão do modelo desejado. Para obter essas informações, pode-se entrar em contato com o autor. Estes dados serão passados apenas para os orientadores do trabalho e para a banca do TCC.

Como alternativa, pode-se criar um fork do [projeto](https://app.roboflow.com/chess-recognition-tcc-ifsc), que encontra-se público no Roboflow. Em seguida, criar o dataset e treinar o próprio modelo YOLOv11 com o conjunto de imagens construído pelo autor e as mesmas configurações que constam abaixo:

```
Preprocessing
Auto-Orient: Applied
Resize: Stretch to 640x640

Augmentations
Outputs per training example: 7
Flip: Horizontal, Vertical
Saturation: Between -20% and +20%
Brightness: Between -20% and +20%
```

Por fim, para rodar o projeto com as imagens de teste predefinidas disponíveis no repositório em `test_images`, entre no diretório `application` e rode o programa:
```
cd application
python3 main.py
```

Caso deseje ver o processamento das imagens e a detecção de objetos passo-a-passo, entre no arquivo `application/config.py` e altere a variável `SHOW_IMAGES` para o valor `True` antes de rodar a aplicação. Para cada imagem, pressione a tecla ENTER para avançar.