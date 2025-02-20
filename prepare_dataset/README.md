## Renomeador de imagens

Este é um script shell simples cuja única função é renomear as imagens com o nome padrão criado pelo telefone, no formato YYYMMDD_hhmmss.jpg para um formato 0001.jpg, por exemplo. 

O único objetivo é ter as imagens devidamente organizadas em ordem cronológica da mais antiga para a mais atual, facilitando a sua manipulação e criação do conjunto de imagens.

O script renomeia todas as imagens no diretório em que se encontra.

### Como rodar:
```
./rename_images.sh 1
```

O parâmetro passado por linha de comando será a numeração inicial dada para a imagem mais antiga.