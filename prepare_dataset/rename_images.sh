#!/bin/bash

# Este script renomeia todas imagens imagens do whatsapp no formato YYYMMDD_hhmmss.* no diretório para uma ordem crescente iniciada pelo numero passado como argumento, da mais antiga para a mais nova.

# Verificar se o número inicial foi passado como argumento
if [ -z "$1" ]; then
  echo "Uso: $0 <número_inicial>"
  exit 1
fi

# Armazenar o número inicial fornecido
contador=$1

# Loop para percorrer os arquivos no diretório atual que seguem o formato YYYYMMDD_hhmmss.*
# Ordenados pelo nome do arquivo (o que corresponde à ordem cronológica)
for file in $(ls -1 | grep -E '^[0-9]{8}_[0-9]{6}' | sort); do
  # Extrair a extensão do arquivo
  extensao="${file##*.}"

  # Criar o novo nome com o número incrementado
  novo_nome=$(printf "%04d.%s" $contador $extensao)

  # Renomear o arquivo
  mv "$file" "$novo_nome"

  # Incrementar o contador
  contador=$((contador + 1))
done

