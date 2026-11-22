# Identificação de constelações em fotografias do céu

Projeto de Visão Computacional 2026/27 (FCUP), docentes Miguel Coimbra e Hélder Oliveira.

## Autores
- João Morais, up202307077
- Rodrigo Mendes, up202308366

## Descrição

A partir de uma fotografia do céu noturno, sem mais nenhuma informação (data, local,
orientação ou distância focal), o programa encontra as estrelas, descobre que parte do céu
está na imagem e desenha por cima as linhas e os nomes das constelações.

O método usa só técnicas clássicas de visão computacional, em três fases:

1. **Deteção de estrelas** (`preprocessing.py`, `segmentation.py`, `detection.py`):
   top-hat morfológico em várias escalas para tirar o fundo, máscara céu/paisagem para
   ignorar árvores, montanhas e luzes, e um filtro de forma que só aceita manchas pequenas
   e redondas.
2. **Correspondência com o catálogo** (`matching.py`, `matching_sphere.py`): triângulos
   formados pelas estrelas mais brilhantes, descritos por razões entre lados. Os lados
   são medidos em ângulos na esfera celeste, para não serem deformados pelas objetivas
   grande angulares; como o campo de visão não se conhece, experimentam-se vários.
3. **Pose da câmara** (`matching_sphere.py`, `projection.py`): com três pares
   estrela–catálogo, a rotação sai pelo algoritmo de Kabsch; a solução é aceite se
   explicar estrelas suficientes da imagem. O resultado é desenhado por `overlay.py`.

O catálogo é o HYG v4.1 (8 920 estrelas até magnitude 6,5) e as linhas das constelações são
as do Stellarium (88 constelações IAU).

## Instalação

Testado com Python 3.14.

```sh
python -m venv .venv
source .venv/bin/activate        # fish: source .venv/bin/activate.fish
pip install -r requirements.txt
```

O catálogo já filtrado está em `data/catalog/`. Os índices de triângulos
(`triangles*.npz`) são calculados automaticamente na primeira execução (alguns segundos).
Para voltar a gerar o catálogo a partir do HYG original (~34 MB): `python src/catalog.py`.

## Como correr

Todos os comandos são corridos a partir da raiz do projeto.

```sh
# identificar constelações numa ou mais fotos (resultado em results/demo/)
python src/main.py data/samples/orion_01_niittee.jpg
python src/main.py --samples                  # todas as fotos de data/samples/
python src/main.py foto.jpg --out pasta/      # outra pasta de saída
python src/main.py foto.jpg --mask 2          # versão 2 da máscara do céu

# avaliar com as etiquetas de referência (tabelas em results/metrics/)
python src/evaluate.py                        # conjunto de teste bloqueado (data/test3/)
python src/evaluate.py --set samples          # test, test2, test3, samples ou all
python src/evaluate.py --synthetic            # também as imagens sintéticas

# gerar as 20 imagens sintéticas (data/synthetic/) e as figuras da fase 1
python src/fase1_demo.py
```

As imagens sintéticas não estão no repositório: é preciso correr `fase1_demo.py` antes
de usar `--synthetic`.

## Estrutura

```
src/
  main.py              pipeline completo e linha de comandos
  evaluate.py          avaliação com as etiquetas (labels.csv) de cada conjunto
  catalog.py           catálogo de estrelas e linhas das constelações
  preprocessing.py     top-hat e estimativa do ruído
  segmentation.py      máscara céu / paisagem (versões 1 e 2)
  detection.py         deteção de estrelas e filtro de forma
  matching.py          triângulos no plano (versão 1) e avaliação nas sintéticas
  matching_sphere.py   triângulos na esfera e pose da câmara (versão usada)
  projection.py        modelo de câmara e projeção do catálogo
  overlay.py           desenho das constelações por cima da foto
  synthetic.py         gerador de céus sintéticos com verdade de referência
  fase1_demo.py        gera as imagens sintéticas e as figuras da fase 1
data/
  catalog/             estrelas (mag <= 6,5) e linhas das constelações
  samples/             15 fotos usadas para afinar o método
  test/, test2/, test3/  conjuntos de teste (ver abaixo)
  proprias/            fotos tiradas por nós com o telemóvel
results/
  metrics/             tabelas da avaliação (máscara v1; mascara_v2/ para a v2)
  fotos_proprias/      resultados nas nossas fotos
report/                relatório em LaTeX (relatorio.tex) e figuras
presentation/          apresentação
```

## Dados

As fotografias de `data/samples/` e dos conjuntos de teste vêm do Wikimedia Commons; autor,
licença e ligação de cada uma estão no `FONTES.md` de cada pasta. Cada conjunto tem
um `labels.csv` com as constelações que a foto mostra; uma linha sem constelações é um
controlo negativo (o resultado certo é "sem solução").

Separámos as fotos usadas para afinar o método das fotos de teste. `data/test/` e
`data/test2/` deixaram de ser limpos porque foram usados para diagnosticar e corrigir a
máscara do céu; `data/test3/` foi juntado depois disso e é o teste atual. A regra é não
mudar parâmetros depois de ver os resultados do teste.

## Resultados (máscara v1)

| Conjunto | Fotos | Certas | Erradas | Sem solução | Controlos negativos |
|---|---|---|---|---|---|
| `samples` (afinação) | 15 | 15 | 0 | 0 | — |
| `test` | 11 | 8 | 0 | 3 | 1/1 rejeitado |
| `test2` | 8 | 7 | 0 | 1 | 1/1 rejeitado |
| `test3` (bloqueado) | 7 | 3 | 1 | 3 | — |

Nas 20 imagens sintéticas todas as constelações são encontradas, com 98,6 % das estrelas
bem identificadas. Nas fotos certas o erro mediano da pose é de 2 a 4 píxeis, e cada foto
demora entre 5 e 15 s. Tabelas completas em `results/metrics/resultados.md`.

Nas nossas fotos (`data/proprias/`), Órion foi identificado nas duas fotos nítidas; a foto
tremida só foi resolvida depois de reduzida a 25 % (explicação em
`data/proprias/explicacao_como_nao_se_engana.txt`).

## Limitações

- Quase todas as falhas vêm da máscara do céu ou de fotos fora do âmbito: luzes de cidade e
  árvores confundidas com estrelas, panoramas com mais de 100° de campo, exposições longas
  sem seguimento (estrelas em riscos) e mosaicos editados.
- O modelo de lente (radial, um parâmetro) pode ajustar demais em fotos editadas.
- Os conjuntos de teste são pequenos, por isso as percentagens têm muita incerteza.

O relatório (`report/relatorio.tex`) descreve o método e a avaliação em pormenor.
