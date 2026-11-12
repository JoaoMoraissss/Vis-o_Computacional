# Resultados da avaliação

## Desenvolvimento (data/samples, usadas para afinar)

- fotos com constelações: 15 -> certas 15 (100%), erradas 0, sem solução 0
- precisão (das soluções dadas, quantas certas): 100%
- nas certas: erro da pose mediano 3.7 px, recall da deteção (mag<=4.5) 83%, precisão das 30 mais brilhantes 77%, 191 estrelas identificadas por foto
- tempo médio 6.0 s/foto; resolvidas à 1.ª tentativa: 13/15

| foto | esperadas | encontradas | veredicto | tentativa | estrelas | score | seg | fov | erro_px | det_recall | det_prec30 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cas_01_noirlab.jpg | Cas | Cas,Per | certo | 1 | 357 | 0.67 | 8.9 | 43.7 | 6.5 | 0.81 | 0.7 |
| cas_02_bresson.jpg | Cas | Cas | certo | 1 | 146 | 0.53 | 6.0 | 25.3 | 4.0 | 1.0 | 0.8 |
| cru_01_eso_madsen.jpg | Cru | Cir,Cru,Mus | certo | 1 | 174 | 0.63 | 4.8 | 27.0 | 2.4 | 1.0 | 0.8 |
| cru_02_sardaka.jpg | Cru | Cen,Cir,Cru,Mus | certo | 1 | 73 | 0.63 | 4.6 | 37.2 | 4.4 | 0.76 | 0.83 |
| cru_03_colinj.jpg | Cru | Cen,Cru | certo | 1 | 57 | 0.9 | 1.0 | 61.1 | 3.7 | 0.82 | 0.93 |
| orion_01_niittee.jpg | Ori | CMa,Eri,Mon,Ori | certo | 1 | 467 | 0.7 | 6.6 | 66.9 | 5.2 | 0.8 | 0.9 |
| orion_02_eso_bardon.jpg | Ori | Ori | certo | 1 | 120 | 0.73 | 5.8 | 22.4 | 0.8 | 0.95 | 0.73 |
| orion_03_casa_tuntorp.jpg | Ori | Ori | certo | 1 | 95 | 0.57 | 7.0 | 22.3 | 4.5 | 0.35 | 0.6 |
| orion_04_lybrook.jpg | Ori | Eri,Gem,Lep,Ori,Tau | certo | 1 | 413 | 0.83 | 7.1 | 64.7 | 2.1 | 0.9 | 0.97 |
| sco_01_eso_antares.jpg | Sco | Sco,Sgr | certo | 2 | 222 | 0.67 | 10.0 | 30.2 | 1.7 | 0.97 | 0.83 |
| sco_02_cafuego.jpg | Sco | Ara,Sco | certo | 1 | 250 | 0.7 | 4.1 | 38.3 | 0.9 | 1.0 | 0.8 |
| uma_01_dunas.jpg | UMa | Dra,UMa,UMi | certo | 1 | 182 | 0.93 | 1.3 | 63.1 | 1.9 | 1.0 | 1.0 |
| uma_02_sandvik.jpg | UMa | Dra,UMa | certo | 1 | 208 | 0.7 | 5.0 | 45.3 | 4.8 | 0.96 | 0.63 |
| uma_03_nuvens_brastad.jpg | UMa | UMa | certo | 1 | 51 | 0.4 | 4.4 | 53.1 | 6.9 | 0.5 | 0.53 |
| uma_04_shalaev.jpg | UMa | Boo,Cep,Dra,UMa,UMi | certo | 3 | 50 | 0.57 | 13.4 | 81.8 | 3.2 | 0.7 | 0.5 |

## Teste 1 (data/test; usado para diagnosticar a máscara do céu)

- fotos com constelações: 11 -> certas 10 (91%), erradas 0, sem solução 1
- controlos negativos: 1 -> corretamente rejeitados 1
- precisão (das soluções dadas, quantas certas): 100%
- nas certas: erro da pose mediano 4.2 px, recall da deteção (mag<=4.5) 72%, precisão das 30 mais brilhantes 78%, 173 estrelas identificadas por foto
- tempo médio 6.1 s/foto; resolvidas à 1.ª tentativa: 9/10

| foto | esperadas | encontradas | veredicto | tentativa | estrelas | score | seg | fov | erro_px | det_recall | det_prec30 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cas_03_4shadoww.jpg | Cas | Cas | certo | 1 | 136 | 0.93 | 1.0 | 26.2 | 2.0 | 1.0 | 1.0 |
| cru_04_rauleliasdrago.jpg | Cru | Car,Cru | certo | 2 | 254 | 0.43 | 8.0 | 52.7 | 4.4 | 0.47 | 0.6 |
| gem_01_noirlab.jpg | Gem | Gem | certo | 1 | 191 | 0.73 | 4.7 | 43.6 | 4.8 | 0.81 | 0.7 |
| leo_01_bresson.jpg | Leo | Leo | certo | 1 | 105 | 0.73 | 6.3 | 47.6 | 4.5 | 0.9 | 0.8 |
| orion_05_schwen.jpg | Ori | Eri,Lep,Ori,Tau | certo | 1 | 308 | 0.93 | 1.2 | 40.4 | 4.1 | 0.86 | 0.97 |
| orion_06_tuntorp_floresta.jpg | Ori | Ori | certo | 1 | 81 | 0.43 | 6.4 | 52.2 | 9.2 | 0.24 | 0.47 |
| orion_07_odense.jpg | CMa,Ori | Ori | certo | 1 | 30 | 0.5 | 3.7 | 65.6 | 2.3 | 0.21 | 0.63 |
| tau_01_bresson.jpg | Tau | Tau | certo | 1 | 132 | 0.9 | 1.3 | 27.1 | 1.3 | 0.87 | 0.87 |
| tri_01_bresson.jpg | Aql,Cyg,Lyr | Cyg,Lyr | certo | 1 | 201 | 0.5 | 8.3 | 55.0 | 7.1 | 0.86 | 0.73 |
| tri_02_taipei.jpg | Aql,Cyg,Lyr | Aql,Cyg,Del,Sge | certo | 1 | 294 | 0.93 | 1.3 | 71.6 | 0.8 | 1.0 | 1.0 |
| uma_05_blackpool.jpg | - | - | rejeitou | - | 0 | 0.0 | 3.7 |  |  |  |  |
| uma_06_postes.jpg | UMa | - | falhou | - | 0 | 0.0 | 26.9 |  |  |  |  |

## Teste 2 (data/test2; usado para corrigir a máscara do céu)

- fotos com constelações: 8 -> certas 7 (88%), erradas 0, sem solução 1
- controlos negativos: 1 -> corretamente rejeitados 1
- precisão (das soluções dadas, quantas certas): 100%
- nas certas: erro da pose mediano 2.1 px, recall da deteção (mag<=4.5) 79%, precisão das 30 mais brilhantes 90%, 171 estrelas identificadas por foto
- tempo médio 9.2 s/foto; resolvidas à 1.ª tentativa: 6/7

| foto | esperadas | encontradas | veredicto | tentativa | estrelas | score | seg | fov | erro_px | det_recall | det_prec30 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cru_05_halemaumau.jpg | Cru | Cen,Cru,Lup | certo | 1 | 76 | 0.83 | 3.9 | 44.3 | 2.7 | 0.77 | 0.83 |
| orion_08_betula.jpg | Ori | Eri,Mon,Ori | certo | 1 | 250 | 0.7 | 6.2 | 35.4 | 3.9 | 0.69 | 0.8 |
| orion_09_tenda.jpg | Ori | Ori | certo | 2 | 133 | 0.73 | 10.9 | 62.8 | 2.1 | 0.54 | 0.8 |
| orion_10_scion.jpg | Ori | Eri,Lep,Ori | certo | 1 | 219 | 0.8 | 10.1 | 31.1 | 4.6 | 0.97 | 0.87 |
| orion_11_alma.jpg | Ori | CMa,Lep,Mon,Ori | certo | 1 | 340 | 0.77 | 6.2 | 72.8 | 1.6 | 0.67 | 1.0 |
| sco_03_assy_turgen.jpg | Sco | - | falhou | - | 0 | 0.0 | 21.6 |  |  |  |  |
| uma_07_holma.jpg | UMa | Boo,UMa | certo | 1 | 131 | 0.97 | 1.1 | 64.8 | 0.3 | 1.0 | 1.0 |
| uma_08_blackpool_geograph.jpg | - | - | rejeitou | - | 0 | 0.0 | 22.3 |  |  |  |  |
| uma_09_yongtai.jpg | UMa | Dra,UMa,UMi | certo | 1 | 47 | 0.93 | 0.3 | 71.5 | 0.2 | 0.92 | 0.97 |

## Teste 3 (data/test3, bloqueado: nunca visto)

- fotos com constelações: 7 -> certas 2 (29%), erradas 1, sem solução 4
- precisão (das soluções dadas, quantas certas): 67%
- nas certas: erro da pose mediano 3.8 px, recall da deteção (mag<=4.5) 90%, precisão das 30 mais brilhantes 90%, 242 estrelas identificadas por foto
- tempo médio 14.7 s/foto; resolvidas à 1.ª tentativa: 1/3

| foto | esperadas | encontradas | veredicto | tentativa | estrelas | score | seg | fov | erro_px | det_recall | det_prec30 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| car_01_paranal.jpg | Car | - | falhou | - | 0 | 0.0 | 26.4 |  |  |  |  |
| cru_06_nz_jardim.jpg | Cru | Aps,Ara,Cen,Cir,Lup,Mus,Nor,Pav,Sco,TrA | errado | 2 | 667 | 0.7 | 8.4 | 47.9 | 2.6 | 0.88 | 0.77 |
| orion_12_atacama.jpg | Ori | - | falhou | - | 0 | 0.0 | 22.1 |  |  |  |  |
| orion_13_neve.jpg | Ori | Eri,Lep,Ori | certo | 1 | 227 | 0.9 | 1.8 | 34.6 | 3.4 | 0.83 | 0.87 |
| orion_14_vlt.jpg | Ori | - | falhou | - | 0 | 0.0 | 25.1 |  |  |  |  |
| orion_15_panorama.jpg | Ori | - | falhou | - | 0 | 0.0 | 12.9 |  |  |  |  |
| uma_10_brofjorden.jpg | UMa | Dra,UMa | certo | 2 | 257 | 0.37 | 6.2 | 63.3 | 4.2 | 0.96 | 0.93 |
