# Resultados da avaliação

## Desenvolvimento (data/samples, usadas para afinar)

- fotos com constelações: 15 -> certas 15 (100%), erradas 0, sem solução 0
- precisão (das soluções dadas, quantas certas): 100%
- nas certas: erro da pose mediano 3.6 px, recall da deteção (mag<=4.5) 79%, precisão das 30 mais brilhantes 76%, 198 estrelas identificadas por foto
- tempo médio 6.0 s/foto; resolvidas à 1.ª tentativa: 13/15

| foto | esperadas | encontradas | veredicto | tentativa | estrelas | score | seg | fov | erro_px | det_recall | det_prec30 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cas_01_noirlab.jpg | Cas | Cas,Per | certo | 1 | 357 | 0.67 | 9.0 | 43.7 | 6.5 | 0.78 | 0.7 |
| cas_02_bresson.jpg | Cas | Cas | certo | 1 | 146 | 0.53 | 5.9 | 25.3 | 4.0 | 1.0 | 0.8 |
| cru_01_eso_madsen.jpg | Cru | Cir,Cru,Mus | certo | 1 | 174 | 0.63 | 4.8 | 27.0 | 2.4 | 1.0 | 0.8 |
| cru_02_sardaka.jpg | Cru | Cen,Cir,Cru,Mus | certo | 1 | 73 | 0.63 | 4.6 | 37.2 | 4.4 | 0.76 | 0.83 |
| cru_03_colinj.jpg | Cru | Cen,Cru | certo | 1 | 58 | 0.9 | 0.9 | 61.1 | 3.6 | 0.61 | 0.93 |
| orion_01_niittee.jpg | Ori | CMa,Eri,Lep,Mon,Ori | certo | 1 | 550 | 0.53 | 6.5 | 66.7 | 4.3 | 0.66 | 0.9 |
| orion_02_eso_bardon.jpg | Ori | Ori | certo | 1 | 120 | 0.73 | 5.7 | 22.4 | 0.8 | 0.95 | 0.73 |
| orion_03_casa_tuntorp.jpg | Ori | Ori | certo | 1 | 95 | 0.57 | 7.1 | 22.3 | 4.5 | 0.35 | 0.6 |
| orion_04_lybrook.jpg | Ori | Eri,Gem,Lep,Ori,Tau | certo | 1 | 413 | 0.83 | 7.2 | 64.7 | 2.1 | 0.9 | 0.97 |
| sco_01_eso_antares.jpg | Sco | Sco,Sgr | certo | 2 | 228 | 0.67 | 9.8 | 30.2 | 1.8 | 0.85 | 0.83 |
| sco_02_cafuego.jpg | Sco | Ara,Sco | certo | 1 | 250 | 0.7 | 4.2 | 38.3 | 0.9 | 1.0 | 0.8 |
| uma_01_dunas.jpg | UMa | Dra,UMa,UMi | certo | 1 | 182 | 0.93 | 1.3 | 63.1 | 1.9 | 1.0 | 1.0 |
| uma_02_sandvik.jpg | UMa | Dra,UMa | certo | 1 | 220 | 0.57 | 4.3 | 45.4 | 5.1 | 0.73 | 0.5 |
| uma_03_nuvens_brastad.jpg | UMa | UMa | certo | 1 | 51 | 0.4 | 4.5 | 53.1 | 6.9 | 0.5 | 0.53 |
| uma_04_shalaev.jpg | UMa | Boo,Cep,Dra,UMa,UMi | certo | 3 | 50 | 0.57 | 13.6 | 81.8 | 3.2 | 0.7 | 0.5 |

## Teste (data/test, nunca vistas)

- fotos com constelações: 11 -> certas 8 (73%), erradas 0, sem solução 3
- controlos negativos: 1 -> corretamente rejeitados 1
- precisão (das soluções dadas, quantas certas): 100%
- nas certas: erro da pose mediano 4.3 px, recall da deteção (mag<=4.5) 68%, precisão das 30 mais brilhantes 77%, 204 estrelas identificadas por foto
- tempo médio 8.0 s/foto; resolvidas à 1.ª tentativa: 5/8

| foto | esperadas | encontradas | veredicto | tentativa | estrelas | score | seg | fov | erro_px | det_recall | det_prec30 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cas_03_4shadoww.jpg | Cas | Cas | certo | 1 | 136 | 0.93 | 1.0 | 26.2 | 2.0 | 1.0 | 1.0 |
| cru_04_rauleliasdrago.jpg | Cru | Car,Cru | certo | 2 | 281 | 0.53 | 7.9 | 52.5 | 4.3 | 0.52 | 0.67 |
| gem_01_noirlab.jpg | Gem | Gem | certo | 1 | 191 | 0.73 | 4.8 | 43.6 | 4.8 | 0.81 | 0.7 |
| leo_01_bresson.jpg | Leo | - | falhou | - | 0 | 0.0 | 4.5 |  |  |  |  |
| orion_05_schwen.jpg | Ori | Eri,Lep,Ori,Tau | certo | 2 | 317 | 0.83 | 15.1 | 40.4 | 4.3 | 0.77 | 0.87 |
| orion_06_tuntorp_floresta.jpg | Ori | Ori | certo | 1 | 81 | 0.43 | 6.4 | 52.2 | 9.2 | 0.24 | 0.47 |
| orion_07_odense.jpg | CMa,Ori | - | falhou | - | 0 | 0.0 | 16.3 |  |  |  |  |
| tau_01_bresson.jpg | Tau | Tau | certo | 1 | 132 | 0.9 | 1.3 | 27.1 | 1.3 | 0.87 | 0.87 |
| tri_01_bresson.jpg | Aql,Cyg,Lyr | Lyr | certo | 1 | 127 | 0.53 | 6.9 | 55.7 | 5.4 | 0.58 | 0.83 |
| tri_02_taipei.jpg | Aql,Cyg,Lyr | Aql,Cyg,Del,Lyr,Sge | certo | 2 | 369 | 0.77 | 10.8 | 71.6 | 1.0 | 0.61 | 0.77 |
| uma_05_blackpool.jpg | - | - | rejeitou | - | 0 | 0.0 | 3.7 |  |  |  |  |
| uma_06_postes.jpg | UMa | - | falhou | - | 0 | 0.0 | 17.7 |  |  |  |  |
