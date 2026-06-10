# GraphGuard Pix
## Hackathon Inova AI
## Equipe  
- João Victor Pereira Bicalho
- Pedro Huck Henrique
- Tulio Gonçalves Vieira  

**Detecção de redes de fraude no Pix com Graph Neural Networks.**

Os sistemas antifraude atuais perguntam *"essa transação é suspeita?"*.
O GraphGuard pergunta *"essa conta faz parte de uma rede suspeita?"* — e é por
isso que ele pega contas mula novas que passariam por qualquer filtro
transacional.

> Projeto desenvolvido para hackathon de IA antifraude. MVP funcional com dados
> sintéticos, pronto para demonstração ao vivo.

---

## A ideia em 30 segundos

Uma transferência de R$900 parece normal isoladamente. Mas se a conta de destino
recebeu R$4.800 dois minutos antes e os distribuiu em seis valores menores para
contas criadas na última semana, isso é **smurfing** — e nenhum sistema que
analisa transações uma a uma consegue ver esse padrão.

O GraphGuard modela todas as transações como um **grafo** (contas = nós,
transferências = arestas) e usa as features de rede de cada conta para detectar
três padrões de fraude:

| Padrão | Assinatura no grafo |
|--------|---------------------|
| **Smurfing** | Hub com alto grau de saída em janela curta de tempo |
| **Fan-out / layering** | Cadeia profunda com repasse rápido entre camadas |
| **Contas mula** | Recebem de remetentes desconectados e repassam em lote |

---

## Quickstart

```bash
git clone <seu-repo>
cd graphguard-pix
pip install -r requirements.txt
python src/demo.py
```

Saída esperada (resumida):

```
5. O número que vende o produto
  Filtro por transação (baseline) pega : 35% das contas mula
  GraphGuard (rede)               pega : 100% das contas mula
  Ganho de cobertura                   : +65% pontos
```

A demo gera `data/graph.png` (estático, para slides) e `data/graph.html`
(interativo, arraste os nós — abra no navegador para a apresentação ao vivo).

---

## Arquitetura

```
[data_generator]  ->  transações sintéticas + fraude injetada (com rótulos)
        |
        v
[graph_builder]   ->  grafo networkx + 11 features de rede por conta
        |
        v
[model]           ->  Plano B: features + GradientBoosting   (sempre roda)
        |             Plano A: GraphSAGE (PyG)                (opcional)
        |             -> score de risco por conta (0..1)
        v
[visualize]       ->  PNG (matplotlib) + HTML interativo (pyvis)
        |
        v
[demo]            ->  orquestra tudo e imprime a métrica do pitch
```

### Os dois planos (estratégia de hackathon)

O erro clássico de hackathon é apostar tudo numa GNN que não converge a tempo.
Por isso o projeto tem dois caminhos:

- **Plano B (padrão):** features de rede + `GradientBoosting` do scikit-learn.
  Não é uma GNN, mas demonstra exatamente a mesma tese e treina em segundos.
  **Sempre funciona.**
- **Plano A (opcional):** GraphSAGE de 2 camadas em PyTorch Geometric. Rode com
  `python src/demo.py --plan a` (requer `pip install torch torch-geometric`). Se
  o PyG não estiver instalado, o sistema cai para o Plano B automaticamente.

Comece pelo B para garantir a demo; tente o A se sobrar tempo.

---

## As features de rede (o coração do projeto)

Cada conta é descrita por 11 features que capturam o comportamento *de rede* —
não a transação isolada:

| Feature | O que captura |
|---------|---------------|
| `in_degree` / `out_degree` | número de contrapartes |
| `total_in` / `total_out` | volume financeiro |
| `flow_ratio` | `out/in` — mulas ficam perto de 1 (tudo que entra, sai) |
| `n_senders` / `n_receivers` | dispersão das contrapartes |
| `fanout` | razão de distribuição (smurfing tem fanout alto) |
| `avg_forward_delay` | velocidade entre receber e repassar |
| `betweenness` | centralidade de intermediação (mulas são pontes) |
| `amount_in_max` | maior valor recebido (pega o hub) |

O dataset inclui **comerciantes legítimos** como confundidores (alto grau de
entrada + repasse rápido a fornecedores). Sem eles, qualquer conta com repasse
rápido pareceria fraude e o problema seria trivial. Com eles, o modelo é
obrigado a aprender a *combinação* de sinais — como na vida real.

---

## Estrutura dos arquivos

```
graphguard-pix/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                 # artefatos gerados (png, html) — ignorado pelo git
└── src/
    ├── data_generator.py # transações sintéticas + fraude injetada
    ├── graph_builder.py  # grafo + features de rede
    ├── model.py          # Plano B (sklearn) + Plano A (GraphSAGE)
    ├── visualize.py      # PNG estático + HTML interativo
    └── demo.py           # runner ponta-a-ponta (entrypoint da demo)
```

Cada módulo roda isolado para teste:

```bash
python src/data_generator.py   # inspeciona os dados gerados
python src/graph_builder.py    # compara features de fraude vs legítimas
python src/model.py            # treina e mostra AUC + importância das features
python src/visualize.py        # gera as visualizações
```

---

## Do MVP ao produto

Este MVP usa dados sintéticos. Em produção, o dado do banco é adquirido pelo
modelo de menor atrito possível — **API com pseudonimização**: o banco envia IDs
de conta com hash (sem CPF) + metadados (valor, timestamp) e recebe o score de
volta. Prevenção a fraude se enquadra no legítimo interesse da LGPD e não exige
consentimento individual. Modelos mais avançados (on-premise, federated learning
cross-banco) entram conforme a relação amadurece.

| Fase | Marco |
|------|-------|
| POC (hackathon) | demo com dados sintéticos |
| Piloto (30–60 dias) | banco envia amostra anonimizada; relatório de clusters |
| Contrato | API/SaaS com SLA (<100ms p99), DPA, registro Unicad |
| Federação | grafo cross-banco com network effect |

---

## Roteiro de pitch (3 min)

1. **Problema:** "Uma transação de R$900 parece normal. E se faz parte de uma
   rede de lavagem com 12 contas? Nenhum banco vê isso hoje."
2. **Insight:** "Fraude não vive numa transação. Vive numa rede."
3. **Demo:** abrir `data/graph.html`, mostrar que os nós de fraude são
   invisíveis isoladamente, rodar `demo.py`, ver os clusters acenderem.
4. **Número:** "Filtro por transação pega 35% das mulas. Nós pegamos 100%."
5. **Negócio:** "A BCB 501 obriga rejeitar contas com fundada suspeita, mas não
   diz como. Nós somos o como — e ficamos mais fortes a cada banco que entra."

---

## Licença

MIT — use livremente.
