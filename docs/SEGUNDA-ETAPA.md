# Segunda etapa — Treinando a GNN com datasets reais

Esta etapa tira o GraphGuard dos dados sintéticos e treina a GNN com datasets
reais de fraude financeira, usando **transfer learning entre domínios**:

```
Elliptic (pré-treino)  ──►  IBM AML HI-Small (fine-tune)  ──►  IBM AML LI-Small (validação)
   ~200k transações            estrutura bancária,                proporção realista
   rótulos confiáveis          padrões de lavagem explícitos      de fraude (~0,05%)
```

O backbone GraphSAGE aprende a "propagar sinal de fraude por um grafo" no
Elliptic e é **transferido** para a estrutura bancária do IBM AML. A validação
final é feita no LI-Small porque sua taxa de lavagem (~0,05%) espelha o mercado
real — generalizar nela é o que demonstra maturidade.

---

## 1. Decisões de arquitetura (e por quê)

| Decisão | Escolha | Motivo |
|---|---|---|
| Granularidade do nó | **Conta** | Alinha com o produto: *"essa conta faz parte de uma rede suspeita?"* |
| Features heterogêneas | **Encoder por dataset + backbone compartilhado** | Elliptic tem 166 features dadas; IBM AML tem 11 features de rede engenheiradas. Pesos de message-passing só transferem se a dimensão de entrada for comum — o encoder resolve isso. |
| Backbone | **GraphSAGE** (2 camadas) | Já é o Plano A; rápido e robusto (~95% AUROC no Elliptic). |
| Compute | **1 GPU + NeighborLoader** | Mini-batch por amostragem de vizinhança; variantes Small (e Medium se quiser). |

### A ressalva honesta (diga isso aos juízes antes que eles perguntem)
No Elliptic o nó é a **transação**; no IBM AML e no produto o nó é a **conta**.
O transfer learning aqui é do **backbone de message-passing** (aprende a
estrutura de propagação de fraude), não de um mapeamento conta↔transação. O
encoder específico por dataset absorve a diferença de semântica e de dimensão.
É uma escolha defensável e documentada — não um detalhe varrido para baixo do tapete.

---

## 2. Pré-requisitos de ambiente

> Verificado em 2026-06: `torch` (2.12.1) e `torch-geometric` (2.8.0) já publicam
> wheels para **Python 3.14** — não é mais necessário um venv com 3.11/3.12.
> Use o `.venv` que já existe na raiz do projeto.

**Instalação core** (inclui `pyyaml` para rodar o treinador):

```bash
pip install -r requirements.txt    # networkx, pandas, numpy, sklearn, pyyaml, etc.
```

**PyTorch + PyG** (necessário para o treino da GNN; não necessário para a demo sintética):

```bash
# CPU:
pip install torch torch-geometric

# GPU (CUDA 12.1, exemplo):
# pip install torch --index-url https://download.pytorch.org/whl/cu121
# pip install torch-geometric
```

---

## 3. Baixar os datasets

Os arquivos NÃO são versionados (ver `.gitignore`). Coloque-os assim:

```
data/raw/
├── elliptic/
│   ├── elliptic_txs_features.csv
│   ├── elliptic_txs_classes.csv
│   └── elliptic_txs_edgelist.csv
└── ibm-aml/
    ├── HI-Small_Trans.csv
    └── LI-Small_Trans.csv
```

Via Kaggle CLI (precisa do `~/.kaggle/kaggle.json`):

```bash
pip install kaggle

kaggle datasets download -d ellipticco/elliptic-data-set -p data/raw/elliptic --unzip
kaggle datasets download -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml -p data/raw/ibm-aml --unzip
```

> O segundo dataset é grande (várias variantes HI/LI em Small/Medium/Large). Você
> só precisa de `HI-Small_Trans.csv` e `LI-Small_Trans.csv` para este pipeline.
> Os nomes dos arquivos extraídos podem variar — ajuste `trans_file` no YAML se
> necessário.

---

## 4. Rodar o treino

```bash
# pipeline completo (pré-treino -> fine-tune -> validação)
python src/train_transfer.py --config configs/transfer.yaml

# só um estágio (ex.: re-validar com um backbone já salvo)
python src/train_transfer.py --config configs/transfer.yaml --only validate_ibm_li_small

# forçar CPU
python src/train_transfer.py --config configs/transfer.yaml --device cpu
```

Os pesos do backbone são salvos em `data/checkpoints/`. A saída imprime, por
estágio, **AUROC**, **Average Precision** e **recall@FPR<5%** — a métrica que
vende o produto (recall mantendo no máximo 5% de falsos positivos).

---

## 5. Como o código está organizado

```
configs/
└── transfer.yaml          # toda a configuração do pipeline de 3 estágios

src/
├── datasets/
│   ├── common.py          # GraphSample — contrato comum (x, edge_index, y, masks) + to_pyg()
│   ├── account_features.py# as 11 features de rede, vetorizadas (groupby) p/ escala
│   ├── elliptic.py        # Elliptic  -> GraphSample (nó = transação)
│   ├── ibm_aml.py         # IBM AML   -> GraphSample (nó = conta)
│   └── __init__.py        # load_dataset(cfg) despacha pelo 'kind'
├── gnn/
│   ├── model.py           # encoder por dataset + backbone GraphSAGE + cabeça (save/load do backbone)
│   ├── metrics.py         # AUROC, AP, recall@FPR
│   └── train.py           # train_stage() + run_pipeline() (orquestra o transfer)
└── train_transfer.py      # CLI: lê o YAML e roda o pipeline
```

Todo loader entrega o mesmo `GraphSample`, então adicionar um novo dataset é
escrever um loader e uma entrada no YAML — nada no modelo muda.

---

## 6. Configurar / ajustar

Tudo vive em `configs/transfer.yaml`:

- **Trocar a ordem ou os datasets do transfer:** edite a lista `pipeline:`.
- **CPU em vez de GPU:** `device: cpu` e `train.neighbor_sampling.enabled: false`
  (full-batch; o Elliptic cabe em memória).
- **Variante Medium do IBM AML:** troque `trans_file` por `HI-Medium_Trans.csv`
  (precisa de GPU; ative `neighbor_sampling`).
- **Betweenness por conta:** `compute_betweenness: true` (caro — só em grafos
  pequenos; em grafos grandes a coluna fica zerada por padrão).
- **Mais capacidade do modelo:** `model.hidden_dim` / `model.num_layers`.

---

## 7. Próximos passos sugeridos (se sobrar tempo)

- **BAF (Bank Account Fraud, NeurIPS 2022):** ~1M de contas com fraude de
  abertura (1,1%). Não é grafo, mas serve para *warm-start* dos atributos de nó
  — pré-treinar um MLP nas features tabulares de conta e inicializar o encoder
  do IBM AML com esses pesos.
- **GAT / EvolveGCN:** trocar o backbone para atenção (explicabilidade no pitch)
  ou temporal (~97%+, usa os time steps do Elliptic). O `model.arch` no YAML já
  está reservado para isso.
- **Line graph do IBM AML:** converter a IBM AML em transação-como-nó para casar
  100% com a semântica do Elliptic (transfer learning conceitualmente limpo).