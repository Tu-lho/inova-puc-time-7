<div align="center">

# 🏆 GraphGuard Pix

### Detecção de redes de fraude no Pix com Graph Neural Networks

[![1º Lugar — Hackathon Inova AI](https://img.shields.io/badge/🥇%201º%20Lugar-Hackathon%20Inova%20AI-2DD4BF?style=for-the-badge)](.)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](.)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-GradientBoosting-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](.)
[![License: MIT](https://img.shields.io/badge/License-MIT-94A3B8?style=for-the-badge)](LICENSE)

*Os sistemas antifraude atuais perguntam "essa transação é suspeita?".*
*O GraphGuard pergunta **"essa conta faz parte de uma rede suspeita?"***

</div>

---

## 🎯 O pitch em um parágrafo

Fraudes no Pix custaram **R$ 4,9 bilhões em 2024** — e menos de 9% foi recuperado. O problema não está nas transações: está nas **redes invisíveis** que as conectam. Uma transferência de R$ 900 parece normal isoladamente. Mas se a conta de destino recebeu R$ 4.800 dois minutos antes e distribuiu em seis valores menores para contas criadas na última semana, isso é smurfing — e nenhum sistema que analisa transação por transação consegue ver. O GraphGuard modela cada transferência como aresta de um grafo, aplica uma Graph Neural Network que aprende o comportamento de vizinhança de cada conta, e entrega um score de risco em menos de 100ms via API — detectando contas mula novas **antes do primeiro golpe**, quando todos os filtros tradicionais ainda marcam zero.

---

## 🥇 Hackathon Inova AI — 1º Lugar

> Projeto desenvolvido e apresentado no **Hackathon Inova AI**. A solução foi classificada em **primeiro lugar** pela originalidade da abordagem de grafo, pelo MVP funcional entregue em 3 horas e pelo potencial de mercado diante do novo arcabouço regulatório do Banco Central (Resoluções BCB 501 e 506/2025).

### 👥 Equipe

| Integrante |  
|-----------|-------|  
| **João Victor P. Bicalho**  
| **Pedro Huck** |  
| **Tulio Gonçalves Vieira**  

---

## 💡 A ideia em 30 segundos

O GraphGuard modela todas as transações Pix como um **grafo** — contas são nós, transferências são arestas — e detecta três padrões de fraude que só aparecem na estrutura da rede:

| Padrão | Assinatura no grafo |
|--------|---------------------|
| **Smurfing** | Hub com alto grau de saída em janela curta de tempo |
| **Fan-out / layering** | Cadeia profunda com repasse rápido entre camadas |
| **Contas mula** | Recebem de remetentes desconectados e repassam em lote |

Uma conta mula criada ontem, sem histórico e sem denúncia, **passa em todos os filtros existentes** — e é detectada pelo GraphGuard porque a rede dela já é suspeita.

---

## 🚀 Quickstart

```bash
git clone https://github.com/<seu-usuario>/graphguard-pix.git
cd graphguard-pix
pip install -r requirements.txt
python src/demo.py
```

Saída esperada:

```
================================================================
  5. O número que vende o produto
================================================================
  Filtro por transação (baseline) pega : 35% das contas mula
  GraphGuard (rede)               pega : 100% das contas mula
  Ganho de cobertura                   : +65% pontos
```

A demo gera dois artefatos em `data/`:
- `graph.png` — imagem estática para slides
- `graph.html` — grafo interativo, abra no navegador e arraste os nós

---

## 📊 Resultados

<div align="center">

| Métrica | Valor |
|---------|-------|
| AUC-ROC | **0,99** |
| Average Precision | **0,93** |
| Cobertura de contas mula (baseline) | 35% |
| Cobertura de contas mula (GraphGuard) | **100%** |
| Latência de resposta (API) | < 100ms |

</div>

---

## 🏗️ Arquitetura

```
[data_generator]  →  transações sintéticas + fraude injetada (com rótulos)
        │
        ▼
[graph_builder]   →  grafo networkx + 11 features de rede por conta
        │
        ▼
[model]           →  Plano B: features + GradientBoosting   (sempre roda)
        │             Plano A: GraphSAGE (PyG)               (opcional)
        │             → score de risco por conta (0..1)
        ▼
[visualize]       →  PNG estático + HTML interativo autocontido
        │
        ▼
[demo]            →  orquestra tudo e imprime a métrica do pitch
```

### Os dois planos — estratégia de hackathon

O erro clássico é apostar tudo numa GNN que não converge a tempo. O projeto tem dois caminhos:

- **Plano B (padrão, sempre roda):** features de rede + `GradientBoosting`. Não é uma GNN, mas demonstra exatamente a mesma tese e treina em segundos.
- **Plano A (opcional):** GraphSAGE de 2 camadas via PyTorch Geometric. Rode com `python src/demo.py --plan a`. Se o PyG não estiver instalado, o sistema cai automaticamente para o Plano B.

---

## 🧠 As 11 features de rede

Cada conta é descrita por features que capturam comportamento *de vizinhança* — não a transação isolada:

| Feature | O que captura |
|---------|---------------|
| `in_degree` / `out_degree` | número de contrapartes únicas |
| `total_in` / `total_out` | volume financeiro total |
| `flow_ratio` | `out/in` — contas mula ficam perto de 1 |
| `n_senders` / `n_receivers` | dispersão das contrapartes |
| `fanout` | razão de distribuição (smurfing tem fanout alto) |
| `avg_forward_delay` | velocidade entre receber e repassar |
| `betweenness` | centralidade de intermediação (mulas são pontes) |
| `amount_in_max` | maior valor recebido em uma única transferência |

> O dataset inclui **comerciantes legítimos como confundidores** — contas com alto grau de entrada e repasse rápido a fornecedores. Isso força o modelo a aprender a *combinação* de sinais, não um único atalho, resultando num AUC realista de 0,99 em vez de um suspeito 1,00.

---

## 📁 Estrutura do repositório

```
graphguard-pix/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                  # artefatos gerados — ignorado pelo git
│   └── .gitkeep
└── src/
    ├── data_generator.py  # gera transações sintéticas com fraude injetada
    ├── graph_builder.py   # constrói o grafo e extrai as 11 features
    ├── model.py           # Plano B (sklearn) + Plano A (GraphSAGE)
    ├── visualize.py       # PNG estático + HTML autocontido (sem lib/ local)
    └── demo.py            # runner ponta-a-ponta — entrypoint principal
```

Cada módulo roda isolado:

```bash
python src/data_generator.py   # inspeciona os dados gerados
python src/graph_builder.py    # compara features de fraude vs legítimas
python src/model.py            # treina e exibe AUC + importância das features
python src/visualize.py        # gera as visualizações em data/
```

---

## 🌐 Visualização interativa

O `graph.html` gerado é **100% autocontido** — usa o vis-network via CDN e não depende de nenhuma pasta `lib/` local. Pode ser aberto de três formas:

```bash
# 1. Duplo-clique no arquivo (Windows / Mac)
# No Linux:
xdg-open data/graph.html

# 2. Servidor local (útil para apresentação em rede)
python3 -m http.server 8080
# acesse http://localhost:8080/data/graph.html

# 3. GitHub Pages — funciona direto, sem configuração extra
```

---

## 🏦 Do MVP ao produto

Em produção, os bancos enviam transações pseudonimizadas via API (IDs em hash SHA-256, sem CPF em claro) e recebem o score de risco de volta em menos de 100ms. Prevenção a fraude se enquadra no **legítimo interesse da LGPD** — sem necessidade de consentimento individual.

| Fase | Marco | Prazo |
|------|-------|-------|
| POC | demo com dados sintéticos | ✅ hoje |
| Piloto | banco envia 90 dias anonimizados; relatório de clusters | 30–60 dias |
| Contrato | API/SaaS com SLA, DPA e registro Unicad (IN BCB 590/25) | Q3 |
| Federação | grafo cross-banco com federated learning e network effect | escala |

---

## ⚖️ Contexto regulatório

- **Resolução BCB 501/2025** — obriga rejeitar transações para contas com "fundada suspeita de fraude", mas delega o critério a cada instituição. O GraphGuard é esse critério.
- **Resolução BCB 506/2025** — multas diárias de R$ 10 mil por descumprimento. Antifraude deixou de ser custo operacional e virou risco regulatório existencial.
- **MED 2.0 (fev/2026)** — mecanismo de devolução recuperou menos de 9% dos valores em 2025. Prevenir é 11× mais eficaz que tentar recuperar.

---

## 📜 Licença

MIT — use, modifique e distribua livremente.

---

<div align="center">

Feito com 🧠 e ☕ pela **Equipe GraphGuard**
**Hackathon Inova AI · 2026 · 🥇 1º Lugar**

*João Victor P. Bicalho · Pedro Huck · Tulio Gonçalves Vieira*

</div>
