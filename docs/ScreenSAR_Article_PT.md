pa# ScreenSAR: Plataforma Automatizada para Curadoria de Dados QSAR, Modelagem e Triagem Virtual

## Resumo

A modelagem de Relações Quantitativas Estrutura-Atividade (QSAR) é um dos pilares da descoberta moderna de fármacos, permitindo a previsão da atividade biológica a partir da estrutura química. No entanto, a confiabilidade dos modelos preditivos QSAR é fortemente dependente da qualidade dos dados de entrada e da robustez do fluxo de trabalho de modelagem. Este artigo apresenta o **ScreenSAR**, uma plataforma abrangente e automatizada, desenvolvida para simplificar e unificar a curadoria de dados químicos, o desenvolvimento de modelos de aprendizado de máquina (Machine Learning) e a **triagem virtual de alto rendimento (High-Throughput Virtual Screening)**. A ferramenta oferece uma arquitetura modular que engloba a limpeza padronizada de dados, engenharia de características (descritores) e treinamento de múltiplos modelos. Uma inovação fundamental é seu **motor de processamento em lotes (chunk-based)**, que permite a triagem confiável de **milhões de compostos** utilizando hardware comercial padrão. Demonstramos a utilidade do ScreenSAR para melhorar a reprodutibilidade, eficiência e acessibilidade aos fluxos de trabalho da pesquisa farmacêutica e quimioinformática.

---

## 1. Introdução: Descoberta de Fármacos Guiada por Dados

A indústria farmacêutica está passando por uma mudança de paradigma impulsionada pela Inteligência Artificial (IA) e pelo Aprendizado de Máquina (ML). A integração de "Big Data" oriunda de triagens experimentais (HTS) e repositórios públicos como ChEMBL e PubChem possibilitou o desenvolvimento de modelos preditivos que podem acelerar significativamente a fase de otimização de protótipos (Lead Optimization).

### 1.1. O Paradoxo da Qualidade dos Dados
Apesar desses avanços, o princípio "Lixo que entra, lixo que sai" (Garbage In, Garbage Out) continua sendo o principal gargalo. Estudos estimam que até **80% do tempo de um cientista de dados** é gasto na limpeza dos dados, em vez de na modelagem em si. Representações químicas inconsistentes, ambiguidades estereoquímicas e ruídos experimentais em grandes conjuntos de dados públicos muitas vezes levam à criação de modelos QSAR com baixo poder preditivo prospectivo.

### 1.2. O ScreenSAR Frente ao Estado da Arte
O **ScreenSAR** avança o estado da arte ao democratizar o acesso à curadoria robusta. A plataforma combina os rigorosos e padronizados algoritmos de bibliotecas baseadas em código (como o RDKit) com a acessibilidade de uma interface web moderna, preenchendo efetivamente a lacuna entre a mineração bruta de dados e a entrega de modelos QSAR acionáveis de alta qualidade, sem a necessidade de habilidades refinadas em programação por parte do usuário final.

## 2. Materiais e Métodos

### 2.1. Componentes Operacionais Centrais
O ScreenSAR está estruturado em torno de dois pilares funcionais interligados, desenvolvidos para separar o desenvolvimento dos modelos da fase de sua aplicação e predição:

1.  **Curadoria e Fabricação de Modelos**: Este componente ingere dados brutos de atividade biológica (por exemplo, originários do ChEMBL ou ensaios *in-house*) e executa o *pipeline* de ponta-a-ponta: curadoria de dados, cálculo de descritores e emprego de aprendizado de máquina. Ele avalia automaticamente múltiplos algoritmos, seleciona estatisticamente o melhor modelo e o exporta como um artefato consolidado em formato `.pkl` (Python Pickle).
2.  **Triagem Virtual e Predição Prospectiva**: Um componente independente que permite o carregamento do modelo pré-treinado para varrer quimiotecas desconhecidas. Ele foi projetado para **produção de altíssimo rendimento**, removendo o custo computacional do treinamento do ambiente de produção e concentrando-se puramente na geração instantânea de descritores e predição massiva, lidando com bibliotecas de até **milhões de compostos**.

### 2.2. Arquitetura do Sistema
A aplicação foi construída em **Python** utilizando o framework **Streamlit**, assegurando uma interface de usuário responsiva e hospedável via web. A base do software obedece a um padrão modular:

*   **`core/`**: Regras de negócio científico (`curation.py`, `modeling.py`, `applicability_domain.py`, `db.py`).
*   **`ui/`**: Componentes da interface visual do usuário (`dashboard.py`, `prediction.py`, `auth.py`, `sidebar.py`).
*   **`utils/`**: Ferramentas auxiliares para geração de relatórios e traduções globais.

### 2.3. Pipeline de Curadoria de Dados (`CuradoriaQSAR`)
Automatiza passos cruciais para assegurar a integridade estrutural e biológica:
1.  **Normalização Química**: Usa o RDKit para processar SMILES, convertendo-os em suas formas canônicas, removendo sais e solventes, retendo somente o fragmento orgânico principal e neutralizando isômeros em sua representação 2D.
2.  **Padronização de Atividades (Unidades)**: Conversão automática e inteligente de IC50/EC50 brutos (µM, mM, M) para a linha de base em **nM**. Permite ainda escalar e linearizar para o formato **pIC50**.
3.  **Gestão de Duplicatas e *Activity Cliffs***: Localiza duplicatas biológicas baseadas no *SMILES*. Valores com variação inferior a 1 unidade logarítmica (10x) são combinados através da média geométrica. Discrepâncias maiores acionam a remoção conservadora do composto.
4.  **Binarização**: Rotula os candidatos em "Ativos" (1) ou "Inativos" (0) com base no ponto de corte numérico estipulado pelo usuário.

### 2.4. Espaço Químico, Descritores e Exclusão de Outliers
Para extrair um padrão matemático, o ScreenSAR disponibiliza diversas *fingerprints* com parâmetros otimizáveis: Morgan (similares a ECFP), MACCS e RDKit path-based. 
Possui **Ferramentas de Análise Visual de Espaço Químico**, tais como a análise exploratória bidimensional gerada usando Análise de Componentes Principais (PCA) e a **Análise de Scaffolds (Bemis-Murcko)** para identificar esquemas moleculares abundantes. Anomalias estatísticas são mapeadas e usuários podem remover iterativamente **outliers e estruturas aberrantes**.

### 2.5. Modelagem por Machine Learning (`ModeladorQSAR`)
Os modelos treinam automaticamente sobre o conjunto limpo sob premissas de validação cruzada rigorosa:
*   **Algoritmos Nativos**: Random Forest, Máquinas de Vetores de Suporte (SVM), Gradient Boosting, K-Nearest Neighbors (KNN), e Regressão Logística.
*   **Índice de Modelabilidade (MODI)**: Calcula matematicamente se a variação intrínseca do conjunto de dados permite um aprendizado coerente, informando ao pesquisador o estado inicial dos dados.
*   **Métricas e Avaliação Superior**: As pontuações são classificadas pelo Coeficiente de Correlação de Matthews (MCC), além do F1-Score, Sensibilidade, Especificidade, AUC (gerando curvas ROC comparativas) e Acurácia.

### 2.6. Módulo Avançado de Triagem Virtual (Virtual Screening)
No modo preditivo, os cientistas importam seus modelos de forma isolada. Este ambiente possui refinamentos para garantir alta confiabilidade das predições:
* **Domínio de Aplicabilidade (AD):** Cada modelo traz consigo métricas sobre quais instâncias o treinaram. Cada nova molécula fornecida pelo usuário na triagem tem a sua "distância" mapeada perante o espaço químico do banco de dados conhecido. Moléculas muito distantes caem fora do *Domain of Applicability*, guiando pesquisadores com alertas sobre falsos positivos por não estarem na mesma área de competência química da IA.
* **Mapeamento de Sobreposição do Treino (`In_Training_Set`):** Faz um cruzamento em tempo real validando se as moléculas agora descobertas com alta "confiança" de atividade foram meramente reproduções "decoradas" por estarem exatas na base matriz inicial de ensinamento, priorizando descobertas novas *de facto*.
* **Processamento Fracionado (Chunk Engine):** Mitiga o consumo excessivo de memória RAM rodando a base infinita do usuário em pedaços temporários configuráveis e autolimpados (ex: blocos de 5.000 ou 10.000).

## 3. Resultados

### 3.1. Experiência do Usuário (UX)
A plataforma consolida as premissas em uma interface única "em uma página só" a partir do login, dotada de um painel esquerdo modular interativo. Fornece ferramentas visuais gráficas alimentadas por bancos reativos e interliga o mundo da manipulação via web a lógicas locais extremamente eficientes da biblioteca RDKit, outrora restritas a terminais de comando.

### 3.2. Relatórios Consolidados Automatizados
Na ponta da esteira preditiva e da modelagem, arquivos PDF dinâmicos são criados instantaneamente contendo estatísticas da base original, métricas de divisão (Train/Test Split), curvas de convergência, e registro dos modelos mais eficazes, padronizando a documentação formal do estudo de desenho de fármacos para teses ou indústrias reguladas (como o cumprimento parcial de manuais da OECD para elaboração de modelos QSAR locais).

## 4. Pontos Fortes e Limitações

### 4.1. Pontos Fortes
*   **Automação Sistêmica**: Redução drástica no hiato de integração entre preparar um "banco sujo" colhido em mineração no ChEMBL e o uso final de IAs sofisticadas no mesmo dado.
*   **Transparência Dimensional Aplicada**: Identificação imediata sobre confiança e "viéses" com as colunas dedicadas do Domínio de Aplicabilidade e Presença Histórica no Treinamento geradas no output CSV de triagem virtual.
*   **Democratização Global**: Com o suporte nativo do menu interativo de até múltiplas moedas idiomáticas no painel esquerdo sem impactar desempenho de backend.

### 4.2. Limitações
*   **Simplificação Bidimensional**: Ao neutralizar estereoquímica como via de regra para estabilizar modelos em larga escala sobre "espinhas dorsais" (*scaffolds*), o Screening não abrange complexidades enancioméricas apuradas dependentes de ancoramento tridimensional complexo (ex: Docking guiado ou simulações MD seriam sequencias lógicas do output).
*   **Estratificação Limitada por Redes Neurais Profundas**: Foco maior atual reside nos classificadores de sub-símbolos e de ML tradicionais focados em rapidez, necessitando eventual extensão aos modelos generativos diretos em nuvem escalonada. 

## 5. Conclusão

O **ScreenSAR** democratiza drasticamente o uso correto de ferramentas quimioinformáticas nas descobertas preliminares da farmácia, aproximando equipes interdisciplinares entre a bancada orgânica empírica e cientistas de dados especializados. Ao absorver toda a engenharia estressante de limpeza química, normalização e controle de "vazamento de competência de dados" (*Applicability Domain* e *Overfitting*), capacitando qualquer pesquisador ou pequeno laboratório a não somente modelar com precisão clínica, como a varrer rapidamente catálogos comerciais imensos (como milhões de reagentes) buscando encontrar os inibidores ocultos mais propícios. Extensões futuras priorizarão modelagem de toxicidade combinada e integrações em nuvem.

---
*Gerado via Assistente Baseado em IA ScreenSAR.*
