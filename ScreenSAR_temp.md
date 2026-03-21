**SCREENSAR: An Automated QSAR Factory for Predictive Virtual
Screening**

**\
**[Carlos S. H. Shiraishi]{.ul}^12^, Marcus T. Scotti^3,4^, Eugene
Murakov^4^, Miguel A. Prieto^2^, Sandrina A. Heleno^1^ and Rui M. V.
Abreu^1^

^1^ CIMO-ESA, Instituto Politécnico de Bragança, Campus de Sta Apolónia,
Apartado 1172, 5301-855 Bragança, Portugal.\
^2^ Universidade de Vigo, Nutrition and Bromatology Group, Department of
Analytical Chemistry and Food Science, Faculty of Science, E-32004
Ourense, Spain.\
^3^ Department of Chemistry, Federal University of Paraíba, João Pessoa,
Brazil.\
^4^ Laboratory for Molecular Modeling, UNC Eshelman School of Pharmacy,
The University of North Carolina at Chapel Hill.

**\
**

**Resumo** A modelagem de Relações Quantitativas Estrutura-Atividade
(QSAR) é um pilar da descoberta moderna de fármacos, permitindo a
previsão da atividade biológica a partir da estrutura química. No
entanto, a confiabilidade dos modelos QSAR depende fortemente da
qualidade dos dados de entrada e da robustez do fluxo de trabalho de
modelagem. Este artigo apresenta o ScreenSAR, uma plataforma abrangente
e automatizada projetada para simplificar a curadoria de dados químicos,
o desenvolvimento de modelos de aprendizado de máquina e a triagem
virtual. A ferramenta oferece uma arquitetura modular que compreende
limpeza padronizada de dados, engenharia de características
(fingerprints de Morgan, MACCS, RDKit), análise de outliers e
treinamento de múltiplos modelos com ranqueamento automático de
desempenho. Os modelos validados são serializados como artefatos
portáveis (.pkl), permitindo sua implantação direta em um módulo
dedicado para a triagem virtual (screening) eficiente de novas
bibliotecas químicas. Demonstramos a utilidade do ScreenSAR em aprimorar
a reprodutibilidade, a eficiência e a acessibilidade em fluxos de
trabalho de pesquisa farmacêutica.

![A screen shot of a computer screen Description automatically
generated](media/image1.png){width="5.473567366579178in"
height="2.9854166666666666in"}

## 1. Introdução

### A IA tem o potencial de revolucionar a criação de novos medicamentos, substituindo os métodos tradicionais de "tentativa e erro", que são lentos, caros e altamente dependentes de extensivas campanhas experimentais. Em vez de testar milhares de moléculas de forma empírica, algoritmos de aprendizado de máquina conseguem aprender padrões complexos entre estrutura química e atividade biológica, priorizando apenas os candidatos mais promissores para validação experimental. Ferramentas de IA conseguem analisar bases de dados massivas para prever, com elevada precisão, parâmetros como eficácia, seletividade, propriedades farmacocinéticas (ADME) e toxicidade antes mesmo da síntese ou teste in vitro. Essa abordagem reduz custos, diminui o tempo de desenvolvimento e aumenta a taxa de sucesso nas fases pré-clínicas, tradicionalmente marcadas por elevadas taxas de falha. Essas tecnologias também auxiliam na identificação de novos alvos terapêuticos por meio da integração de dados ômicos (genômica, proteômica, transcriptômica) e redes biológicas, permitindo compreender mecanismos moleculares complexos associados a doenças multifatoriais. Além disso, modelos preditivos podem antecipar interações medicamentosas e efeitos off-target, contribuindo para o desenho de fármacos mais seguros e específicos. A indústria farmacêutica passa, portanto, por uma mudança de paradigma impulsionada pela Inteligência Artificial (IA) e pelo Aprendizado de Máquina (ML). A integração de Big Data proveniente de triagens de alto rendimento (HTS), dados clínicos, literatura científica e repositórios públicos como ChEMBL e PubChem possibilitou o desenvolvimento de modelos preditivos robustos capazes de acelerar significativamente a fase de otimização de compostos-guia (lead optimization). No entanto, apesar do avanço tecnológico, existe um desafio crítico: a democratização do acesso a essas ferramentas. Grande parte dos modelos de IA em drug discovery exige conhecimento avançado em programação, manipulação de bases de dados, quimioinformática e infraestrutura computacional. Em uma era em que muitos pesquisadores da área biológica e farmacêutica não possuem formação em ciência de dados ou desenvolvimento de software, torna-se essencial a criação de plataformas intuitivas e orientadas ao usuário. Aplicativos científicos com interfaces gráficas amigáveis (GUI), capazes de receber entradas simples como SMILES ou estruturas químicas e retornar predições interpretáveis de bioatividade, toxicologia e similaridade estrutural, representam um passo fundamental para aproximar a IA do laboratório. Essas soluções reduzem barreiras técnicas, aumentam a reprodutibilidade e permitem que o pesquisador foque na interpretação biológica e na tomada de decisão estratégica --- e não na complexidade do código. Assim, o verdadeiro impacto da IA na indústria farmacêutica não depende apenas da sofisticação dos algoritmos, mas também da capacidade de transformar modelos complexos em ferramentas acessíveis, transparentes e centradas no usuário, promovendo uma ciência mais inclusiva, colaborativa e eficiente.

### 1.1. O Paradoxo da Qualidade dos Dados

### Apesar desses avanços, o princípio "Garbage In, Garbage Out" (Lixo entra, Lixo sai) continua sendo o principal gargalo\[1\]. Estudos estimam que até 80% do tempo de um cientista de dados é gasto na curadoria, padronização e limpeza de dados, em vez de na modelagem propriamente dita\[2\]. Representações químicas inconsistentes, ambiguidades estereoquímicas, erros de anotação, duplicações estruturais, diferentes protocolos experimentais e ruído biológico em grandes bases de dados públicas frequentemente levam a modelos QSAR não confiáveis e com baixo poder preditivo prospectivo \[3\]. Bases amplamente utilizadas como ChEMBL \[4\] e PubChem \[5\] contêm dados extremamente valiosos, provenientes de diferentes laboratórios, protocolos e condições experimentais. No entanto, variações nos tipos de ensaio, como IC₅₀ e Ki, diferenças de pH, utilização de distintas linhagens celulares e critérios heterogêneos para definir atividade biológica podem introduzir vieses significativos e inconsistências nos conjuntos de dados e gerando ruidos. Sem curadoria rigorosa, que envolva:

i.  ### padronização estrutural das moléculas.

ii. ### normalização de unidades e escalas de atividade.

iii. ### remoção de duplicatas e valores discrepantes.

iv. ### Definição clara e consistente dos endpoints biológicos.

Modelos treinados com conjuntos artificialmente equilibrados podem
otimizar métricas como balanced accuracy, mas não refletem adequadamente
o desempenho em triagem virtual, onde compostos ativos são raros. O
estudo mostra que modelos treinados com dados naturalmente
desbalanceados apresentam maior taxa de verdadeiros positivos nas
melhores predições e melhor hit rate quando avaliados por métricas mais
apropriadas, como positive predictive value. Assim, o desbalanceamento
não deve ser necessariamente corrigido, mas interpretado de acordo com o
objetivo prático do modelo \[6\]. Modelos treinados em dados mal curados
tendem a apresentar excelente desempenho interno, mas falham quando
aplicados prospectivamente, evidenciando problemas de generalização e
domínio de aplicabilidade (Figura 1). Modelos treinados em dados mal
curados tendem a apresentar excelente desempenho interno, mas falham
quando aplicados prospectivamente, evidenciando problemas de
generalização e domínio de aplicabilidade.

![A person in a robe standing in front of a globe Description
automatically generated](media/image2.png){width="6.268055555555556in"
height="3.41875in"}

**Figura 1**: Representação conceitual do Domínio de Aplicabilidade (DA)
em modelos QSAR para a descoberta de fármacos. Inspirada na clássica
gravura de Flammarion, a ilustração demarca a fronteira algorítmica
entre a interpolação confiável e a extrapolação especulativa em modelos
de aprendizado de máquina. O interior da cúpula (\"Espaço Químico
Conhecido\") define o espaço multivariado de características (feature
space) coberto pelo conjunto de treinamento; moléculas situadas nesta
região permitem previsões robustas com alta confiabilidade estatística
(interpolação). A barreira física rompida pelo observador simboliza o
limiar matemático do DA (por exemplo, distância de Mahalanobis ou
limites de leverage). O espaço exterior (\"Extrapolação\") ilustra o
vasto e inexplorado universo químico; previsões de IA para novos
compostos nesta região carecem de suporte estatístico prévio, resultando
em alta incerteza e exigindo validação experimental rigorosa.

### 

### Portanto, a qualidade dos dados não é apenas uma etapa preliminar, mas o alicerce de qualquer estratégia baseada em IA para descoberta de fármacos. Investimentos em pipelines robustos de curadoria, validação externa rigorosa, definição de domínio de aplicabilidade e métricas alinhadas ao objetivo final (por exemplo, priorização em virtual screening) são essenciais para transformar modelos computacionais em ferramentas verdadeiramente confiáveis e translacionais.

### 1.2. Soluções Existentes vs. ScreenSAR

As soluções atuais de ponta para a curadoria de dados variam de
plataformas comerciais (ex.: Pipeline Pilot, nós de limpeza no KNIME) a
bibliotecas pesadas em código (ex.: RDKit, MolVS).

-   **Ferramentas comerciais** oferecem interfaces de usuário refinadas,
    mas apresentam custos de licenciamento proibitivos e opacidade de
    \"caixa-preta\".

-   **Bibliotecas de código aberto** oferecem transparência, mas exigem
    grande experiência em programação, criando uma barreira para
    químicos medicinais.

Como demonstrado na **Tabela 1**, as ferramentas de quimioinformática
atuais frequentemente forçam um compromisso entre acessibilidade e
facilidade de uso. Enquanto suítes comerciais como AutoQSAR e Pipeline
Pilot oferecem alta automação a um custo proibitivo com algoritmos de
\'caixa-preta\', plataformas de código aberto como o KNIME exigem uma
curva de aprendizado íngreme para a construção de fluxos de trabalho.
Ferramentas baseadas na web, como o OCHEM, levantam preocupações de
confidencialidade devido à necessidade de *upload* de dados
proprietários.

**Tabela 1: Comparação entre o ScreenSAR e plataformas de
quimioinformática estabelecidas no mercado.\
**

**\
**

  **Plataforma / Software**    **Categoria**           **Custo / Licença**        **Curva de Aprendizado**         **Nível de Automação (AutoML)**   **Privacidade dos Dados**    **Relatórios Automatizados**   **Reference**
  ---------------------------- ----------------------- -------------------------- -------------------------------- --------------------------------- ---------------------------- ------------------------------ ---------------
  **ScreenSAR**                Aplicação Web / Local   Gratuito (*Open-source*)   Baixa (Interface guiada)         Alto (*End-to-end*)               Total (Execução local)       Sim (PDF nativo)               
  **AutoQSAR**(Schrödinger)    Suíte Comercial         Comercial                  Baixa                            Alto (*Black-box*)                Total                        Sim                            \[7\]
  **KNIME** (+ RDKit)          Baseado em Nós          Gratuito (*Open-source*)   Alta (Exige montagem de fluxo)   Configurável (Manual)             Total                        Requer configuração extra      \[8\]
  **Pipeline Pilot**(BIOVIA)   Corporativo / Nós       Comercial                  Média / Alta                     Configurável                      Total (Servidor próprio)     Sim                            \[9\]
  **OCHEM**                    Plataforma Web          Gratuito (Acadêmico)       Média                            Alto                              Parcial (*Upload*na nuvem)   Não nativamente                \[10\]
  **DataWarrior**              Software Desktop        Gratuito (*Open-source*)   Baixa                            Baixo (Foco em visualização)      Total                        Não                            \[11\]

O **ScreenSAR** avança o estado da arte ao democratizar o acesso a uma
curadoria robusta. Ele combina algoritmos rigorosos e padronizados de
bibliotecas de código com a acessibilidade de uma interface web moderna,
preenchendo efetivamente a lacuna entre a mineração de dados brutos e
modelos QSAR acionáveis e de alta qualidade.

## 2. Materiais e Métodos

### 2.1. Componentes Operacionais Centrais

O ScreenSAR é estruturado em torno de dois pilares funcionais distintos,
porém interconectados, projetados para separar o desenvolvimento do
modelo de sua aplicação:

1.  **Fabricação e Otimização de Modelos**: Este componente ingere dados
    brutos de atividade biológica (ex.: do ChEMBL ou de ensaios
    internos) para executar o pipeline de ponta a ponta: curadoria de
    dados, engenharia de características e aprendizado de máquina. Ele
    avalia automaticamente múltiplos algoritmos e identifica o modelo
    com melhor desempenho, que é então serializado e exportado como um
    artefato .pkl (Python Pickle).

2.  **Triagem Virtual e Implantação**: Este componente autônomo permite
    que os usuários importem um modelo .pklpré-treinado para triar
    bibliotecas químicas externas. Ele contorna a carga computacional do
    treinamento, focando exclusivamente na geração rápida de descritores
    e na previsão de atividade para novas entidades moleculares.

### 2.2. Arquitetura do Sistema

A aplicação foi construída usando **Python** e o
framework **Streamlit**, garantindo uma interface de usuário web
responsiva. O código-fonte segue um padrão de design modular estruturado
da seguinte forma:

Plaintext

src/

├── core/ \# Lógica de Negócios Científica

│ ├── curation.py \# Limpeza e padronização de dados

│ ├── modeling.py \# Treinamento e validação de modelos QSAR

│ └── applicability.py \# Avaliação do Domínio de Aplicabilidade
(baseado em k-NN)

├── ui/ \# Componentes da Interface de Usuário

│ ├── dashboard.py \# Layout principal da aplicação

│ ├── prediction.py \# Interface de triagem virtual

│ └── sidebar.py \# Controle de navegação

└── utils/ \# Utilitários Auxiliares

├── report.py \# Geração automatizada de PDF

└── translations.py \# Dicionário de Internacionalização (i18n)

*(Figura 2: Fluxo de Trabalho Conceitual do ScreenSAR - Inserir diagrama
aqui)*

### 2.3. Pipeline de Curadoria de Dados

O módulo CuradoriaQSAR automatiza etapas críticas para garantir a
integridade dos dados:

1.  **Normalização Química**: Canonicalização de SMILES usando RDKit,
    remoção de sais/solventes, seleção do maior fragmento orgânico e
    neutralização de estereoisômeros para uma representação 2D
    consistente.

2.  **Padronização de Atividade**: Conversão automática de diversas
    unidades (μM, mM, M) para uma linha de base padrão em **nM**.
    Opcionalmente, calcula o **pIC50** (-log10(M)) para linearizar a
    atividade para a modelagem.

3.  **Gerenciamento de Duplicatas**: Identifica duplicatas biológicas
    (mesmo InChIKey/SMILES).

    -   *Concordantes*: Agrega valores usando a **Média Geométrica**.

    -   *Discordantes*: Remove compostos se a variação de atividade
        exceder 1 unidade logarítmica (10x).

4.  **Classificação Binária**: Atribui rótulos de \"Ativo\" (1) ou
    \"Inativo\" (0) com base em limiares definidos pelo usuário (ex.:
    IC50 \< 100 nM).

### 2.4. Engenharia de Características e Espaço Químico

Para preparar os dados para aprendizado de máquina, o ScreenSAR oferece
opções flexíveis de geração de descritores:

-   **Fingerprints**:
    Suporta *fingerprints* topológicos **Morgan** (tipo ECFP), **MACCS
    Keys** e **RDKit**. Os usuários podem personalizar o comprimento em
    bits (ex.: 1024, 2048) e o raio.

-   **Visualização**: A **Análise de Componentes Principais (PCA)** é
    usada para reduzir a dimensionalidade e visualizar a cobertura do
    espaço químico de compostos ativos vs. inativos.

-   **Detecção de Outliers**: Análise estatística (média ± 3 DP)
    sinaliza automaticamente potenciais \"abismos de atividade\"
    (*activity cliffs*) ou erros experimentais.

### 2.5. Modelagem por Aprendizado de Máquina

O módulo ModeladorQSAR facilita o treinamento robusto de modelos:

-   Algoritmos: *Random Forest*, Máquinas de Vetores de Suporte
    (SVM), *Gradient Boosting*, *K-Nearest Neighbors* (KNN) e Regressão
    Logística.

-   Modelabilidade: Calcula o MODI (Índice de Modelabilidade) para
    avaliar se o conjunto de dados é adequado para modelagem (MODI \>
    0,65).

-   Validação: Utiliza a divisão estratificada de treino-teste. O
    desempenho é avaliado via Acurácia, *F1-Score*, Sensibilidade,
    Especificidade e Área sob a Curva ROC (AUC).

-   Ranqueamento: Os modelos são classificados automaticamente
    pelo Coeficiente de Correlação de Matthews (MCC), uma métrica
    robusta para conjuntos de dados balanceados e desbalanceados.

### 2.6. Previsão e Triagem Virtual

Um **Módulo de Triagem Virtual** dedicado permite que os usuários
implantem modelos treinados (artefatos .pkl) para prever a atividade de
compostos novos e inéditos. Este módulo aceita listas brutas de SMILES,
gera os descritores apropriados de forma dinâmica e emite previsões
binárias com pontuações de probabilidade.

#### 2.6.1. Escalonamento e Otimização de Memória (Chunking)

Para permitir a triagem de bibliotecas químicas ultra-grandes (milhões de
compostos) em hardware padrão, o ScreenSAR implementa um **Mecanismo de
Processamento em Bloco (Chunk-based Processing)**. Em vez de carregar
todo o conjunto de dados na memória RAM, o sistema:

1.  **Extração Iterativa**: Lê e processa moléculas em lotes
    configuráveis (padrão de 5.000 compostos por bloco).

2.  **Transformação em Tempo Real**: Gera descritores moleculares e
    realiza predições para cada bloco sequencialmente.

3.  **Gerenciamento de Memória**: Limpa descritores temporários após cada
    lote para manter um uso de memória constante e baixo, prevenindo
    erros de \"Falta de Memória\" (OOM) durante triagens de larga escala
    envolvendo **milhões de compostos**.

### 2.7. Detalhes de Implementação e Estrutura do Código

A plataforma ScreenSAR é projetada com o princípio de separação de
responsabilidades, distinguindo a lógica computacional da renderização
da interface. Essa modularidade garante fácil manutenção e
escalabilidade.

#### 2.7.1. Lógica Central (src/core)

A espinha dorsal da aplicação reside em src/core, contendo duas classes
principais que governam o fluxo de trabalho científico:

-   **CuradoriaQSAR** (curation.py): Encapsula todo o pipeline de
    preparação de dados.

    -   *Inicialização*: Aceita dados brutos (CSV/Excel) e define
        limiares de atividade.

    -   *Limpeza Química*: O
        método \_limpar_quimica utiliza rdkit.Chem.SaltRemover para
        remover sais, identifica o maior fragmento orgânico e gera
        SMILES canônicos para garantir representação estrutural única.

    -   *Padronização*: O método executar_pipeline orquestra a conversão
        de unidades (para nM), agrega entradas duplicadas usando médias
        geométricas (para dados concordantes) e filtra a discordância
        biológica (diferença \> 1 unidade logarítmica).

    -   *Geração de Características*: O
        método gerar_fingerprints suporta a geração dinâmica
        de *fingerprints* de Morgan, MACCS e RDKit.

-   **ModeladorQSAR** (modeling.py): Gerencia o ciclo de vida do
    aprendizado de máquina.

    -   *Preparação de Dados*: O método gerar_dados alinha
        os *fingerprints* moleculares (X) com os rótulos de atividade
        biológica (y).

    -   *Treinamento de Modelos*: A função treinar_avaliar implementa
        uma divisão estratificada de treino-teste (padrão 80:20) e
        treina cinco algoritmos distintos (RF, SVM, GBM, KNN, LR)
        usando scikit-learn.

    -   *Modelabilidade*: O método calcular_modi calcula o índice MODI
        usando similaridade de Jaccard para avaliar a viabilidade de
        modelagem do conjunto de dados.

    -   *Domínio de Aplicabilidade*: Integrado via
        classe ApplicabilityDomain, empregando a abordagem de k-Vizinhos
        Mais Próximos (k=5) para definir o espaço químico e
        pré-estabelecer limiares de confiabilidade (Z \< 3.0).

#### 2.7.2. Interface do Usuário, Relatórios e Utilitários

O módulo src/ui gerencia o frontend baseado em Streamlit, com scripts
dedicados para cada aba funcional (ex.: prediction.py para triagem
virtual), garantindo que os cálculos rigorosos de backend sejam
acessíveis através de um painel intuitivo. Adicionalmente, o
módulo src/utils/report.py facilita a comunicação científica:

-   **PDFReport**: Utilizando a biblioteca fpdf, esta classe automatiza
    a geração de documentos. Ela formata especificamente os resultados
    científicos, incluindo curvas ROC e tabelas de métricas, em uma
    estrutura padronizada adequada para submissão regulatória ou
    arquivamento interno.

## 3. Resultados

### 3.1. Experiência do Usuário

A plataforma fornece um painel unificado onde pesquisadores podem
navegar visualmente por todo o pipeline. A integração de um **Resumo
Gráfico** na página inicial comunica imediatamente a lógica do fluxo de
trabalho. O suporte multilíngue garante acessibilidade para uma
comunidade de pesquisa global.

### 3.2. Relatórios Automatizados

O ScreenSAR gera **Relatórios em PDF** profissionais e prontos para a
indústria. Esses documentos resumem o modelo com melhor desempenho,
fornecem matrizes de confusão detalhadas e renderizam curvas ROC de alta
resolução, facilitando a documentação regulatória e relatórios internos.

## 4. Pontos Fortes e Limitações

### 4.1. Pontos Fortes

-   **Automação de Ponta a Ponta**: A ferramenta preenche a lacuna entre
    a recuperação de dados e a modelagem preditiva, reduzindo
    significativamente o \"tempo até a descoberta\" (*time-to-insight*)
    para químicos medicinais.

-   **Integridade de Dados**: O uso de médias geométricas para
    duplicatas concordantes e a padronização rigorosa de unidades
    garante entradas de alta qualidade para modelos de ML.

-   **Inclusividade**: A interface multilíngue (com suporte a 5 idiomas)
    quebra barreiras linguísticas em equipes globais de pesquisa.

-   **Transparência**: A arquitetura de código aberto e fluxos de
    trabalho padronizados aprimoram a reprodutibilidade científica.

### 4.2. Limitações

-   **Sensibilidade da Infraestrutura**: Embora o *chunking* otimize a
    memória permitindo triagens de **milhões de compostos**, o tempo
    total de execução para conjuntos de dados extremamente grandes
    (\>10 milhões) permanece dependente da capacidade do CPU e
    velocidade de leitura/escrita do disco.

-   **Simplificação 2D**: A atual etapa de neutralização de
    estereoquímica, embora útil para triagem geral, pode ignorar nuances
    de abismos de atividade impulsionados por centros quirais
    específicos.

-   **Dependência de Entrada**: O pipeline é atualmente otimizado para
    dados formatados no padrão ChEMBL; conjuntos de dados proprietários
    internos podem exigir adaptação de pré-processamento.

## 5. Conclusão

O ScreenSAR democratiza o acesso a ferramentas quimioinformáticas de
alta qualidade. Ao integrar curadoria rigorosa de dados, avaliação de
domínio de aplicabilidade e aprendizado de máquina avançado em um
ambiente amigável, ele capacita pesquisadores a focar na geração de
hipóteses em vez de lidar manualmente com dados. Capacidades futuras
focarão na integração de *deep learning* e escalabilidade nativa em
nuvem.

## 6. Referências

1.  OECD Principles for the Validation, for Regulatory Purposes, of
    (Q)SAR Models.

2.  Tropsha, A. (2010). Best Practices for QSAR Model
    Development. *Molecular Informatics*.

3.  Fourches, D., et al. (2010). Trust, but Verify: On the Importance of
    Chemical Structure Curation. *Journal of Chemical Information and
    Modeling*.
