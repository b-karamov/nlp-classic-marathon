# CNN

## Архитектура
Минимальная CNN для MNIST: **Conv → ReLU → MaxPool → FC → Softmax**.

- Вход: изображение $1 \times 28 \times 28$
- Свертки: $K$ фильтров $3 \times 3$
- Пулинг: $2 \times 2$ (stride 2)
- Выход: 10 классов

## Диаграмма
```mermaid
flowchart LR
    classDef input fill:#f7f3e9,stroke:#c8bda8,color:#3b2f2f;
    classDef core fill:#e3f2fd,stroke:#64b5f6,color:#0d47a1;
    classDef param fill:#ede7f6,stroke:#9575cd,color:#311b92;
    classDef output fill:#fff3e0,stroke:#ffb74d,color:#e65100;

    subgraph Legend
        direction TB
        L1["input"]:::input
        L2["block"]:::core
        L3["params"]:::param
        L4["output"]:::output
    end

    subgraph Input
        direction TB
        X["image 1x28x28"]:::input
    end

    subgraph FeatureExtractor["Conv block"]
        direction TB
        C["Conv 3x3 x K"]:::core
        R["ReLU"]:::core
        P["MaxPool 2x2"]:::core
        C --> R --> P
    end

    subgraph Classifier
        direction TB
        F["Flatten"]:::core
        FC["FC -> 10"]:::core
        F --> FC
    end

    subgraph Params
        direction TB
        Wc["W_conv (filters)"]:::param
        bc["b_conv (bias)"]:::param
        Wf["W_fc (classifier)"]:::param
        bf["b_fc (bias)"]:::param
    end

    subgraph Output
        direction TB
        Y["p(y=class)"]:::output
    end

    X --> C
    P --> F
    FC --> Y
    Wc --> C
    bc --> C
    Wf --> FC
    bf --> FC
```

## Теория
CNN использует локальные рецептивные поля и разделяемые фильтры, чтобы
выделять устойчивые к сдвигу признаки. Пулинг уменьшает размерность
и делает представление более инвариантным.

## Формулы
**Свертка**
$$
Y_{f,i,j} = \sum_{c=1}^C \sum_{u=1}^K \sum_{v=1}^K
W_{f,c,u,v} \cdot X_{c,i+u,j+v} + b_f
$$

**ReLU**
$$
\text{ReLU}(x) = \max(0, x)
$$

**MaxPool ($2\times 2$)**
$$
P_{c,i,j} = \max_{u,v \in \{0,1\}} X_{c,2i+u,2j+v}
$$

**Классификатор**
$$
z = \text{flatten}(P),\quad
\text{logits} = W z + b,\quad
p = \text{softmax}(\text{logits})
$$

## Применимые задачи
- Классификация изображений (MNIST, CIFAR)
- Извлечение локальных признаков
- Простые задачи компьютерного зрения

## Плюсы
- Мало параметров из-за разделения весов
- Хорошо ловит локальные паттерны
- Быстрее и стабильнее, чем полносвязные сети на изображениях

## Минусы
- Ограниченный контекст без углубления сети
- Требует подбора гиперпараметров (фильтры, пулинг)
- Упрощает геометрию, что может вредить точности
