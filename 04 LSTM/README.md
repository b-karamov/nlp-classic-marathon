# LSTM

## Архитектура
LSTM-ячейка с входными/выходными/забывающими воротами и памятью $c_t$.

- Вход: $x_t \in \mathbb{R}^D$
- Скрытое состояние: $h_t \in \mathbb{R}^H$
- Состояние памяти: $c_t \in \mathbb{R}^H$

## Диаграмма
```mermaid
flowchart LR
    classDef input fill:#f7f3e9,stroke:#c8bda8,color:#3b2f2f;
    classDef core fill:#e3f2fd,stroke:#64b5f6,color:#0d47a1;
    classDef gate fill:#f3e5f5,stroke:#ba68c8,color:#4a148c;
    classDef state fill:#e8f5e9,stroke:#81c784,color:#1b5e20;
    classDef output fill:#fff3e0,stroke:#ffb74d,color:#e65100;

    subgraph Legend
        direction TB
        L1["input"]:::input
        L2["gate"]:::gate
        L3["state"]:::state
        L4["transform"]:::core
        L5["output"]:::output
    end

    subgraph Input
        direction TB
        X["x_t"]:::input
        Hprev["h_{t-1}"]:::state
        Cprev["c_{t-1}"]:::state
    end

    subgraph Cell["LSTM cell"]
        direction TB
        F["f_t"]:::gate
        I["i_t"]:::gate
        O["o_t"]:::gate
        Ctilde["c~_t"]:::core
        C["c_t"]:::state
        H["h_t"]:::state
        F --> C
        I --> C
        Ctilde --> C
        C --> H
        O --> H
    end

    subgraph Output
        direction TB
        Y["output layer"]:::output
    end

    X --> F
    X --> I
    X --> O
    X --> Ctilde
    Hprev --> F
    Hprev --> I
    Hprev --> O
    Hprev --> Ctilde
    Cprev --> C
    H --> Y
```

## Теория
LSTM добавляет память $c_t$ и ворота, что уменьшает затухание градиента
и улучшает работу с длинным контекстом.

## Формулы
**Ворота и состояние (шаг $t$)**
$$
f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f),\quad
i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)
$$
$$
o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o),\quad
\tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)
$$
$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t,\quad
h_t = o_t \odot \tanh(c_t)
$$

**Выходной слой (классификация)**
$$
\text{logits} = W_y h_T + b_y,\quad
p = \text{softmax}(\text{logits})
$$

**Лосс**
$$
L_{cls} = -\log p[y],\quad
L_{lm} = -\frac{1}{T}\sum_t \log p_t[y_t]
$$

## Применимые задачи
- Последовательностная классификация
- Языковое моделирование
- Временные ряды с долгими зависимостями

## Плюсы
- Лучше удерживает дальний контекст
- Стабильнее обучается, чем базовый RNN
- Универсальная архитектура для последовательностей

## Минусы
- Больше параметров и вычислений
- Медленнее обучения, сложнее тюнить
