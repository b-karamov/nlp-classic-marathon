# GRU

## Архитектура
GRU-ячейка с воротами обновления и сброса.

- Вход: $x_t \in \mathbb{R}^D$
- Скрытое состояние: $h_t \in \mathbb{R}^H$

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
    end

    subgraph Cell["GRU cell"]
        direction TB
        Z["z_t"]:::gate
        R["r_t"]:::gate
        Htilde["h~_t"]:::core
        H["h_t"]:::state
        R --> Htilde
        Z --> H
        Htilde --> H
    end

    subgraph Output
        direction TB
        Y["output layer"]:::output
    end

    X --> Z
    X --> R
    X --> Htilde
    Hprev --> Z
    Hprev --> R
    Hprev --> Htilde
    Hprev --> H
    H --> Y
```

## Теория
GRU упрощает LSTM, убирая отдельную память $c_t$, но оставляя ворота,
что часто дает сопоставимое качество при меньшей сложности.

## Формулы
**Ворота и состояние (шаг $t$)**
$$
z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z),\quad
r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)
$$
$$
\tilde{h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h)
$$
$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
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
- Последовательности средней длины

## Плюсы
- Меньше параметров, чем LSTM
- Часто быстрее обучается
- Хороший компромисс качество/скорость

## Минусы
- Меньше контроля над памятью, чем в LSTM
- Может проигрывать на очень длинном контексте
