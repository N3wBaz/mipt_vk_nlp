import torch
import torch.nn as nn
from typing import List, Optional, Callable, Union
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from scripts.model import Model

class Trainer:
    """
    Класс для обучения и оценки модели.
    """
    def __init__(
            self,
            model: Model,
            train_dataset: Union[Dataset, List[Tensor]],
            eval_dataset: Union[Dataset, List[Tensor]],
            n_epochs: int = 3,
            lr: float = 1e-5,
            train_batch_size: int = 1,
            eval_batch_size: int = 1,
            eval_steps: Optional[int] = None,
            collator: Optional[Callable[[List[List[int]]], Tensor]] = None,
            ignore_index: int = -100
    ):
        self.model = model
        self.loss_func = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collator
        )
        self.eval_loader = DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=collator
        )
        self.n_epochs = n_epochs
        self.eval_steps = eval_steps

    def calc_loss(self, logits: Tensor, y: Tensor) -> Tensor:
        """
        Вычисляет потери (loss) на основе предсказанных логитов и целевых меток.
        """
        # Логиты имеют форму (batch_size, seq_len, vocab_size), но для функции потерь нужно (batch_size * seq_len, vocab_size)
        # Истинные метки y имеют форму (batch_size, seq_len), их нужно развернуть в (batch_size * seq_len)
        return self.loss_func(logits.view(-1, logits.size(-1)), y.view(-1))

    def train(self) -> None:
        """
        Запускает процесс обучения модели. После каждой эпохи выводит значение потерь.
        Если задан eval_steps, проводит оценку через каждые eval_steps итераций.
        """
        progress_bar = tqdm(total=self.n_epochs * len(self.train_loader))
        iterations = 0
        for _ in range(self.n_epochs):
            for ids in self.train_loader:
                iterations += 1
                self.model.train()
                
                # Готовим входы (текущие токены) и выходы (следующие токены)
                x = ids[:, :-1]  # Все токены, кроме последнего
                y = ids[:, 1:]   # Все токены, начиная со второго
                
                # Получаем логиты и считаем лосс
                logits, _ = self.model(x)
                loss = self.calc_loss(logits, y)
                
                progress_bar.update()
                progress_bar.set_description(f'epoch={iterations // len(self.train_loader)}, loss={loss.item()}')
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Если eval_steps задано, оцениваем модель
                if self.eval_steps is not None and iterations % self.eval_steps == 0:
                    print(f'epoch={iterations // len(self.train_loader)}, eval_loss={self.evaluate()}')

    def evaluate(self) -> float:
        """
        Оценивает модель на наборе данных для оценки, вычисляя среднее значение потерь.
        """
        self.model.eval()
        total_loss = 0.0
        
        for ids in self.eval_loader:
            # Готовим входы (текущие токены) и выходы (следующие токены)
            x = ids[:, :-1]  # Все токены, кроме последнего
            y = ids[:, 1:]   # Все токены, начиная со второго
            
            with torch.no_grad():
                # Получаем логиты и считаем лосс
                logits, _ = self.model(x)
                loss = self.calc_loss(logits, y)
                total_loss += loss.item() / len(self.eval_loader)
        
        return total_loss
