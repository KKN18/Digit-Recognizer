# Digit-Recognizer
AI model recognizing digit image

## Environment
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bQtLXtg7eMvmWmvYppJ9FKW1XAf_86LT#scrollTo=dc3rIWyriNtO)

## Dataset
<img width="500" src="https://user-images.githubusercontent.com/63842546/214025128-ab8c5fe9-5df3-4516-a4e4-01eceb66082a.png"/>
[Kaggle Dataest](https://www.kaggle.com/competitions/digit-recognizer/data)

## Model
### Predict probability of each digit from 0 to 10

```python
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1) # 28
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) # 14

        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1) # 14
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2) # 7

        self.cnn3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1) # 14
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2) # 3

        self.fc1 = nn.Linear(64 * 3 * 3, 10)
```

## Result
### Input Image
<img width="200" src="https://user-images.githubusercontent.com/63842546/214025450-080e1f81-d934-49ad-bf2a-0b6f638fa9f7.png"/>

predicted = 2

### Input Image
<img width="200" src="https://user-images.githubusercontent.com/63842546/214025612-e14f0d8a-9bef-4958-8516-67d51ae9a16a.png"/>

predicted = 7

### Input Image
<img width="200" src="https://user-images.githubusercontent.com/63842546/214026083-9d2c2c52-8625-4e3e-abfd-d73d1d9e988d.png"/>

predicted = 5

### Input Image
<img width="200" src="https://user-images.githubusercontent.com/63842546/214025967-3ed9a560-cd38-44a1-a674-53c36da5bb7b.png"/>

predicted = 6
