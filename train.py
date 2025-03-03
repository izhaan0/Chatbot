import json
from nltk_utils import tokenize,stem,bag_of_words
import numpy as np

import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader



from model import NeuralNet

with open('intents.json' , 'r') as f:
    intents=json.load(f)
    
all_words=[]
tags=[]
xy = []       #Stores pattern and the corr3sponding tag

for intent in intents['intents']:
    tag=intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w=tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag)) 

ignore_words=['?','!',',','.']
all_words=[stem(w) for w in all_words if w not in ignore_words]
all_words=sorted(set(all_words))
tags=sorted(set(tags))

# print(all_words)
# print(tags)

#Creating our train data
X_train=[]
y_train=[]

for (pattern_sentence,tag) in xy:
    bag=bag_of_words(pattern_sentence,all_words)
    X_train.append(bag)
    
    label = tags.index(tag)
    y_train.append(label)   # 1 hot encoded vector
    
#Converting the training data to numpy array
X_train=np.array(X_train)
y_train=np.array(y_train)



    
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = torch.tensor(X_train, dtype=torch.float32)  # Convert to tensor
        self.y_data = torch.tensor(y_train, dtype=torch.long)  # Convert to tensor

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


#hyperparameters
batch_size=16
hidden_size=8
input_size=len(X_train[0])
output_size=len(tags)
learning_rate=0.001 
num_epochs=1000




dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size,hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

            # Forward pass
        output = model(words)
        loss = criterion(output, labels)

            # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss={loss.item():.4f}')

print(f'Final loss, Loss={loss.item():.4f}')

data={
    "model_state":model.state_dict(),
    "input_size":input_size,
    "output_size":output_size,
    "hidden_size":hidden_size,
    "all_words":all_words,
    "tags":tags
}
        
FILE ="data.pth"
torch.save(data,FILE)



print(f'training complete . file saved to {FILE}')    

