import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw

batch_size = 64
learning_rate = 0.001
num_epochs = 10
model_path = "mnist_cnn.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def train():
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), model_path)

def predict(img):
    model.eval()
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
        return predicted

def evaluate():
    correct = 0
    total = 0
    for image, label in test_loader:
        image, label = image.to(device), label.to(device)
        total += label.size(0)
        correct += (predict(image) == label).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

def load_model():
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")


class DigitRecognizerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Handwritten Digit Recognizer")
        self.canvas = tk.Canvas(master, width=280*2, height=280*2, bg="white")
        self.canvas.grid(row=0, column=0, columnspan=2)
        self.clear_button = tk.Button(master, text="Clear", command=self.clear_canvas)
        self.clear_button.grid(row=1, column=0)
        self.predict_button = tk.Button(master, text="Predict", command=self.predict_digit)
        self.predict_button.grid(row=1, column=1)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.image = Image.new("L", (280*2, 280*2), color=0)
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        x, y = event.x, event.y
        r = 12
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280*2, 280*2), color=0)
        self.draw = ImageDraw.Draw(self.image)

    def predict_digit(self):
        img = self.image.resize((28, 28))
        img_tensor = transform(img).unsqueeze(0).to(device)

        digit = predict(img_tensor).item()
        print(f"Predicted Digit: {digit}")
        tk.messagebox.showinfo("Prediction", f"The digit is: {digit}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    try:
        load_model()
    except FileNotFoundError:
        print("No saved model found. Training a new model")
        train()
        evaluate()

    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()