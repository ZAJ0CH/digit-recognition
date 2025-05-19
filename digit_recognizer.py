import os
import uuid
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog

batch_size = 64
test_batch_size = 256
learning_rate = 0.001
weight_decay = 1e-4
num_epochs = 64
writer = None
model_path = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomCrop(28, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

inference_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class TestImageDataset(Dataset):
    def __init__(self):
        self.img_dir ='./custom_test_dataset'
        self.image_files = [f for f in os.listdir(self.img_dir) if f.endswith('.png')]
        self.transform = inference_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        label = int(img_name.split('_')[-1].split('.')[0])

        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label

custom_test_dataset = TestImageDataset()

custom_test_loader = torch.utils.data.DataLoader(
    custom_test_dataset,
    batch_size=test_batch_size,
    shuffle=False
)

train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           download=True,
                                           transform=train_transform)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          download=True,
                                          transform=inference_transform)


train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size

train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(dataset=train_subset,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)

val_loader = DataLoader(dataset=val_subset,
                         batch_size=batch_size,
                         shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def train():
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total

        val_loss, val_accuracy = validate()

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        if writer is not None:
            writer.add_scalar('Training Loss', train_loss, epoch + 1)
            writer.add_scalar('Training Accuracy', train_accuracy, epoch + 1)
            writer.add_scalar('Validation Loss', val_loss, epoch + 1)
            writer.add_scalar('Validation Accuracy', val_accuracy, epoch + 1)

def validate():
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / len(val_loader)
    val_accuracy = 100 * correct / total

    return val_loss, val_accuracy



def test():
    model.eval()
    results = {}
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    results['test_accuracy'] = 100 * correct / total
    print(f"Test Accuracy: {results['test_accuracy']:.2f}%")

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in custom_test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    results['custom_test_accuracy'] = 100 * correct / total
    print(f"Custom Test Accuracy: {results['custom_test_accuracy']:.2f}%")

    global writer
    if writer is not None:
        try:
            writer.add_scalar('Test Accuracy', results['test_accuracy'])
            writer.add_scalar('Custom Test Accuracy', results['custom_test_accuracy'])
        except Exception as e:
            print(f"Warning: Could not write to TensorBoard: {e}")

    global model_path
    try:
        if model_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            model_path = f"models/mnist_cnn_{timestamp}.pth"
            print(f"No model path specified, saving to: {model_path}")

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
    except Exception as e:
        print(f"Warning: Could not save model: {e}")

    return results


def predict(img):
    model.eval()
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs.data, 1)
        return predicted

class DigitRecognizerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Hand written digit recognition")

        self.model_path = None
        self.is_model_loaded = False
        self.last_x, self.last_y = None, None

        self.digit = None

        self.canvas = tk.Canvas(master, width=560, height=560, bg="white", relief="ridge", borderwidth=2)
        self.canvas.grid(row=0, column=0, columnspan=4, pady=10)

        self.prediction_label = tk.Label(master, text="Prediction: None", font=("Arial", 14))
        self.prediction_label.grid(row=1, column=0, columnspan=4, pady=5)

        self.load_button = tk.Button(master, text="Load Model", command=self.load_model, bg="#4CAF50", fg="white")
        self.load_button.grid(row=2, column=0, padx=5, pady=5)

        self.train_button = tk.Button(master, text="Train model", command=self.train_model, bg="#008CBA", fg="white")
        self.train_button.grid(row=2, column=1, padx=5, pady=5)

        self.test_button = tk.Button(master, text="Test Model", command=self.run_test, bg="#FF9800", fg="white")
        self.test_button.grid(row=3, column=2, padx=5, pady=5)

        self.clear_button = tk.Button(master, text="Clear", command=self.clear_canvas, bg="#FF5733", fg="white")
        self.clear_button.grid(row=2, column=2, padx=5, pady=5)

        self.save_correct_button = tk.Button(master, text="Save as Correct", command=self.save_as_correct, bg="#4CAF50",
                                             fg="white")
        self.save_correct_button.grid(row=3, column=0, padx=5, pady=5)

        self.save_incorrect_button = tk.Button(master, text="Save as Incorrect", command=self.save_as_incorrect,
                                               bg="#f44336", fg="white")
        self.save_incorrect_button.grid(row=3, column=1, padx=5, pady=5)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_last_position)

        self.image = Image.new("L", (560, 560), color=0)
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        if self.is_model_loaded:
            x, y = event.x, event.y
            r = 10

            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")
            self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

            if self.last_x is not None and self.last_y is not None:
                self.canvas.create_line(self.last_x, self.last_y, x, y, fill="black", width=20, capstyle=tk.ROUND, smooth=tk.TRUE)
                self.draw.line([self.last_x, self.last_y, x, y], fill=255, width=2 * r)

            self.last_x, self.last_y = x, y
            self.predict_digit()

    def reset_last_position(self, event):
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.image = Image.new("L", (560, 560), color='white')
        self.canvas.delete("all")
        self.image = Image.new("L", (560, 560), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.digit = None
        self.prediction_label.config(text="Prediction: None")

    def predict_digit(self):
        img = self.image.resize((28, 28))
        img_tensor = inference_transform(img).unsqueeze(0).to(device)
        self.digit = predict(img_tensor).item()
        print(f"Predicted Digit: {self.digit}")
        self.prediction_label.config(text=f"Prediction: {self.digit}")

    def train_model(self):
        global writer, model_path
        if writer is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            logdir = f"tf_logs/{timestamp}"
            writer = SummaryWriter(log_dir=logdir)
            model_path = f"models/mnist_cnn_{timestamp}.pth"
            print(f"Created new TensorBoard logdir: {logdir}")

        train()
        test()
        self.load_model()

    def load_model(self):
        self.model_path = filedialog.askopenfilename(
            title="Select Model File",
        )
        if self.model_path:
            try:
                model.load_state_dict(torch.load(self.model_path, map_location=device))
                self.is_model_loaded = True

                model_timestamp = os.path.basename(self.model_path).split('_')[-1].split('.')[0]
                logdir = f"tf_logs/{model_timestamp}"

                global writer
                writer = SummaryWriter(log_dir=logdir)

                print(f"Model loaded from {self.model_path}")
                print(f"Continuing TensorBoard logging in {logdir}")
                messagebox.showinfo("Model Loaded",
                                    f"Model loaded successfully from:\n{self.model_path}\n"
                                    f"Continuing TensorBoard logging in:\n{logdir}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model:\n{e}")

    def save_image(self, folder, label):
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, f"{uuid.uuid4().hex}_{label}.png")
        self.image.save(filename)
        self.clear_canvas()

    def save_as_correct(self):
        self.save_image("correct", self.digit)

    def save_as_incorrect(self):
        self.save_image("incorrect", simpledialog.askstring(title= "Label", prompt="Enter Label"))

    def run_test(self):
        if not self.is_model_loaded:
            messagebox.showerror("Error", "Please load a model first!")
            return

        try:
            global model_path
            if model_path is None and self.model_path is not None:
                model_path = self.model_path

            results = test()
            messagebox.showinfo("Test Complete",
                                f"Model testing completed!\n"
                                f"Test Accuracy: {results['test_accuracy']:.2f}%\n"
                                f"Custom Test Accuracy: {results['custom_test_accuracy']:.2f}%")
        except Exception as e:
            messagebox.showerror("Test Error", f"Error during testing:\n{e}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.resizable(False, False)
    root.mainloop()