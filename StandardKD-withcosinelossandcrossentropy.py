import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

class StudentModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(StudentModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

class TeacherModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TeacherModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

def train_with_all_losses(
    teacher, student, train_loader, epochs, learning_rate,
    T, ce_loss_weight, kl_loss_weight, cosine_loss_weight, device
):
    ce_loss = nn.CrossEntropyLoss()
    cosine_similarity = nn.CosineSimilarity(dim=-1)
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.eval()
    student.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits = teacher(inputs)

            student_logits = student(inputs)

            soft_targets = F.softmax(teacher_logits / T, dim=-1)
            student_log_probs = F.log_softmax(student_logits / T, dim=-1)
            kl_loss = F.kl_div(student_log_probs, soft_targets, reduction="batchmean") * (T**2)

            ce = ce_loss(student_logits, labels)

            cosine_sim = cosine_similarity(student_logits, teacher_logits)
            cosine_loss = 1 - cosine_sim.mean()

            loss = (
                ce_loss_weight * ce +
                kl_loss_weight * kl_loss +
                cosine_loss_weight * cosine_loss
            )

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

input_dim = 768
output_dim = 4
batch_size = 32
num_samples = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train = torch.randn(num_samples, input_dim)
y_train = torch.randint(0, output_dim, (num_samples,))
X_test = torch.randn(int(num_samples * 0.2), input_dim)
y_test = torch.randint(0, output_dim, (int(num_samples * 0.2),))

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

nn_deep = TeacherModel(input_dim, output_dim).to(device)
new_nn_light = StudentModel(input_dim, output_dim).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(new_nn_light.parameters(), lr=0.001)
new_nn_light.train()
for epoch in range(5):
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = new_nn_light(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

test_accuracy_light_ce = test(new_nn_light, test_loader, device)

train_with_all_losses(
    teacher=nn_deep,
    student=new_nn_light,
    train_loader=train_loader,
    epochs=10,
    learning_rate=0.001,
    T=2,
    ce_loss_weight=0.5,
    kl_loss_weight=0.4,
    cosine_loss_weight=0.1,
    device=device
)

test_accuracy_deep = test(nn_deep, test_loader, device)
test_accuracy_light_all = test(new_nn_light, test_loader, device)

print(f"Teacher accuracy: {test_accuracy_deep:.2f}%")
print(f"Student accuracy (CE only): {test_accuracy_light_ce:.2f}%")
print(f"Student accuracy (CE + KL + Cosine): {test_accuracy_light_all:.2f}%")
