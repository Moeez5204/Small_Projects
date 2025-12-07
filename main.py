import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class ProjectSelector:
    """Main class to select and run different PyTorch projects"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def show_menu(self):
        """Display available projects"""
        print("\n" + "=" * 50)
        print("PYTORCH PROJECT SELECTOR")
        print("=" * 50)
        print("1. Binary Classification (Synthetic Data)")
        print("2. Multi-Class Classification (MNIST)")
        print("3. Regression Model")
        print("4. Simple CNN (Fashion-MNIST)")
        print("5. Autoencoder (Image Denoising)")
        print("6. All Projects (Run sequentially)")
        print("0. Exit")
        print("=" * 50)

    def run(self):
        """Main execution loop"""
        while True:
            self.show_menu()
            try:
                choice = input("\nEnter your choice (0-6): ").strip()

                if choice == '0':
                    print("Exiting... Goodbye!")
                    break
                elif choice == '1':
                    self.run_binary_classification()
                elif choice == '2':
                    self.run_mnist_classification()
                elif choice == '3':
                    self.run_regression()
                elif choice == '4':
                    self.run_fashion_cnn()
                elif choice == '5':
                    self.run_autoencoder()
                elif choice == '6':
                    self.run_all_projects()
                else:
                    print("Invalid choice! Please enter 0-6")
            except KeyboardInterrupt:
                print("\nInterrupted by user")
                break
            except Exception as e:
                print(f"Error: {e}")

    def run_all_projects(self):
        """Run all projects sequentially"""
        print("\nRunning all projects sequentially...")
        projects = [
            ("Binary Classification", self.run_binary_classification),
            ("MNIST Classification", self.run_mnist_classification),
            ("Regression", self.run_regression),
            ("Fashion CNN", self.run_fashion_cnn),
            ("Autoencoder", self.run_autoencoder)
        ]

        for name, func in projects:
            print(f"\n{'=' * 60}")
            print(f"PROJECT: {name}")
            print('=' * 60)
            try:
                func()
            except Exception as e:
                print(f"Error in {name}: {e}")

    # ========== PROJECT 1: Binary Classification ==========
    def run_binary_classification(self):
        """Binary classification with synthetic data"""
        print("\n" + "=" * 40)
        print("PROJECT 1: Binary Classification")
        print("=" * 40)

        # Generate synthetic data
        X, y = make_classification(
            n_samples=1000, n_features=20, n_informative=15,
            n_redundant=5, random_state=42
        )
        y = y.reshape(-1, 1)

        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_test_t = torch.FloatTensor(X_test).to(self.device)
        y_test_t = torch.FloatTensor(y_test).to(self.device)

        # Create datasets
        train_dataset = TensorDataset(X_train_t, y_train_t)
        test_dataset = TensorDataset(X_test_t, y_test_t)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32)

        # Define model
        class BinaryClassifier(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                    nn.Sigmoid()
                )

            def forward(self, x):
                return self.net(x)

        model = BinaryClassifier(X_train.shape[1]).to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training
        print("Training binary classifier...")
        n_epochs = 50
        train_losses = []

        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            train_losses.append(epoch_loss / len(train_loader))
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {train_losses[-1]:.4f}")

        # Evaluation
        model.eval()
        with torch.no_grad():
            predictions = model(X_test_t)
            predictions = (predictions > 0.5).float()
            accuracy = (predictions == y_test_t).float().mean()

        print(f"\nTest Accuracy: {accuracy.item():.4f}")

        # Plot results
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.subplot(1, 2, 2)
        plt.scatter(range(len(y_test[:50])), y_test[:50].flatten(),
                    alpha=0.5, label='True', s=100)
        plt.scatter(range(len(predictions[:50])), predictions[:50].cpu().numpy().flatten(),
                    alpha=0.5, label='Predicted', marker='x', s=100)
        plt.title('Predictions vs True (First 50 samples)')
        plt.xlabel('Sample')
        plt.ylabel('Class')
        plt.legend()
        plt.tight_layout()
        plt.savefig('binary_classification_results.png')
        plt.show()

        print("\nResults saved to 'binary_classification_results.png'")

    # ========== PROJECT 2: MNIST Classification ==========
    def run_mnist_classification(self):
        """MNIST digit classification"""
        print("\n" + "=" * 40)
        print("PROJECT 2: MNIST Classification")
        print("=" * 40)

        # Data loading
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)

        # Use subsets for faster training
        train_indices = torch.randperm(len(train_dataset))[:3000]
        test_indices = torch.randperm(len(test_dataset))[:1000]

        train_subset = torch.utils.data.Subset(train_dataset, train_indices)
        test_subset = torch.utils.data.Subset(test_dataset, test_indices)

        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=64)

        # Define model
        class MNISTNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(28 * 28, 128)
                self.fc2 = nn.Linear(128, 64)
                self.fc3 = nn.Linear(64, 10)
                self.dropout = nn.Dropout(0.25)

            def forward(self, x):
                x = x.view(-1, 28 * 28)
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = F.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return x

        model = MNISTNet().to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training
        print("Training MNIST classifier...")
        n_epochs = 15
        train_losses = []
        train_accs = []

        for epoch in range(n_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100. * correct / total
            train_losses.append(epoch_loss)
            train_accs.append(epoch_acc)

            print(f"Epoch {epoch + 1}/{n_epochs}: Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")

        # Evaluation
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        test_loss /= len(test_loader)
        test_acc = 100. * correct / total

        print(f"\nTest Results: Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")

        # Visualize some predictions
        data_iter = iter(test_loader)
        images, labels = next(data_iter)
        images, labels = images.to(self.device), labels.to(self.device)

        with torch.no_grad():
            outputs = model(images)
            _, preds = outputs.max(1)

        # Plot results
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 3, 1)
        plt.plot(train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')

        plt.subplot(1, 3, 2)
        plt.plot(train_accs)
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')

        plt.subplot(1, 3, 3)
        images_np = images.cpu().numpy()
        for i in range(6):
            plt.subplot(2, 3, i + 1)
            plt.imshow(images_np[i][0], cmap='gray')
            plt.title(f'True: {labels[i].item()}\nPred: {preds[i].item()}')
            plt.axis('off')

        plt.tight_layout()
        plt.savefig('mnist_results.png')
        plt.show()

        print("\nResults saved to 'mnist_results.png'")

    # ========== PROJECT 3: Regression ==========
    def run_regression(self):
        """Regression model with synthetic data"""
        print("\n" + "=" * 40)
        print("PROJECT 3: Regression Model")
        print("=" * 40)

        # Generate synthetic regression data
        X, y = make_regression(
            n_samples=1000, n_features=10, n_informative=8,
            noise=20, random_state=42
        )
        y = y.reshape(-1, 1)

        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)
        y_train = scaler_y.fit_transform(y_train)
        y_test = scaler_y.transform(y_test)

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_test_t = torch.FloatTensor(X_test).to(self.device)
        y_test_t = torch.FloatTensor(y_test).to(self.device)

        # Define model
        class RegressionNet(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Linear(16, 1)
                )

            def forward(self, x):
                return self.net(x)

        model = RegressionNet(X_train.shape[1]).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training
        print("Training regression model...")
        n_epochs = 100
        train_losses = []

        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()
            predictions = model(X_train_t)
            loss = criterion(predictions, y_train_t)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.4f}")

        # Evaluation
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_t)
            test_loss = criterion(y_pred, y_test_t)

            # Inverse transform for original scale
            y_pred_orig = scaler_y.inverse_transform(y_pred.cpu().numpy())
            y_test_orig = scaler_y.inverse_transform(y_test)

            mae = np.mean(np.abs(y_pred_orig - y_test_orig))
            r2 = 1 - (np.sum((y_test_orig - y_pred_orig) ** 2) /
                      np.sum((y_test_orig - np.mean(y_test_orig)) ** 2))

        print(f"\nTest MSE: {test_loss.item():.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")

        # Plot results
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(train_losses)
        plt.title('Training Loss (MSE)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.subplot(1, 3, 2)
        plt.scatter(y_test_orig[:100], y_pred_orig[:100], alpha=0.6)
        plt.plot([y_test_orig.min(), y_test_orig.max()],
                 [y_test_orig.min(), y_test_orig.max()], 'r--', lw=2)
        plt.title('Predictions vs True Values')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')

        plt.subplot(1, 3, 3)
        plt.plot(y_test_orig[:50], 'b-', label='True', alpha=0.7)
        plt.plot(y_pred_orig[:50], 'r--', label='Predicted', alpha=0.7)
        plt.title('Sample Predictions')
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.legend()

        plt.tight_layout()
        plt.savefig('regression_results.png')
        plt.show()

        print("\nResults saved to 'regression_results.png'")

    # ========== PROJECT 4: Simple CNN (Fashion-MNIST) ==========
    def run_fashion_cnn(self):
        """CNN for Fashion-MNIST"""
        print("\n" + "=" * 40)
        print("PROJECT 4: Fashion-MNIST CNN")
        print("=" * 40)

        # Data loading
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST('./data', train=False, transform=transform)

        # Use subsets
        train_indices = torch.randperm(len(train_dataset))[:5000]
        test_indices = torch.randperm(len(test_dataset))[:1000]

        train_subset = torch.utils.data.Subset(train_dataset, train_indices)
        test_subset = torch.utils.data.Subset(test_dataset, test_indices)

        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=64)

        # Fashion-MNIST classes
        classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        # Define CNN
        class FashionCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.fc1 = nn.Linear(64 * 7 * 7, 128)
                self.fc2 = nn.Linear(128, 10)
                self.dropout = nn.Dropout(0.25)

            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = x.view(-1, 64 * 7 * 7)
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x

        model = FashionCNN().to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training
        print("Training Fashion CNN...")
        n_epochs = 10
        train_losses = []

        for epoch in range(n_epochs):
            model.train()
            running_loss = 0.0

            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            train_losses.append(avg_loss)
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")

        # Evaluation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        accuracy = 100. * correct / total
        print(f"\nTest Accuracy: {accuracy:.2f}%")

        # Visualize predictions
        data_iter = iter(test_loader)
        images, labels = next(data_iter)
        images, labels = images.to(self.device), labels.to(self.device)

        with torch.no_grad():
            outputs = model(images)
            _, preds = outputs.max(1)

        # Plot results
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 3, 1)
        plt.plot(train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')

        plt.subplot(1, 3, 2)
        plt.bar(range(10), [100 * (preds == i).sum().item() / len(preds)
                            for i in range(10)])
        plt.title('Class Distribution in Predictions')
        plt.xlabel('Class')
        plt.ylabel('Percentage (%)')

        plt.subplot(1, 3, 3)
        for i in range(6):
            plt.subplot(2, 3, i + 1)
            plt.imshow(images[i].cpu().numpy().squeeze(), cmap='gray')
            true_label = classes[labels[i].item()]
            pred_label = classes[preds[i].item()]
            color = 'green' if labels[i] == preds[i] else 'red'
            plt.title(f'True: {true_label}\nPred: {pred_label}', color=color)
            plt.axis('off')

        plt.tight_layout()
        plt.savefig('fashion_cnn_results.png')
        plt.show()

        print("\nResults saved to 'fashion_cnn_results.png'")

    # ========== PROJECT 5: Autoencoder ==========
    def run_autoencoder(self):
        """Autoencoder for MNIST digit denoising"""
        print("\n" + "=" * 40)
        print("PROJECT 5: Autoencoder (Image Denoising)")
        print("=" * 40)

        # Data loading with noise
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)

        # Use subsets
        train_indices = torch.randperm(len(train_dataset))[:3000]
        test_indices = torch.randperm(len(test_dataset))[:500]

        train_subset = torch.utils.data.Subset(train_dataset, train_indices)
        test_subset = torch.utils.data.Subset(test_dataset, test_indices)

        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=16)

        # Define Autoencoder
        class Autoencoder(nn.Module):
            def __init__(self):
                super().__init__()
                # Encoder
                self.encoder = nn.Sequential(
                    nn.Linear(28 * 28, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 16),  # Bottleneck
                )
                # Decoder
                self.decoder = nn.Sequential(
                    nn.Linear(16, 32),
                    nn.ReLU(),
                    nn.Linear(32, 64),
                    nn.ReLU(),
                    nn.Linear(64, 128),
                    nn.ReLU(),
                    nn.Linear(128, 28 * 28),
                    nn.Sigmoid()  # Output between 0 and 1
                )

            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded

        model = Autoencoder().to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Add noise function
        def add_noise(images, noise_factor=0.3):
            noisy = images + noise_factor * torch.randn_like(images)
            return torch.clamp(noisy, 0., 1.)

        # Training
        print("Training autoencoder...")
        n_epochs = 20
        train_losses = []

        for epoch in range(n_epochs):
            model.train()
            running_loss = 0.0

            for data, _ in train_loader:
                data = data.view(-1, 28 * 28).to(self.device)
                noisy_data = add_noise(data)

                optimizer.zero_grad()
                reconstructed = model(noisy_data)
                loss = criterion(reconstructed, data)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            train_losses.append(avg_loss)
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")

        # Test with visualization
        model.eval()
        data_iter = iter(test_loader)
        images, _ = next(data_iter)
        images = images.view(-1, 28 * 28).to(self.device)

        with torch.no_grad():
            noisy_images = add_noise(images)
            reconstructed = model(noisy_images)

        # Plot results
        plt.figure(figsize=(15, 4))

        plt.subplot(1, 4, 1)
        plt.plot(train_losses)
        plt.title('Training Loss (MSE)')
        plt.xlabel('Epoch')

        # Show original, noisy and reconstructed images
        n_images = 3
        for i in range(n_images):
            plt.subplot(3, n_images, i + 1)
            plt.imshow(images[i].cpu().reshape(28, 28), cmap='gray')
            plt.title('Original')
            plt.axis('off')

            plt.subplot(3, n_images, i + 1 + n_images)
            plt.imshow(noisy_images[i].cpu().reshape(28, 28), cmap='gray')
            plt.title('Noisy')
            plt.axis('off')

            plt.subplot(3, n_images, i + 1 + 2 * n_images)
            plt.imshow(reconstructed[i].cpu().reshape(28, 28), cmap='gray')
            plt.title('Reconstructed')
            plt.axis('off')

        plt.tight_layout()
        plt.savefig('autoencoder_results.png')
        plt.show()

        print("\nResults saved to 'autoencoder_results.png'")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='PyTorch Project Selector')
    parser.add_argument('--project', type=int, choices=range(0, 7),
                        help='Run specific project (0-6, see menu for details)')

    args = parser.parse_args()

    selector = ProjectSelector()

    if args.project is not None:
        # Run specific project
        if args.project == 0:
            print("Exiting...")
            return
        elif args.project == 1:
            selector.run_binary_classification()
        elif args.project == 2:
            selector.run_mnist_classification()
        elif args.project == 3:
            selector.run_regression()
        elif args.project == 4:
            selector.run_fashion_cnn()
        elif args.project == 5:
            selector.run_autoencoder()
        elif args.project == 6:
            selector.run_all_projects()
    else:
        # Run interactive menu
        selector.run()


if __name__ == "__main__":
    main()