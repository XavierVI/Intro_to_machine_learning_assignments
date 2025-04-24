import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import numpy as np

from sklearn.model_selection import KFold



def train_model(
    model,
    train_loader,
    test_loader,
    device,
    num_epochs,
    learning_rate,
    l2_reg
):
    model.train()
    # instantiate optimizer and loss function
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=l2_reg  # L2 regularization
    )
    test_accuracy = 0

    # Training loop
    for epoch in range(num_epochs):
        # accumulators
        avg_loss = 0
        correct = 0
        total = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            # forward pass
            predictions = model(X)
            # compute the loss
            loss = loss_fn(predictions, y)
            avg_loss += loss.item()

            # backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # compute the accuracy
            probabilities = torch.nn.functional.softmax(predictions, dim=1)
            y_hat = torch.argmax(probabilities, dim=1)
            correct += (y_hat == y).sum().item()
            total += y.size(0)
            # print(f'Batch Loss: {loss}')
            # print(f'Avg. Accuracy: {correct * 100 / total}')

        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(
            '============================================================================')
        print(f'Avg. Loss: {avg_loss/len(train_loader):.4f}')
        print(f'Avg. Accuracy: {correct*100/total:.4f}')
        test_accuracy = get_test_accuracy(model, test_loader, device)
        print(f'Test accuracy: {test_accuracy*100:.4f}')

        # if test_accuracy >= 0.9:
        #     print('Reached desired accuracy early, exiting training loop')
        #     break

    print('Training finished!')

    # return the final test accuracy and time cost
    return test_accuracy


def get_test_accuracy(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            predictions = model(X)
            probabilities = torch.nn.functional.softmax(predictions, dim=1)
            # compute the accuracy
            y_hat = torch.argmax(probabilities, dim=1)
            correct += (y_hat == y).sum().item()
            total += y.size(0)

    model.train()
    return correct / total

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Functions for cross-validation
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def reset_weights(m):
  '''
    Resets the model's weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()



def perform_kfold(
        dataset,
        model_class,
        input_size,
        k_folds=5,
        num_epochs=50,
        batch_size=64,
        learning_rate=1e-5,
        l2_reg=0.0001):

  # For fold results
  results = {}

  # Set fixed random number seed
  torch.manual_seed(42)

  # Define the K-fold Cross Validator
  kfold = KFold(n_splits=k_folds, shuffle=True)


  # Start print
  print('--------------------------------')

  # K-fold Cross Validation model evaluation
  for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):

    # Print
    print(f'FOLD {fold}')
    print('--------------------------------')

    # set device to be a CUDA device if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_dataset = Subset(dataset, train_ids)
    test_dataset = Subset(dataset, test_ids)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        pin_memory=True
    )

    # instantiate the model and load it on the device
    # model = FNN(X.size(1))
    # model.to(device)
    # model.apply(reset_weights)
    model = model_class(input_size)
    model.to(device)
    model.apply(reset_weights)

    train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        l2_reg=l2_reg
    )

    # Process is complete.
    print('Training process has finished. Saving trained model.')

    # Print about testing
    print('Starting testing')

    # Evaluation for this fold
    correct, total = 0, 0
    with torch.no_grad():

      # Iterate over the test data and generate predictions
      for i, data in enumerate(test_loader, 0):

        # Get inputs
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)

        # Generate outputs
        outputs = model(inputs)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        # Set total and correct
        predicted = torch.argmax(probabilities, dim=1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

      # Print accuracy
      accuracy = 100.0 * correct / total
      print('Accuracy for fold %d: %d %%' % (fold, accuracy))
      print('--------------------------------')
      results[fold] = accuracy

  # Print fold results
  total_accuracy = 0.0
  for value in results.values():
    total_accuracy += value
  avg_accuracy = total_accuracy / len(results)

  print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
  print('--------------------------------')
  for key, value in results.items():
    print(f'Fold {key}: {value} %')
  print(f'Average: {avg_accuracy} %')

  return results, avg_accuracy


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Functions for bagging
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def create_bootstrap(n_samples, data_tensor, label_tensor):
    """
    Returns a subset of the dataset using bagging with replacement.
    """
    indices = np.random.default_rng(1).choice(
        n_samples, size=n_samples, replace=True)
    return data_tensor[indices], label_tensor[indices]



