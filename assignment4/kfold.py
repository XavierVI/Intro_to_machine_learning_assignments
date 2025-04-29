import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import KFold
from fnn import FNN, train_model, load_movie_reviews

def reset_weights(m):
  '''
    Try resetting model weights to avoid
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

  # X, y = load_movie_reviews('./movie_data.csv', dataset_size, max_features)
  # dataset = TensorDataset(X, y)
    
  # Start print
  print('--------------------------------')

  # K-fold Cross Validation model evaluation
  for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
    
    # Print
    print(f'FOLD {fold}')
    print('--------------------------------')

    #### set device to be a CUDA device if available
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

    #### instantiate the model and load it on the device
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

        # Set total and correct
        _, predicted = torch.max(outputs.data, 1)
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


if __name__ == '__main__':
  # Load your data
  dataset_size = 50_000
  max_features = 20_000
  X, y = load_movie_reviews('../movie_data.csv', dataset_size, max_features)
  dataset = TensorDataset(X, y)

  # Call the k-fold cross validation function
  results, avg_accuracy = perform_kfold(
    dataset=dataset,
    model_class=FNN,
    input_size=X.size(1),
    k_folds=5,
    num_epochs=50,
    batch_size=64
  )

